# Copyright 2026 Deepslate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

import aiohttp

from .options import DeepslateOptions
from ._utils import build_ws_url

logger = logging.getLogger("deepslate.core")

_RETRIABLE_4XX = frozenset({408, 429})


class HandshakeRejectedError(Exception):
    """The Deepslate server permanently rejected the WebSocket handshake.

    Raised for 4xx responses other than 408/429 (e.g. bad credentials, an
    unknown vendor/organization, or a model the organization cannot use).
    """

    def __init__(self, status: int, model: Optional[str] = None) -> None:
        super().__init__(_handshake_rejected_message(status, model))
        self.status = status
        self.model = model


def _handshake_rejected_message(status: int, model: Optional[str]) -> str:
    prefix = f"Deepslate rejected the WebSocket handshake (HTTP {status})"
    if status == 401:
        return f"{prefix}: check DEEPSLATE_API_KEY / api_key."
    if status in (403, 404):
        if model:
            return (
                f"{prefix} for model '{model}': check vendor_id / organization_id, "
                "or the organization may not have access to this model."
            )
        return f"{prefix}: check vendor_id / organization_id."
    return f"{prefix}."


class BaseDeepslateClient:
    """Manages WebSocket connectivity to the Deepslate Realtime API.

    Handles URL construction, authentication headers, HTTP session
    lifecycle, and exponential-backoff reconnection. Both
    ``deepslate-livekit`` and ``deepslate-pipecat`` use this class via
    composition so that all transport logic lives in one place.
    """

    def __init__(
        self,
        opts: DeepslateOptions,
        user_agent: str,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._opts = opts
        self._user_agent = user_agent
        # If a session is injected we don't own it and won't close it.
        self._http_session = http_session
        self._http_session_owned = http_session is None

    @property
    def user_agent(self) -> str:
        """The User-Agent string sent on realtime connections."""
        return self._user_agent

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _build_ws_url(self) -> str:
        if self._opts.ws_url:
            return self._opts.ws_url
        return build_ws_url(
            self._opts.base_url,
            self._opts.vendor_id,
            self._opts.organization_id,
            self._opts.model,
        )

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"User-Agent": self._user_agent}
        if self._opts.api_key:
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        return headers

    async def connect(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection to Deepslate and return it.

        Raises:
            HandshakeRejectedError: the server answered the handshake with a
                non-retriable 4xx status.
            aiohttp.ClientError: any other (retriable) connection failure.
        """
        url = self._build_ws_url()
        headers = self._build_headers()
        logger.debug(f"connecting to Deepslate: {url}")
        try:
            return await self._ensure_http_session().ws_connect(url=url, headers=headers)
        except aiohttp.WSServerHandshakeError as e:
            if 400 <= e.status < 500 and e.status not in _RETRIABLE_4XX:
                raise HandshakeRejectedError(e.status, self._opts.model) from e
            raise

    async def run_with_retry(
        self,
        run_session: Callable[[aiohttp.ClientWebSocketResponse], Awaitable[None]],
        *,
        should_continue: Callable[[], bool],
        on_fatal_error: Callable[[Exception], Awaitable[None]],
        on_connect_attempt: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Connect and run ``run_session`` with exponential-backoff retries.

        ``run_session`` receives the open ``ClientWebSocketResponse`` and
        should block until the connection ends (cleanly or otherwise).

        On a retriable ``aiohttp.ClientError``, the loop waits and
        reconnects.  Once ``max_retries`` is exceeded, on a
        ``HandshakeRejectedError``, or on any unexpected exception, ``on_fatal_error`` is called and the loop
        exits.  ``should_continue`` is checked before every attempt so the
        caller can stop the loop externally.  ``on_connect_attempt``, if
        given, is awaited immediately before each dial (not before the
        backoff sleep), so callers can time each individual connection
        attempt rather than just the first one.
        """
        num_retries = 0
        max_retries = self._opts.max_retries

        while should_continue():
            try:
                if on_connect_attempt is not None:
                    await on_connect_attempt()
                ws = await self.connect()
                await run_session(ws)
                num_retries = 0  # reset on clean exit
            except aiohttp.ClientError as e:
                if num_retries >= max_retries:
                    logger.error(f"connection failed after {num_retries} retries: {e}")
                    await on_fatal_error(e)
                    return
                num_retries += 1
                retry_interval = min(2**num_retries, 30)
                logger.warning(
                    f"connection failed (attempt {num_retries}/{max_retries}), "
                    f"retrying in {retry_interval}s: {e}"
                )
                await asyncio.sleep(retry_interval)
            except HandshakeRejectedError as e:
                logger.error(f"connection rejected: {e}")
                await on_fatal_error(e)
                return
            except Exception as e:
                logger.error(f"unexpected error in Deepslate session: {e}")
                await on_fatal_error(e)
                return

    async def aclose(self) -> None:
        """Close the HTTP session if this client owns it."""
        if self._http_session_owned and self._http_session is not None:
            await self._http_session.close()
            self._http_session = None
