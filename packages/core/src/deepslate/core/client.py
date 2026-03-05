from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

import aiohttp

from .options import DeepslateOptions
from ._utils import build_ws_url

logger = logging.getLogger("deepslate.core")


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
        )

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"User-Agent": self._user_agent}
        if self._opts.api_key:
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        return headers

    async def connect(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket connection to Deepslate and return it."""
        url = self._build_ws_url()
        headers = self._build_headers()
        logger.debug(f"connecting to Deepslate: {url}")
        return await self._ensure_http_session().ws_connect(url=url, headers=headers)

    async def run_with_retry(
        self,
        run_session: Callable[[aiohttp.ClientWebSocketResponse], Awaitable[None]],
        *,
        should_continue: Callable[[], bool],
        on_fatal_error: Callable[[Exception], Awaitable[None]],
    ) -> None:
        """Connect and run ``run_session`` with exponential-backoff retries.

        ``run_session`` receives the open ``ClientWebSocketResponse`` and
        should block until the connection ends (cleanly or otherwise).

        On a retriable ``aiohttp.ClientError``, the loop waits and
        reconnects.  Once ``max_retries`` is exceeded, or on any
        unexpected exception, ``on_fatal_error`` is called and the loop
        exits.  ``should_continue`` is checked before every attempt so the
        caller can stop the loop externally.
        """
        num_retries = 0
        max_retries = self._opts.max_retries

        while should_continue():
            try:
                ws = await self.connect()
                await run_session(ws)
                num_retries = 0  # reset on clean exit
            except aiohttp.ClientError as e:
                if num_retries >= max_retries:
                    logger.error(f"connection failed after {num_retries} retries: {e}")
                    await on_fatal_error(e)
                    return
                num_retries += 1
                retry_interval = min(2 ** num_retries, 30)
                logger.warning(
                    f"connection failed (attempt {num_retries}/{max_retries}), "
                    f"retrying in {retry_interval}s: {e}"
                )
                await asyncio.sleep(retry_interval)
            except Exception as e:
                logger.error(f"unexpected error in Deepslate session: {e}")
                await on_fatal_error(e)
                return

    async def aclose(self) -> None:
        """Close the HTTP session if this client owns it."""
        if self._http_session_owned and self._http_session is not None:
            await self._http_session.close()
            self._http_session = None