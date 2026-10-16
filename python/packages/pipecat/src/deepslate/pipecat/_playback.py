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

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_BYTES_PER_SAMPLE = 2


@dataclass
class _Chunk:
    turn_id: Optional[int]
    end: int
    remaining: int


class _PlaybackTracker:
    """Maps audio written by the output transport back to turn byte offsets."""

    def __init__(self, report: Callable[[int, int], None]):
        self._report = report
        self._chunks: Deque[_Chunk] = deque()
        self._turn_bytes: Dict[int, int] = {}
        self._wire_rate = 0
        self._wire_frame_bytes = 0
        self._sink_samples = 0

    def received(self, turn_id: Optional[int], nbytes: int, sample_rate: int, channels: int) -> None:
        self._wire_rate = sample_rate
        self._wire_frame_bytes = _BYTES_PER_SAMPLE * channels
        end = nbytes
        if turn_id is not None:
            end += self._turn_bytes.get(turn_id, 0)
            self._turn_bytes[turn_id] = end
        self._chunks.append(_Chunk(turn_id, end, nbytes))

    def played(self, samples: int, sample_rate: int) -> None:
        before = self._wire_bytes(sample_rate)
        self._sink_samples += samples
        budget = self._wire_bytes(sample_rate) - before
        positions: Dict[int, int] = {}
        while budget and self._chunks:
            chunk = self._chunks[0]
            taken = min(budget, chunk.remaining)
            chunk.remaining -= taken
            budget -= taken
            if chunk.turn_id is not None:
                positions[chunk.turn_id] = chunk.end - chunk.remaining
            if not chunk.remaining:
                self._chunks.popleft()
        for turn_id, position in positions.items():
            self._report(position, turn_id)

    def discard(self) -> None:
        self._chunks.clear()

    def forget_turns(self) -> None:
        for chunk in self._chunks:
            chunk.turn_id = None
        self._turn_bytes.clear()

    def _wire_bytes(self, sink_rate: int) -> int:
        return self._sink_samples * self._wire_rate // sink_rate * self._wire_frame_bytes


class DeepslatePlaybackReporter(FrameProcessor):
    """Reports how much assistant audio the output transport has written.

    Place it right after ``transport.output()``; create it with
    ``DeepslateRealtimeLLMService.playback_reporter()``.
    """

    def __init__(self, tracker: _PlaybackTracker, **kwargs):
        super().__init__(**kwargs)
        self._tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            samples = len(frame.audio) // (_BYTES_PER_SAMPLE * frame.num_channels)
            self._tracker.played(samples, frame.sample_rate)
        elif isinstance(frame, InterruptionFrame):
            self._tracker.discard()
        await self.push_frame(frame, direction)
