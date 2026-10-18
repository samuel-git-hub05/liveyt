"""YouTube/microphone audio ingestion, Silero phrase segmentation, and extraction."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioFrame:
    start_sample: int
    pcm_s16le: bytes
    sample_rate: int = 16_000

    @property
    def sample_count(self) -> int:
        return len(self.pcm_s16le) // 2

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start_sample: int
    end_sample: int
    pcm_s16le: bytes
    sample_rate: int = 16_000

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


class AudioSource(Protocol):
    def frames(self) -> AsyncIterator[AudioFrame]: ...


class YoutubeAudioSource:
    """Run yt-dlp -> ffmpeg and yield 100 ms mono 16 kHz signed-PCM frames."""

    def __init__(
        self,
        url: str,
        *,
        yt_dlp: str = "yt-dlp",
        ffmpeg: str = "ffmpeg",
        sample_rate: int = 16_000,
        frame_ms: int = 100,
        realtime: bool = True,
    ) -> None:
        self.url = url
        self.yt_dlp = yt_dlp
        self.ffmpeg = ffmpeg
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.realtime = realtime

    async def frames(self) -> AsyncIterator[AudioFrame]:
        if shutil.which(self.yt_dlp) is None:
            raise RuntimeError("yt-dlp was not found. Install it with: py -m pip install yt-dlp")
        if shutil.which(self.ffmpeg) is None:
            raise RuntimeError(f"ffmpeg executable {self.ffmpeg!r} was not found")
        downloader = await asyncio.create_subprocess_exec(
            self.yt_dlp,
            "--no-warnings",
            "--no-progress",
            "--quiet",
            "--format",
            "bestaudio/best",
            "--output",
            "-",
            self.url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if downloader.stdout is None:
            raise RuntimeError("Could not create yt-dlp output pipe")
        decoder = await asyncio.create_subprocess_exec(
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if decoder.stdin is None or decoder.stdout is None:
            downloader.terminate()
            decoder.terminate()
            raise RuntimeError("Could not create ffmpeg audio pipes")

        async def pump_download() -> None:
            try:
                while data := await downloader.stdout.read(64 * 1024):
                    decoder.stdin.write(data)
                    await decoder.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("ffmpeg closed its input pipe")
            finally:
                if not decoder.stdin.is_closing():
                    decoder.stdin.close()
                    try:
                        await decoder.stdin.wait_closed()
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        pump_task = asyncio.create_task(pump_download(), name="yt-dlp-to-ffmpeg")
        samples_per_frame = max(1, self.sample_rate * self.frame_ms // 1000)
        bytes_per_frame = samples_per_frame * 2
        pending = bytearray()
        cursor = 0
        clock_origin = time.monotonic()
        try:
            while data := await decoder.stdout.read(bytes_per_frame):
                pending.extend(data)
                while len(pending) >= bytes_per_frame:
                    chunk = bytes(pending[:bytes_per_frame])
                    del pending[:bytes_per_frame]
                    yield AudioFrame(cursor, chunk, self.sample_rate)
                    cursor += samples_per_frame
                    if self.realtime:
                        due_at = clock_origin + cursor / self.sample_rate
                        delay = due_at - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
            if pending:
                aligned = bytes(pending[: len(pending) - len(pending) % 2])
                if aligned:
                    yield AudioFrame(cursor, aligned, self.sample_rate)
            await pump_task
            decoder_code = await decoder.wait()
            downloader_code = await downloader.wait()
            if downloader_code not in (0, None):
                raise RuntimeError(
                    f"yt-dlp exited with status {downloader_code}; check the URL or YouTube access"
                )
            if decoder_code not in (0, None):
                raise RuntimeError(f"ffmpeg exited with status {decoder_code} while decoding YouTube audio")
        finally:
            if not pump_task.done():
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)
            for process in (decoder, downloader):
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=3)
                    except TimeoutError:
                        process.kill()
                        await process.wait()


class DirectMicAudioSource:
    """Capture a Windows/PortAudio input with PyAudio and resample to 16 kHz."""

    def __init__(self, device: str | int | None = None, *, sample_rate: int = 16_000) -> None:
        self.device = device
        self.sample_rate = sample_rate

    async def frames(self) -> AsyncIterator[AudioFrame]:
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("Install PyAudio for direct microphone mode: pip install pyaudio") from exc

        pa = pyaudio.PyAudio()
        stream = None
        try:
            device_index = self._resolve_device(pa, self.device)
            if device_index is None:
                device_info = pa.get_default_input_device_info()
            else:
                device_info = pa.get_device_info_by_index(device_index)
            native_rate = self.sample_rate
            try:
                pa.is_format_supported(
                    self.sample_rate,
                    input_device=device_index,
                    input_channels=1,
                    input_format=pyaudio.paInt16,
                )
            except (ValueError, OSError):
                native_rate = int(device_info.get("defaultSampleRate", self.sample_rate))
                try:
                    pa.is_format_supported(
                        native_rate,
                        input_device=device_index,
                        input_channels=1,
                        input_format=pyaudio.paInt16,
                    )
                except (ValueError, OSError) as exc:
                    raise RuntimeError(
                        f"Microphone {device_info.get('name', '')!r} does not support mono PCM capture"
                    ) from exc
            frames_per_buffer = max(1, native_rate // 10)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=native_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=frames_per_buffer,
            )
            logger.info(
                "Capturing microphone %s at %d Hz (output normalized to %d Hz)",
                device_info.get("name", "default input"),
                native_rate,
                self.sample_rate,
            )
            cursor = 0
            while True:
                native_pcm = await asyncio.to_thread(
                    stream.read, frames_per_buffer, exception_on_overflow=False
                )
                native = np.frombuffer(native_pcm, dtype="<i2").astype(np.float32) / 32768.0
                converted = (
                    native
                    if native_rate == self.sample_rate
                    else _resample(native, native_rate, self.sample_rate)
                )
                output_pcm = (np.clip(converted, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                if output_pcm:
                    yield AudioFrame(cursor, output_pcm, self.sample_rate)
                    cursor += len(output_pcm) // 2
        finally:
            if stream is not None:
                try:
                    await asyncio.to_thread(stream.stop_stream)
                finally:
                    await asyncio.to_thread(stream.close)
            pa.terminate()

    @staticmethod
    def _resolve_device(pa: object, reference: str | int | None) -> int | None:
        if reference is None:
            return None
        if isinstance(reference, int) or str(reference).isdigit():
            return int(reference)
        needle = str(reference).casefold()
        matches = []
        for index in range(pa.get_device_count()):  # type: ignore[attr-defined]
            device = pa.get_device_info_by_index(index)  # type: ignore[attr-defined]
            if (
                int(device.get("maxInputChannels", 0)) > 0
                and needle in str(device.get("name", "")).casefold()
            ):
                matches.append(index)
        if not matches:
            raise RuntimeError(f"No microphone device matches {reference!r}; run --list-devices")
        return matches[0]


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if samples.size == 0 or source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)
    target_length = max(1, int(round(len(samples) * target_rate / source_rate)))
    return np.interp(
        np.linspace(0, len(samples) - 1, target_length),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


class VoiceDetector(Protocol):
    def probability(self, pcm_s16le: bytes, sample_rate: int) -> float: ...

    def reset(self) -> None: ...


class SileroVADDetector:
    """Small-window Silero VAD wrapper; Silero expects 512 samples at 16 kHz."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is not installed. Install requirements.txt and the matching PyTorch build."
            ) from exc
        requested = device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA is unavailable to PyTorch; running Silero VAD on CPU")
            requested = "cpu"
        self.torch = torch
        self.device = requested
        self.model = load_silero_vad(onnx=False)
        self.model.eval()
        self.model.to(self.device)

    def probability(self, pcm_s16le: bytes, sample_rate: int) -> float:
        if sample_rate != 16_000:
            raise ValueError("Silero streaming VAD expects 16 kHz audio")
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size != 512:
            raise ValueError("Silero VAD windows must contain exactly 512 samples")
        tensor = self.torch.from_numpy(samples).to(self.device)
        with self.torch.inference_mode():
            return float(self.model(tensor, sample_rate).item())

    def reset(self) -> None:
        reset = getattr(self.model, "reset_states", None)
        if callable(reset):
            reset()


class EnergyVADDetector:
    """Dependency-light fallback detector for constrained machines and testing."""

    def __init__(self, rms_threshold: float = 0.012) -> None:
        self.rms_threshold = rms_threshold

    def probability(self, pcm_s16le: bytes, sample_rate: int) -> float:
        del sample_rate
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        return 0.9 if rms >= self.rms_threshold else 0.0

    def reset(self) -> None:
        return None


class VADPhraseSegmenter:
    """Turn a PCM byte stream into phrase segments using VAD and silence endings.

    Natural silence boundaries are preferred. A configurable hard limit is a
    safety valve for continuous speech; with the 4 s target and 8 s hard limit,
    normal phrases are not chopped simply to satisfy a strict timer.
    """

    WINDOW_SAMPLES = 512

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        backend: str = "silero",
        device: str = "cpu",
        threshold: float = 0.5,
        silence_ms: int = 420,
        min_speech_ms: int = 240,
        min_phrase_seconds: float = 2.0,
        max_phrase_seconds: float = 4.0,
        hard_phrase_seconds: float = 8.0,
        max_join_gap_seconds: float = 0.8,
        detector: VoiceDetector | None = None,
    ) -> None:
        if sample_rate != 16_000:
            raise ValueError("LiveYT VAD currently requires 16 kHz mono audio")
        if backend not in {"silero", "energy"}:
            raise ValueError("VAD backend must be 'silero' or 'energy'")
        if not 0.0 < threshold < 1.0:
            raise ValueError("VAD threshold must be between 0 and 1")
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.min_phrase_samples = int(sample_rate * min_phrase_seconds)
        self.max_phrase_samples = int(sample_rate * max_phrase_seconds)
        self.hard_phrase_samples = int(sample_rate * hard_phrase_seconds)
        self.max_join_gap_samples = int(sample_rate * max_join_gap_seconds)
        if self.min_phrase_samples < 1 or self.max_phrase_samples < self.min_phrase_samples:
            raise ValueError("phrase duration limits must be positive and max >= min")
        if self.max_join_gap_samples < 0:
            raise ValueError("max_join_gap_seconds cannot be negative")
        if self.hard_phrase_samples < self.max_phrase_samples:
            raise ValueError("hard_phrase_seconds must be >= max_phrase_seconds")
        self.detector = detector or (
            SileroVADDetector(device=device) if backend == "silero" else EnergyVADDetector()
        )
        self._pending = bytearray()
        self._processed_samples = 0
        self._pre_roll: deque[tuple[int, bytes]] = deque(maxlen=8)
        self._active = False
        self._segment_start = 0
        self._segment = bytearray()
        self._speech_samples = 0
        self._silence_in_segment = 0
        self._short_pending: SpeechSegment | None = None

    def feed(self, pcm_s16le: bytes) -> list[SpeechSegment]:
        if len(pcm_s16le) % 2:
            pcm_s16le = pcm_s16le[:-1]
        self._pending.extend(pcm_s16le)
        events: list[SpeechSegment] = []
        frame_bytes = self.WINDOW_SAMPLES * 2
        while len(self._pending) >= frame_bytes:
            frame = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            events.extend(self._accept_frame(frame, self._processed_samples))
            self._processed_samples += self.WINDOW_SAMPLES
        return events

    def reset(self, *, start_sample: int | None = None) -> None:
        """Discard buffered state (used when translation is paused).

        ``start_sample`` preserves the source timeline across a pause so later
        voice overlays keep matching the original audio clock.
        """
        self._pending.clear()
        if start_sample is not None:
            self._processed_samples = start_sample
        self._pre_roll.clear()
        self._active = False
        self._segment_start = 0
        self._segment = bytearray()
        self._speech_samples = 0
        self._silence_in_segment = 0
        self._short_pending = None
        self.detector.reset()

    def flush(self) -> list[SpeechSegment]:
        events: list[SpeechSegment] = []
        if self._pending:
            real_bytes = len(self._pending)
            padded = bytes(self._pending) + bytes(self.WINDOW_SAMPLES * 2 - real_bytes)
            events.extend(self._accept_frame(padded, self._processed_samples, real_bytes=real_bytes))
            self._processed_samples += real_bytes // 2
            self._pending.clear()
        if self._active:
            events.extend(self._finish_segment(force=True))
        if self._short_pending is not None:
            events.append(self._short_pending)
            self._short_pending = None
        return events

    def _accept_frame(
        self, pcm: bytes, start_sample: int, *, real_bytes: int | None = None
    ) -> list[SpeechSegment]:
        valid_bytes = len(pcm) if real_bytes is None else real_bytes
        if valid_bytes <= 0:
            return []
        padded = (
            pcm if len(pcm) == self.WINDOW_SAMPLES * 2 else pcm + bytes(self.WINDOW_SAMPLES * 2 - len(pcm))
        )
        probability = self.detector.probability(padded, self.sample_rate)
        valid_pcm = pcm[:valid_bytes]
        samples_in_frame = valid_bytes // 2

        ready: list[SpeechSegment] = []
        if not self._active:
            if (
                self._short_pending is not None
                and start_sample - self._short_pending.end_sample > self.max_join_gap_samples
            ):
                ready.append(self._short_pending)
                self._short_pending = None
            if probability >= self.threshold:
                self._active = True
                if self._pre_roll:
                    self._segment_start = self._pre_roll[0][0]
                    self._segment = bytearray().join(item[1] for item in self._pre_roll)
                else:
                    self._segment_start = start_sample
                    self._segment = bytearray()
                self._segment.extend(valid_pcm)
                self._speech_samples = samples_in_frame
                self._silence_in_segment = 0
                self._pre_roll.clear()
            else:
                self._pre_roll.append((start_sample, valid_pcm))
            return ready

        self._segment.extend(valid_pcm)
        if probability >= self.threshold:
            self._speech_samples += samples_in_frame
            self._silence_in_segment = 0
        else:
            self._silence_in_segment += samples_in_frame

        duration = len(self._segment) // 2
        if duration >= self.hard_phrase_samples:
            logger.warning(
                "Continuous speech reached the %.1f s VAD hard limit; ending this segment to bound latency",
                self.hard_phrase_samples / self.sample_rate,
            )
            return self._finish_segment(force=True)
        if self._silence_in_segment >= self.silence_samples:
            return self._finish_segment(force=False)
        return []

    def _finish_segment(self, *, force: bool) -> list[SpeechSegment]:
        duration_samples = len(self._segment) // 2
        segment: SpeechSegment | None = None
        if self._speech_samples >= self.min_speech_samples and self._segment:
            segment = SpeechSegment(
                start_sample=self._segment_start,
                end_sample=self._segment_start + duration_samples,
                pcm_s16le=bytes(self._segment),
                sample_rate=self.sample_rate,
            )
            if segment.duration_seconds > self.max_phrase_samples / self.sample_rate and not force:
                logger.debug("Natural phrase boundary exceeded the target phrase length")
        self._active = False
        self._segment = bytearray()
        self._speech_samples = 0
        self._silence_in_segment = 0
        self._pre_roll.clear()
        self.detector.reset()
        if segment is None:
            return []
        return self._aggregate_short_phrase(segment, force=force)

    def _aggregate_short_phrase(self, segment: SpeechSegment, *, force: bool) -> list[SpeechSegment]:
        current_samples = segment.end_sample - segment.start_sample
        pending = self._short_pending
        if pending is None:
            if current_samples < self.min_phrase_samples and not force:
                self._short_pending = segment
                return []
            return [segment]

        gap_samples = segment.start_sample - pending.end_sample
        merged_samples = max(pending.end_sample, segment.end_sample) - pending.start_sample
        if gap_samples <= self.max_join_gap_samples and merged_samples <= self.max_phrase_samples:
            merged = _join_speech_segments(pending, segment)
            self._short_pending = None
            if merged.end_sample - merged.start_sample < self.min_phrase_samples and not force:
                self._short_pending = merged
                return []
            return [merged]

        self._short_pending = None
        ready = [pending]
        if current_samples < self.min_phrase_samples and not force:
            self._short_pending = segment
        else:
            ready.append(segment)
        return ready


def _join_speech_segments(first: SpeechSegment, second: SpeechSegment) -> SpeechSegment:
    """Join adjacent short phrases, retaining their source-time gap as silence."""
    if second.start_sample < first.start_sample:
        first, second = second, first
    first_count = len(first.pcm_s16le) // 2
    offset = second.start_sample - first.start_sample
    if offset >= first_count:
        gap = bytes((offset - first_count) * 2)
        combined = first.pcm_s16le + gap + second.pcm_s16le
    else:
        overlap = first_count - offset
        second_tail = second.pcm_s16le[min(len(second.pcm_s16le), overlap * 2) :]
        combined = first.pcm_s16le + second_tail
    return SpeechSegment(
        start_sample=first.start_sample,
        end_sample=first.start_sample + len(combined) // 2,
        pcm_s16le=combined,
        sample_rate=first.sample_rate,
    )


def extract_audio_track(
    video_path: str | Path,
    output_wav: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = 16_000,
) -> Path:
    """Extract the first video audio stream as mono 16 kHz PCM WAV."""
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_wav).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if shutil.which(ffmpeg) is None:
        raise RuntimeError(f"ffmpeg executable {ffmpeg!r} was not found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        os.fspath(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip()[-2_000:]
        raise RuntimeError(f"ffmpeg audio extraction failed (exit {result.returncode}): {detail}")
    return destination
