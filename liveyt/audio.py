"""PCM/WAV helpers, live timeline mixer, and audio output sinks."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def pcm16_wav(pcm: bytes, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Wrap little-endian signed 16-bit PCM in a standards-compliant WAV header."""
    import wave

    if channels < 1 or sample_rate < 1:
        raise ValueError("sample_rate and channels must be positive")
    if len(pcm) % (channels * 2):
        pcm = pcm[: len(pcm) - (len(pcm) % (channels * 2))]
    target = io.BytesIO()
    with wave.open(target, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return target.getvalue()


def wav_to_float32(wav_data: bytes) -> tuple[np.ndarray, int]:
    """Decode common PCM WAV encodings to mono float32 samples in [-1, 1]."""
    import wave

    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        signed = (signed ^ 0x800000) - 0x800000
        samples = signed.astype(np.float32) / 8_388_608.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")

    if channels > 1:
        samples = samples[: (len(samples) // channels) * channels].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), sample_rate


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clean = np.nan_to_num(np.asarray(samples, dtype=np.float32), copy=False)
    return (np.clip(clean, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono float32 audio using linear interpolation (no SciPy required)."""
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if source_rate == target_rate or data.size == 0:
        return np.ascontiguousarray(data)
    target_length = max(1, int(round(data.size * target_rate / source_rate)))
    old_positions = np.arange(data.size, dtype=np.float64)
    new_positions = np.linspace(0, data.size - 1, target_length, dtype=np.float64)
    return np.ascontiguousarray(np.interp(new_positions, old_positions, data).astype(np.float32))


def write_float_wav(path: str, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    """Write float samples as mono PCM16 WAV."""
    import wave

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(float32_to_pcm16(samples))


def wav_duration_seconds(wav_data: bytes) -> float:
    import wave

    with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
        if wav_file.getframerate() <= 0:
            return 0.0
        return wav_file.getnframes() / wav_file.getframerate()


def decode_audio_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Sarvam returned invalid base64 audio") from exc


@dataclass(slots=True)
class RenderedAudioBlock:
    hindi: np.ndarray
    english: np.ndarray


class LiveMixTimeline:
    """Delay live source audio, then overlay translations at source timestamps.

    The buffer provides time for STT/translation/TTS to finish before the matching
    original phrase reaches the output. Both output tracks retain Telugu at the
    configured background gain; translated audio is added at unity gain.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        background_gain: float = 0.15,
        latency_seconds: float = 10.0,
        block_samples: int = 1_600,
    ) -> None:
        if not 0.0 <= background_gain <= 1.0:
            raise ValueError("background_gain must be in [0, 1]")
        if latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative")
        self.sample_rate = sample_rate
        self.background_gain = background_gain
        self.latency_samples = int(sample_rate * latency_seconds)
        self.block_samples = max(1, block_samples)
        self.input_samples = 0
        self.render_cursor = 0
        self._original: deque[tuple[int, np.ndarray]] = deque()
        self._overlays: dict[str, list[tuple[int, np.ndarray]]] = {"hindi": [], "english": []}

    def add_original(self, start_sample: int, pcm_s16le: bytes) -> list[RenderedAudioBlock]:
        samples = np.frombuffer(pcm_s16le, dtype="<i2").astype(np.float32) / 32768.0
        if start_sample < self.input_samples:
            raise ValueError("Original audio frames must be provided in timestamp order")
        if start_sample > self.input_samples:
            self._original.append(
                (self.input_samples, np.zeros(start_sample - self.input_samples, dtype=np.float32))
            )
        if samples.size:
            self._original.append((start_sample, samples))
        self.input_samples = start_sample + int(samples.size)
        return self.render_ready()

    def add_overlay(self, language: str, start_sample: int, samples: np.ndarray, sample_rate: int) -> bool:
        if language not in self._overlays:
            raise ValueError("language must be 'hindi' or 'english'")
        voice = resample_audio(samples, sample_rate, self.sample_rate)
        start = int(start_sample)
        if start < self.render_cursor:
            late_by = self.render_cursor - start
            logger.warning(
                "%s translation arrived %.2fs after its live mix window; trimming late audio",
                language,
                late_by / self.sample_rate,
            )
            voice = voice[late_by:]
            start = self.render_cursor
        if voice.size == 0:
            return False
        self._overlays[language].append((start, voice))
        self._overlays[language].sort(key=lambda item: item[0])
        return True

    def render_ready(self, *, flush: bool = False) -> list[RenderedAudioBlock]:
        safe_end = self.input_samples if flush else max(0, self.input_samples - self.latency_samples)
        rendered: list[RenderedAudioBlock] = []
        while self.render_cursor < safe_end:
            end = min(self.render_cursor + self.block_samples, safe_end)
            original = self._read_original(self.render_cursor, end)
            base = original * self.background_gain
            hindi = self._mix_voice("hindi", base, self.render_cursor, end)
            english = self._mix_voice("english", base, self.render_cursor, end)
            rendered.append(RenderedAudioBlock(hindi=hindi, english=english))
            self.render_cursor = end
            self._discard_before(self.render_cursor)
        return rendered

    def _read_original(self, start: int, end: int) -> np.ndarray:
        result = np.zeros(end - start, dtype=np.float32)
        for chunk_start, chunk in self._original:
            chunk_end = chunk_start + len(chunk)
            left, right = max(start, chunk_start), min(end, chunk_end)
            if left < right:
                result[left - start : right - start] = chunk[left - chunk_start : right - chunk_start]
        return result

    def _mix_voice(self, language: str, base: np.ndarray, start: int, end: int) -> np.ndarray:
        mixed = base.copy()
        for voice_start, voice in self._overlays[language]:
            voice_end = voice_start + len(voice)
            left, right = max(start, voice_start), min(end, voice_end)
            if left < right:
                mixed[left - start : right - start] += voice[left - voice_start : right - voice_start]
        return np.clip(mixed, -1.0, 1.0)

    def _discard_before(self, cursor: int) -> None:
        while self._original:
            chunk_start, chunk = self._original[0]
            chunk_end = chunk_start + len(chunk)
            if chunk_end <= cursor:
                self._original.popleft()
            elif chunk_start < cursor:
                self._original[0] = (cursor, chunk[cursor - chunk_start :])
                break
            else:
                break
        for language, overlays in self._overlays.items():
            self._overlays[language] = [
                (start, samples) for start, samples in overlays if start + len(samples) > cursor
            ]


class AudioSink:
    async def start(self) -> None:  # pragma: no cover - interface
        return None

    async def write(self, mono: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface
        return None


class NullAudioSink(AudioSink):
    async def write(self, mono: np.ndarray) -> None:
        return None


class SoundDeviceSink(AudioSink):
    """Stream float audio to a named/indexed Windows output such as VB-Cable A."""

    def __init__(self, device: str | int, *, sample_rate: int = 48_000) -> None:
        self.device_ref = device
        self.sample_rate = sample_rate
        self._stream: Any = None
        self._channels = 2
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Install sounddevice to route audio to VB-Cable: pip install sounddevice"
            ) from exc
        device_index = self._resolve_device(sd, self.device_ref)
        info = sd.query_devices(device_index)
        self._channels = min(2, int(info["max_output_channels"]))
        if self._channels < 1:
            raise RuntimeError(f"Audio device {info['name']!r} has no output channels")
        self._stream = sd.OutputStream(
            device=device_index,
            samplerate=self.sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=0,
            latency="low",
        )
        await asyncio.to_thread(self._stream.start)
        logger.info("Audio output opened: %s (%d channels)", info["name"], self._channels)

    @staticmethod
    def _resolve_device(sd: Any, reference: str | int) -> int:
        devices = sd.query_devices()
        if isinstance(reference, int) or (isinstance(reference, str) and reference.isdigit()):
            index = int(reference)
            if index < 0 or index >= len(devices):
                raise RuntimeError(f"Output device index {index} is out of range; use --list-devices")
            return index
        needle = str(reference).strip().casefold()
        matches = [
            index
            for index, device in enumerate(devices)
            if int(device["max_output_channels"]) > 0 and needle in str(device["name"]).casefold()
        ]
        if not matches:
            available = [
                f"{index}: {device['name']}"
                for index, device in enumerate(devices)
                if int(device["max_output_channels"]) > 0
            ]
            raise RuntimeError(
                f"Could not find output device matching {reference!r}. Available output devices: "
                + ("; ".join(available) if available else "none")
            )
        return matches[0]

    async def write(self, mono: np.ndarray) -> None:
        if self._stream is None:
            raise RuntimeError("Audio output has not been started")
        values = resample_audio(mono, 16_000, self.sample_rate)
        if self._channels == 1:
            output = values.reshape(-1, 1)
        else:
            output = np.repeat(values.reshape(-1, 1), self._channels, axis=1)
        async with self._write_lock:
            await asyncio.to_thread(self._stream.write, output)

    async def close(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)


class FfmpegRtmpSink:
    """RTMP output with a black 720p slate and the already-ducked PCM mix."""

    def __init__(self, url: str, *, ffmpeg: str = "ffmpeg", sample_rate: int = 48_000) -> None:
        self.url = url
        self.ffmpeg = ffmpeg
        self.sample_rate = sample_rate
        self._process: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        if shutil.which(self.ffmpeg) is None:
            raise RuntimeError(f"ffmpeg executable {self.ffmpeg!r} was not found")
        self._process = await asyncio.create_subprocess_exec(
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:r=30",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-g",
            "60",
            "-keyint_min",
            "60",
            "-sc_threshold",
            "0",
            "-b:v",
            "1500k",
            "-maxrate",
            "1500k",
            "-bufsize",
            "3000k",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-f",
            "flv",
            self.url,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        logger.info("Started 720p black-slate RTMP output (stream key redacted)")

    async def write(self, mono: np.ndarray) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("RTMP audio output is not running")
        if process.returncode is not None:
            raise RuntimeError(f"RTMP ffmpeg process exited with code {process.returncode}")
        values = resample_audio(mono, 16_000, self.sample_rate)
        stereo = np.repeat(values.reshape(-1, 1), 2, axis=1)
        pcm = float32_to_pcm16(stereo)
        async with self._write_lock:
            process.stdin.write(pcm)
            await process.stdin.drain()

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                process.kill()
                await process.wait()
