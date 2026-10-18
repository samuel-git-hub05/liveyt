"""Async live sermon translation pipeline for YouTube and direct microphone input."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from typing import Any

from .audio import (
    AudioSink,
    FfmpegRtmpSink,
    LiveMixTimeline,
    NullAudioSink,
    SoundDeviceSink,
    wav_to_float32,
)
from .config import Settings
from .sarvam_client import EdgeTTSProvider, SarvamClient, SpeechWebSocketError
from .vad_ingestor import (
    AudioSource,
    DirectMicAudioSource,
    SpeechSegment,
    VADPhraseSegmenter,
    YoutubeAudioSource,
)

logger = logging.getLogger(__name__)


class LivePipeline:
    """Connect an audio source to STT, dual translation/TTS, and audio outputs."""

    def __init__(self, settings: Settings, args: Any) -> None:
        self.settings = settings
        self.args = args
        if args.mode == "youtube":
            self.source: AudioSource = YoutubeAudioSource(
                args.url,
                yt_dlp=settings.yt_dlp,
                ffmpeg=settings.ffmpeg,
                sample_rate=settings.sample_rate,
                realtime=True,
            )
        elif args.mode == "direct":
            self.source = DirectMicAudioSource(args.device, sample_rate=settings.sample_rate)
        else:
            raise ValueError("LivePipeline supports only youtube and direct modes")
        self.stop_event = asyncio.Event()
        self.paused_event = asyncio.Event()
        self.keyboard_task: asyncio.Task[None] | None = None
        self.client = SarvamClient(settings)
        self.tts = EdgeTTSProvider(settings) if args.tts_provider == "edge" else self.client
        self.sinks: dict[str, AudioSink] = {}
        if args.no_playback:
            self.sinks = {"hindi": NullAudioSink(), "english": NullAudioSink()}
        else:
            self.sinks = {
                "hindi": SoundDeviceSink(args.hindi_device, sample_rate=settings.output_sample_rate),
                "english": SoundDeviceSink(args.english_device, sample_rate=settings.output_sample_rate),
            }
        self.rtmp: FfmpegRtmpSink | None = None
        if args.rtmp_url:
            self.rtmp = FfmpegRtmpSink(args.rtmp_url, ffmpeg=settings.ffmpeg)
        self.timeline = LiveMixTimeline(
            sample_rate=settings.sample_rate,
            background_gain=settings.background_gain,
            latency_seconds=args.latency,
        )
        self._pending: set[asyncio.Task[None]] = set()
        self._inflight = asyncio.Semaphore(max(1, args.max_inflight))
        self._stt_fallback_logged = False

    async def run(self) -> None:
        started_sinks: list[AudioSink] = []
        try:
            for sink in self.sinks.values():
                await sink.start()
                started_sinks.append(sink)
            if self.rtmp is not None:
                await self.rtmp.start()
            if self.args.no_playback and self.rtmp is None:
                logger.warning("No playback or RTMP output is enabled; audio will be processed but not heard")
            if self.args.keyboard and sys.stdin.isatty():
                self.keyboard_task = asyncio.create_task(
                    keyboard_control_loop(self.paused_event, self.stop_event),
                    name="live-keyboard-controls",
                )
            logger.info(
                "Live %s dubbing started. Telugu background gain: %.0f%%; voice gain: %.0f%%; "
                "mix latency: %.1fs",
                self.args.mode,
                self.settings.background_gain * 100,
                self.settings.voiceover_gain * 100,
                self.args.latency,
            )
            logger.info("Press P + Enter to pause/resume translation and Q + Enter to stop")
            segmenter = self._new_segmenter()
            was_paused = False
            began_at = time.monotonic()
            async for frame in self.source.frames():
                if self.stop_event.is_set():
                    break
                if self.args.max_runtime and time.monotonic() - began_at >= self.args.max_runtime:
                    logger.info("Maximum live runtime reached")
                    break
                await self._write_blocks(self.timeline.add_original(frame.start_sample, frame.pcm_s16le))
                paused = self.paused_event.is_set()
                if paused:
                    if not was_paused:
                        logger.info("Translation paused; Telugu passthrough continues at 15%%")
                        segmenter.reset(start_sample=frame.end_sample)
                    was_paused = True
                    continue
                if was_paused:
                    logger.info("Translation resumed")
                    segmenter.reset(start_sample=frame.start_sample)
                    was_paused = False
                for phrase in segmenter.feed(frame.pcm_s16le):
                    self._schedule_phrase(phrase)
            for phrase in segmenter.flush():
                if not self.paused_event.is_set():
                    self._schedule_phrase(phrase)
            if self._pending:
                await asyncio.gather(*tuple(self._pending), return_exceptions=True)
            await self._write_blocks(self.timeline.render_ready(flush=True))
        finally:
            self.stop_event.set()
            if self.keyboard_task is not None:
                self.keyboard_task.cancel()
                await asyncio.gather(self.keyboard_task, return_exceptions=True)
            if self._pending:
                for task in tuple(self._pending):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tuple(self._pending), return_exceptions=True)
            if self.rtmp is not None:
                await self.rtmp.close()
            for sink in reversed(started_sinks):
                await sink.close()
            await self.client.close()
            logger.info("Live dubbing stopped")

    def _new_segmenter(self) -> VADPhraseSegmenter:
        return VADPhraseSegmenter(
            sample_rate=self.settings.sample_rate,
            backend=self.settings.vad_backend,
            device=self.settings.vad_device,
            silence_ms=self.settings.vad_silence_ms,
            min_phrase_seconds=self.settings.min_phrase_seconds,
            max_phrase_seconds=self.settings.max_phrase_seconds,
            hard_phrase_seconds=self.settings.hard_phrase_seconds,
        )

    def _schedule_phrase(self, phrase: SpeechSegment) -> None:
        task = asyncio.create_task(self._process_phrase(phrase), name=f"dub-{phrase.start_sample}")
        self._pending.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            self._pending.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error(
                    "Phrase processing failed: %s",
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(finished)

    async def _process_phrase(self, phrase: SpeechSegment) -> None:
        async with self._inflight:
            if self.paused_event.is_set() or self.stop_event.is_set():
                return
            try:
                transcript = await self._transcribe(phrase.pcm_s16le)
                if not transcript:
                    logger.debug(
                        "No transcript returned for phrase at %.2fs",
                        phrase.start_sample / self.settings.sample_rate,
                    )
                    return
                logger.info("Telugu: %s", transcript)
                hindi_text, english_text = await asyncio.gather(
                    self.client.translate(transcript, "hi-IN", glossary=self.settings.glossary),
                    self.client.translate(transcript, "en-IN", glossary=self.settings.glossary),
                )
                logger.info("Hindi: %s", hindi_text)
                logger.info("English: %s", english_text)
                hindi_wav, english_wav = await asyncio.gather(
                    self.tts.synthesize(hindi_text, "hi-IN"),
                    self.tts.synthesize(english_text, "en-IN"),
                )
                hindi_samples, hindi_rate = wav_to_float32(hindi_wav)
                english_samples, english_rate = wav_to_float32(english_wav)
                voice_gain = self.settings.voiceover_gain
                self.timeline.add_overlay(
                    "hindi", phrase.start_sample, hindi_samples * voice_gain, hindi_rate
                )
                self.timeline.add_overlay(
                    "english", phrase.start_sample, english_samples * voice_gain, english_rate
                )
            except SpeechWebSocketError:
                # _transcribe handles the configured REST fallback. Reaching this
                # branch means fallback was deliberately disabled.
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Unable to translate phrase beginning at %.2fs", phrase.start_sample / phrase.sample_rate
                )

    async def _transcribe(self, pcm: bytes) -> str:
        if self.args.stt_transport == "rest":
            return await self.client.transcribe_rest(pcm, self.settings.sample_rate)
        try:
            return await self.client.transcribe_websocket(pcm, self.settings.sample_rate)
        except SpeechWebSocketError:
            if not self.args.stt_fallback:
                raise
            if not self._stt_fallback_logged:
                logger.warning("Saaras WebSocket failed; REST fallback is enabled")
                self._stt_fallback_logged = True
            return await self.client.transcribe_rest(pcm, self.settings.sample_rate)

    async def _write_blocks(self, blocks: list[Any]) -> None:
        for block in blocks:
            selected = self.args.rtmp_language
            writes = [
                self.sinks["hindi"].write(block.hindi),
                self.sinks["english"].write(block.english),
            ]
            if self.rtmp is not None:
                rtmp_audio = block.hindi if selected == "hindi" else block.english
                writes.append(self.rtmp.write(rtmp_audio))
            await asyncio.gather(*writes)


async def keyboard_control_loop(paused: asyncio.Event, stop: asyncio.Event) -> None:
    """Read P/Q controls without a blocking worker thread (Windows and POSIX)."""
    if os.name == "nt":
        try:
            import msvcrt
        except ImportError:  # pragma: no cover - only Windows uses this branch
            return
        while not stop.is_set():
            await asyncio.sleep(0.1)
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == "p":
                    _toggle_pause(paused)
                elif key == "q":
                    logger.info("Stop requested from keyboard")
                    stop.set()
                    return
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()
    try:
        fd = sys.stdin.fileno()
        loop.add_reader(fd, lambda: queue.put_nowait(sys.stdin.readline()))
    except (AttributeError, OSError, RuntimeError):
        logger.warning("Keyboard controls are unavailable on this input stream")
        return
    try:
        while not stop.is_set():
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            command = line.strip().lower()
            if command == "p":
                _toggle_pause(paused)
            elif command == "q":
                logger.info("Stop requested from keyboard")
                stop.set()
                return
    finally:
        loop.remove_reader(fd)


def _toggle_pause(paused: asyncio.Event) -> None:
    if paused.is_set():
        paused.clear()
    else:
        paused.set()
    logger.info("Translation %s", "resumed" if not paused.is_set() else "paused")


def list_live_devices() -> None:
    from .config import discover_audio_devices, discover_pyaudio_devices

    devices = discover_audio_devices()
    pyaudio_devices = discover_pyaudio_devices()
    if not devices and not pyaudio_devices:
        print(
            "No audio devices found. Check your Windows audio drivers and sounddevice/PyAudio installation."
        )
        return
    if devices:
        print("sounddevice devices (use names for --hindi-device / --english-device):")
        for item in devices:
            direction = (
                ", ".join(
                    label
                    for label, count in (("input", item["inputs"]), ("output", item["outputs"]))
                    if count
                )
                or "no audio channels"
            )
            print(f"  {item['index']:>3}: {item['name']}  [{direction}; {item['default_samplerate']} Hz]")
    mic_devices = [item for item in pyaudio_devices if item["inputs"] > 0]
    if mic_devices:
        print("PyAudio microphone indices (use these with --device):")
        for item in mic_devices:
            print(f"  {item['index']:>3}: {item['name']}  [{item['default_samplerate']} Hz]")


def validate_live_dependencies(args: Any, settings: Settings) -> None:
    if shutil.which(settings.ffmpeg) is None:
        raise RuntimeError(f"ffmpeg executable {settings.ffmpeg!r} was not found")
    if args.mode == "youtube" and shutil.which(settings.yt_dlp) is None:
        raise RuntimeError("yt-dlp was not found. Install it with: py -m pip install yt-dlp")
    if args.mode == "direct":
        try:
            import pyaudio  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Direct mode requires PyAudio. Install requirements-live.txt") from exc
    if not args.no_playback:
        try:
            import sounddevice  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("VB-Cable output requires sounddevice. Install requirements-live.txt") from exc
    if args.tts_provider == "edge":
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Edge TTS requires edge-tts. Install requirements-live.txt") from exc
    if args.vad == "silero":
        try:
            import silero_vad  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Live mode requires Silero VAD. Install requirements-live.txt and the matching PyTorch wheel."
            ) from exc
