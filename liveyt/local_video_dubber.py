"""Sequential, VRAM-conscious Telugu-to-Hindi/English local video dubbing."""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .audio import float32_to_pcm16, wav_to_float32, write_float_wav
from .config import Settings
from .sarvam_client import EdgeTTSProvider, SarvamClient, apply_sermon_glossary
from .vad_ingestor import extract_audio_track

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class DubbedSegment:
    source: TranscriptSegment
    hindi: str
    english: str
    hindi_wav: bytes
    english_wav: bytes


class SegmentTranslator(Protocol):
    def translate(self, text: str, target_language: str) -> str: ...


class NllbLocalTranslator:
    """NLLB-200 distilled 600M translator, deliberately loaded on CPU after ASR."""

    TARGET_CODES = {"hi": "hin_Deva", "en": "eng_Latn"}

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        glossary: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local translation needs transformers, sentencepiece, and PyTorch. "
                "Install the optional local-video dependencies from requirements-local-video.txt."
            ) from exc
        self.torch = torch
        self.glossary = glossary or {}
        logger.info("Loading %s on CPU (Whisper has been unloaded from CUDA)", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="tel_Telu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self.model.to("cpu")
        self.model.eval()

    def translate(self, text: str, target_language: str) -> str:
        if target_language not in self.TARGET_CODES:
            raise ValueError("NLLB target must be 'hi' or 'en'")
        text = text.strip()
        if not text:
            return ""
        self.tokenizer.src_lang = "tel_Telu"
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        forced_bos_id = self.tokenizer.convert_tokens_to_ids(self.TARGET_CODES[target_language])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=256,
                num_beams=4,
                do_sample=False,
            )
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return apply_sermon_glossary(result, target_language, self.glossary)


class CloudSegmentTranslator:
    """Mayura-backed translation wrapper for the explicitly selected cloud engine."""

    def __init__(self, client: SarvamClient, glossary: dict[str, dict[str, Any]]) -> None:
        self.client = client
        self.glossary = glossary

    async def translate_pair(self, text: str) -> tuple[str, str]:
        hindi, english = await asyncio.gather(
            self.client.translate(text, "hi-IN", glossary=self.glossary),
            self.client.translate(text, "en-IN", glossary=self.glossary),
        )
        return hindi, english


class LocalVideoDubber:
    """Run extraction, local GPU transcription, translation, TTS, and ffmpeg muxing."""

    def __init__(
        self,
        settings: Settings,
        *,
        input_file: str | Path,
        engine: str = "local",
        whisper_model: str = "small",
        output_dir: str | Path | None = None,
        translation_model: str = "facebook/nllb-200-distilled-600M",
        tts_provider: str = "edge",
        cuda_memory_limit_gb: float = 4.5,
        glossary: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if engine not in {"local", "cloud"}:
            raise ValueError("engine must be 'local' or 'cloud'")
        if whisper_model not in {"small", "large-v3-turbo"}:
            raise ValueError("whisper_model must be 'small' or 'large-v3-turbo'")
        if tts_provider != "edge":
            raise ValueError("Local-video TTS currently uses Edge neural voices")
        self.settings = settings
        self.input_file = Path(input_file).expanduser().resolve()
        self.engine = engine
        self.whisper_model = whisper_model
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else self.input_file.parent
        self.translation_model = translation_model
        self.tts_provider = tts_provider
        if cuda_memory_limit_gb <= 0 or cuda_memory_limit_gb > 4.5:
            raise ValueError("cuda_memory_limit_gb must be greater than 0 and no more than 4.5")
        self.cuda_memory_limit_gb = cuda_memory_limit_gb
        self.glossary = glossary or settings.glossary
        self.edge_tts = EdgeTTSProvider(settings)

    async def run(self) -> tuple[Path, Path]:
        if not self.input_file.is_file():
            raise FileNotFoundError(f"Input video does not exist: {self.input_file}")
        self._validate_runtime()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        hindi_output = self.output_dir / "output_hindi.mp4"
        english_output = self.output_dir / "output_english.mp4"
        with tempfile.TemporaryDirectory(prefix="liveyt-") as temp_dir:
            work = Path(temp_dir)
            source_audio = extract_audio_track(
                self.input_file,
                work / "telugu_16k.wav",
                ffmpeg=self.settings.ffmpeg,
                sample_rate=self.settings.sample_rate,
            )
            duration = self._probe_duration(self.input_file)
            logger.info("Transcribing %.1f seconds of Telugu audio with faster-whisper", duration)
            segments, whisper_model = await asyncio.to_thread(self._transcribe, source_audio)
            try:
                del whisper_model
            except UnboundLocalError:
                pass
            gc.collect()
            _clear_cuda_cache()
            if not segments:
                raise RuntimeError("Whisper did not detect any speech in the input audio")
            logger.info(
                "Whisper found %d speech segments; translating and synthesizing sequentially", len(segments)
            )

            if self.engine == "local":
                translator: NllbLocalTranslator | None = await asyncio.to_thread(
                    NllbLocalTranslator, self.translation_model, self.glossary
                )
                cloud_translator = None
            else:
                translator = None
                cloud_client = SarvamClient(self.settings)
                cloud_translator = CloudSegmentTranslator(cloud_client, self.glossary)

            dubbed: list[DubbedSegment] = []
            try:
                for index, segment in enumerate(segments, start=1):
                    if self.engine == "local":
                        assert translator is not None
                        hindi = await asyncio.to_thread(translator.translate, segment.text, "hi")
                        english = await asyncio.to_thread(translator.translate, segment.text, "en")
                    else:
                        assert cloud_translator is not None
                        hindi, english = await cloud_translator.translate_pair(segment.text)
                    if not hindi.strip() and not english.strip():
                        logger.warning("Skipping empty translations for segment %d", index)
                        continue
                    hindi_wav, english_wav = await asyncio.gather(
                        self.edge_tts.synthesize(hindi, "hi-IN") if hindi.strip() else _empty_audio(),
                        self.edge_tts.synthesize(english, "en-IN") if english.strip() else _empty_audio(),
                    )
                    dubbed.append(DubbedSegment(segment, hindi, english, hindi_wav, english_wav))
                    if index % 20 == 0 or index == len(segments):
                        logger.info("Prepared dubbing for %d/%d segments", index, len(segments))
            finally:
                if cloud_translator is not None:
                    await cloud_translator.client.close()
                if translator is not None:
                    del translator
                    gc.collect()
                    _clear_cuda_cache()

            if not dubbed:
                raise RuntimeError("No translated voiceover segments were generated")
            hindi_voice = work / "hindi_voice.wav"
            english_voice = work / "english_voice.wav"
            hindi_gain = work / "hindi_original_gain.wav"
            english_gain = work / "english_original_gain.wav"
            self._assemble_track(dubbed, "hindi", duration, hindi_voice, hindi_gain)
            self._assemble_track(dubbed, "english", duration, english_voice, english_gain)
            logger.info("Muxing Hindi and English MP4s; video streams are copied, not re-encoded")
            await asyncio.to_thread(self._mux, hindi_voice, hindi_gain, hindi_output)
            await asyncio.to_thread(self._mux, english_voice, english_gain, english_output)
        return hindi_output, english_output

    def _validate_runtime(self) -> None:
        for executable in (self.settings.ffmpeg, self.settings.ffprobe):
            if shutil.which(executable) is None:
                raise RuntimeError(f"Required media executable {executable!r} was not found")
        required = ["faster_whisper", "ctranslate2", "edge_tts"]
        if self.engine == "local":
            required.extend(("torch", "transformers", "sentencepiece"))
        else:
            self.settings.require_sarvam_key()
        import importlib.util

        missing = [module for module in required if importlib.util.find_spec(module) is None]
        if missing:
            raise RuntimeError(
                "Missing local-video dependencies: "
                + ", ".join(missing)
                + ". Install requirements-local-video.txt (and the matching CUDA PyTorch wheel)."
            )
        if self.input_file.resolve() in {
            (self.output_dir / "output_hindi.mp4").resolve(),
            (self.output_dir / "output_english.mp4").resolve(),
        }:
            raise ValueError(
                "Input video cannot also be one of the output_hindi.mp4/output_english.mp4 files"
            )

    def _transcribe(self, audio_path: Path) -> tuple[list[TranscriptSegment], Any]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install faster-whisper and the matching CTranslate2 CUDA runtime: pip install faster-whisper"
            ) from exc

        try:
            import ctranslate2

            use_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            use_cuda = False
        device = "cuda" if use_cuda else "cpu"
        compute_type = "int8_float16" if use_cuda else "int8"
        selected = self.whisper_model
        if use_cuda:
            try:
                import torch

                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                budget_bytes = int(self.cuda_memory_limit_gb * 1024**3)
                logger.info(
                    "CUDA reports %.2f/%.2f GiB free; Whisper VRAM target is <= %.1f GiB",
                    free_bytes / 1024**3,
                    total_bytes / 1024**3,
                    self.cuda_memory_limit_gb,
                )
                if selected == "large-v3-turbo" and free_bytes < budget_bytes:
                    logger.warning(
                        "Free VRAM is below the configured budget; using small instead of large-v3-turbo"
                    )
                    selected = "small"
                if free_bytes < 1.5 * 1024**3:
                    logger.warning("Less than 1.5 GiB CUDA memory is free; running faster-whisper on CPU")
                    device, compute_type = "cpu", "int8"
            except Exception:
                logger.debug("PyTorch could not report CUDA memory; relying on int8_float16 and OOM fallback")
        logger.info("Loading faster-whisper %s on %s (%s)", selected, device, compute_type)
        try:
            model = WhisperModel(
                selected,
                device=device,
                compute_type=compute_type,
                num_workers=1,
                cpu_threads=4,
            )
            raw_segments, _info = model.transcribe(
                str(audio_path),
                language="te",
                beam_size=5,
                vad_filter=True,
                word_timestamps=False,
                condition_on_previous_text=True,
            )
            result = [
                TranscriptSegment(
                    start=max(0.0, float(item.start)),
                    end=max(float(item.start), float(item.end)),
                    text=str(item.text).strip(),
                )
                for item in raw_segments
                if str(item.text).strip()
            ]
            return result, model
        except Exception as exc:
            if not use_cuda or not _looks_like_cuda_oom(exc):
                raise RuntimeError(f"faster-whisper transcription failed: {exc}") from exc
            logger.warning(
                "CUDA ran out of memory with %s; clearing cache and retrying small/int8 on CPU",
                selected,
            )
            try:
                del model
            except UnboundLocalError:
                pass
            gc.collect()
            _clear_cuda_cache()
            fallback = WhisperModel("small", device="cpu", compute_type="int8", num_workers=1, cpu_threads=4)
            raw_segments, _info = fallback.transcribe(
                str(audio_path),
                language="te",
                beam_size=5,
                vad_filter=True,
                word_timestamps=False,
                condition_on_previous_text=True,
            )
            result = [
                TranscriptSegment(
                    start=max(0.0, float(item.start)),
                    end=max(float(item.start), float(item.end)),
                    text=str(item.text).strip(),
                )
                for item in raw_segments
                if str(item.text).strip()
            ]
            return result, fallback

    def _assemble_track(
        self,
        segments: list[DubbedSegment],
        language: str,
        video_duration: float,
        voice_path: Path,
        gain_path: Path,
    ) -> None:
        sample_rate = self.settings.sample_rate
        total_samples = max(1, int(math.ceil(video_duration * sample_rate)))
        voice_track = np.zeros(total_samples, dtype=np.float32)
        gain_track = np.ones(total_samples, dtype=np.float32)
        for item in segments:
            start = max(0, min(total_samples, int(round(item.source.start * sample_rate))))
            end = max(start, min(total_samples, int(round(item.source.end * sample_rate))))
            if end <= start:
                continue
            wav = item.hindi_wav if language == "hindi" else item.english_wav
            if not wav:
                continue
            samples, tts_rate = wav_to_float32(wav)
            fitted = self._fit_audio(samples, tts_rate, end - start)
            voice_track[start:end] += fitted[: end - start] * self.settings.voiceover_gain
            gain_track[start:end] = self.settings.background_gain
        write_float_wav(str(voice_path), np.clip(voice_track, -1.0, 1.0), sample_rate)
        write_float_wav(str(gain_path), gain_track, sample_rate)

    def _fit_audio(self, samples: np.ndarray, sample_rate: int, target_samples: int) -> np.ndarray:
        """Use ffmpeg atempo and pad/trim to the exact Whisper time window."""
        if target_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        if samples.size == 0:
            return np.zeros(target_samples, dtype=np.float32)
        target_duration = target_samples / self.settings.sample_rate
        source_duration = samples.size / sample_rate
        speed = source_duration / target_duration
        factors: list[float] = []
        remaining = speed
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        factors.append(max(0.5, min(2.0, remaining)))
        filters = ",".join(f"atempo={factor:.8f}" for factor in factors)
        source_wav = _float_samples_to_wav(samples, sample_rate)
        process = subprocess.run(
            [
                self.settings.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                "pipe:0",
                "-af",
                filters,
                "-ac",
                "1",
                "-ar",
                str(self.settings.sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=source_wav,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace")[-1_000:]
            raise RuntimeError(f"ffmpeg atempo failed to fit a voiceover: {detail}")
        fitted, fitted_rate = wav_to_float32(process.stdout)
        if fitted_rate != self.settings.sample_rate:
            from .audio import resample_audio

            fitted = resample_audio(fitted, fitted_rate, self.settings.sample_rate)
        if fitted.size < target_samples:
            fitted = np.pad(fitted, (0, target_samples - fitted.size))
        elif fitted.size > target_samples:
            fitted = fitted[:target_samples]
        return np.ascontiguousarray(fitted, dtype=np.float32)

    def _mux(self, voice_path: Path, gain_path: Path, output_path: Path) -> None:
        command = [
            self.settings.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(self.input_file),
            "-i",
            str(voice_path),
            "-i",
            str(gain_path),
            "-filter_complex",
            "[0:a:0]aresample=48000,aformat=channel_layouts=stereo,apad[original];"
            "[1:a:0]aresample=48000,aformat=channel_layouts=stereo[voice];"
            "[2:a:0]aresample=48000,aformat=channel_layouts=stereo[gain];"
            "[original][gain]amultiply[ducked];"
            "[ducked][voice]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]",
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
            "-map_metadata",
            "0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip()[-2_000:]
            raise RuntimeError(
                f"ffmpeg mux failed for {output_path.name} (exit {result.returncode}). "
                "The input video codec must be compatible with MP4 for -c:v copy; "
                f"ffmpeg reported: {detail}"
            )

    def _probe_duration(self, video_path: Path) -> float:
        if shutil.which(self.settings.ffprobe) is None:
            raise RuntimeError(f"ffprobe executable {self.settings.ffprobe!r} was not found")
        result = subprocess.run(
            [
                self.settings.ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-1_000:]
            raise RuntimeError(f"ffprobe could not read video duration: {detail}")
        try:
            duration = float(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("ffprobe returned an invalid video duration") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Input video duration must be positive")
        return duration


def _float_samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    import io
    import wave

    target = io.BytesIO()
    with wave.open(target, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(float32_to_pcm16(samples))
    return target.getvalue()


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _looks_like_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return (
        "out of memory" in text or "cuda_error_out_of_memory" in text or "cublas_status_alloc_failed" in text
    )


async def _empty_audio() -> bytes:
    return b""
