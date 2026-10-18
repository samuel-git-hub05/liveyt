"""Sarvam REST and WebSocket clients plus an Edge TTS fallback."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import shutil
import subprocess
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from .audio import pcm16_wav
from .config import Settings

logger = logging.getLogger(__name__)


class SarvamAPIError(RuntimeError):
    """A non-retryable error returned by a Sarvam endpoint."""


class SpeechWebSocketError(RuntimeError):
    """WebSocket transport exhausted its reconnect policy."""


class SpeechWebSocketTranscriber:
    """Persistent Saaras streaming connection with serialized turns and backoff."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        language_code: str = "te-IN",
        max_retries: int = 5,
        backoff_initial: float = 0.5,
        backoff_max: float = 12.0,
        response_timeout: float = 35.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language_code = language_code
        self.max_retries = max_retries
        self.backoff_initial = backoff_initial
        self.backoff_max = backoff_max
        self.response_timeout = response_timeout
        self._socket: Any = None
        self._lock = asyncio.Lock()

    async def transcribe(self, pcm_s16le: bytes, sample_rate: int = 16_000) -> str:
        wav_data = pcm16_wav(pcm_s16le, sample_rate)
        async with self._lock:
            last_error: BaseException | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    socket = await self._connect()
                    audio_message = {
                        "audio": {
                            "data": base64.b64encode(wav_data).decode("ascii"),
                            "sample_rate": sample_rate,
                            "encoding": "audio/wav",
                        }
                    }
                    await socket.send(json.dumps(audio_message, separators=(",", ":")))
                    await socket.send(json.dumps({"type": "flush"}))
                    transcript = await self._receive_transcript(socket)
                    return transcript.strip()
                except SarvamAPIError:
                    await self._drop_socket()
                    raise
                except Exception as exc:
                    if not self._is_retryable(exc):
                        await self._drop_socket()
                        raise
                    last_error = exc
                    await self._drop_socket()
                    if attempt >= self.max_retries:
                        break
                    wait = min(self.backoff_initial * (2**attempt), self.backoff_max)
                    wait *= random.uniform(0.85, 1.15)
                    logger.warning(
                        "Saaras WebSocket disconnected (attempt %d/%d); reconnecting in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
            raise SpeechWebSocketError(
                f"Saaras WebSocket failed after {self.max_retries + 1} attempts: {last_error}"
            ) from last_error

    async def _connect(self) -> Any:
        if self._socket is not None:
            return self._socket
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Install the WebSocket dependency: pip install websockets") from exc

        endpoint = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        query = urlencode(
            {
                "language-code": self.language_code,
                "model": self.model,
                "mode": "transcribe",
                "sample_rate": "16000",
                "input_audio_codec": "wav",
                "flush_signal": "true",
            }
        )
        uri = f"{endpoint}/speech-to-text/ws?{query}"
        try:
            self._socket = await websockets.connect(
                uri,
                extra_headers={"api-subscription-key": self.api_key},
                open_timeout=20,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=4 * 1024 * 1024,
            )
        except TypeError as exc:
            # websockets 14 renamed extra_headers; requirements pin <14, but this
            # keeps the integration useful when a user overrides that pin.
            if "extra_headers" not in str(exc):
                raise
            self._socket = await websockets.connect(
                uri,
                additional_headers={"api-subscription-key": self.api_key},
                open_timeout=20,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=4 * 1024 * 1024,
            )
        return self._socket

    async def _receive_transcript(self, socket: Any) -> str:
        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=self.response_timeout)
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                raise SarvamAPIError("Saaras WebSocket returned an invalid JSON message") from exc
            response_type = message.get("type")
            data = message.get("data") or {}
            if response_type == "error" or data.get("error"):
                code = str(data.get("code", ""))
                text = str(data.get("error", "Saaras returned an unspecified error"))
                raise SarvamAPIError(f"Saaras WebSocket error {code}: {text}")
            if response_type == "data" and isinstance(data.get("transcript"), str):
                return data["transcript"]
            if response_type == "events":
                continue

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        try:
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            ConnectionClosed = ()  # type: ignore[assignment,misc]
        if ConnectionClosed and isinstance(exc, ConnectionClosed):
            close_code = getattr(exc, "code", None)
            return not (isinstance(close_code, int) and 4000 <= close_code <= 4999)
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403, 404, 422):
            return False
        return isinstance(exc, (OSError, asyncio.TimeoutError, TimeoutError)) or exc.__class__.__name__ in {
            "InvalidStatus",
            "ConnectionClosedError",
            "ConnectionClosedOK",
            "WebSocketException",
        }

    async def _drop_socket(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await socket.close()
            except Exception:
                logger.debug("Error while closing a failed WebSocket", exc_info=True)

    async def close(self) -> None:
        async with self._lock:
            await self._drop_socket()


class SarvamClient:
    """Async REST client for Saaras REST STT, Mayura, and Bulbul."""

    def __init__(self, settings: Settings, *, request_timeout: float = 60.0) -> None:
        self.settings = settings
        self.api_key = settings.require_sarvam_key()
        self._http = httpx.AsyncClient(
            base_url=settings.sarvam_base_url.rstrip("/"),
            timeout=httpx.Timeout(request_timeout, connect=15.0),
            headers={"api-subscription-key": self.api_key},
        )
        self._ws_stt: SpeechWebSocketTranscriber | None = None

    async def transcribe_rest(self, pcm_s16le: bytes, sample_rate: int = 16_000) -> str:
        wav_data = pcm16_wav(pcm_s16le, sample_rate)
        response = await self._http.post(
            "/speech-to-text",
            data={
                "model": self.settings.stt_model,
                "mode": "transcribe",
                "language_code": "te-IN",
            },
            files={"file": ("phrase.wav", wav_data, "audio/wav")},
        )
        self._raise_for_api(response, "Saaras REST STT")
        payload = response.json()
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise SarvamAPIError("Saaras REST response did not include a transcript")
        return transcript.strip()

    async def transcribe_websocket(self, pcm_s16le: bytes, sample_rate: int = 16_000) -> str:
        if self._ws_stt is None:
            self._ws_stt = SpeechWebSocketTranscriber(
                api_key=self.api_key,
                base_url=self.settings.sarvam_base_url,
                model=self.settings.stt_model,
            )
        return await self._ws_stt.transcribe(pcm_s16le, sample_rate)

    async def translate(
        self,
        text: str,
        target_language: str,
        *,
        glossary: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        text = text.strip()
        if not text:
            return ""
        if target_language not in {"hi-IN", "en-IN"}:
            raise ValueError("LiveYT supports Hindi (hi-IN) and English (en-IN) targets")
        if len(text) > 1000 and self.settings.translation_model == "mayura:v1":
            raise ValueError("Mayura supports at most 1,000 characters per request")
        response = await self._http.post(
            "/translate",
            json={
                "input": text,
                "source_language_code": "te-IN",
                "target_language_code": target_language,
                "model": self.settings.translation_model,
                "mode": "formal",
                "speaker_gender": "Male",
                "numerals_format": "international",
            },
        )
        self._raise_for_api(response, "Mayura translation")
        translated = response.json().get("translated_text")
        if not isinstance(translated, str) or not translated.strip():
            raise SarvamAPIError("Mayura returned an empty translation")
        language = "hi" if target_language == "hi-IN" else "en"
        return apply_sermon_glossary(translated.strip(), language, glossary or self.settings.glossary)

    async def synthesize(self, text: str, language_code: str) -> bytes:
        text = text.strip()
        if not text:
            return b""
        if len(text) > 2500:
            raise ValueError("Bulbul supports at most 2,500 characters per TTS request")
        if language_code == "hi-IN":
            speaker = self.settings.hindi_speaker
        elif language_code == "en-IN":
            speaker = self.settings.english_speaker
        else:
            raise ValueError("Bulbul target must be hi-IN or en-IN")
        response = await self._http.post(
            "/text-to-speech",
            json={
                "text": text,
                "model": self.settings.tts_model,
                "language_code": language_code,
                "speaker": speaker,
                "speech_sample_rate": self.settings.tts_sample_rate,
                "output_audio_codec": "wav",
                "pace": 1.0,
                "temperature": 0.5,
            },
        )
        self._raise_for_api(response, "Bulbul text-to-speech")
        audios = response.json().get("audios")
        if not isinstance(audios, list) or not audios or not isinstance(audios[0], str):
            raise SarvamAPIError("Bulbul response did not contain an audio result")
        try:
            audio = base64.b64decode(audios[0], validate=True)
        except (ValueError, TypeError) as exc:
            raise SarvamAPIError("Bulbul response contained malformed base64 audio") from exc
        if not audio:
            raise SarvamAPIError("Bulbul returned an empty audio clip")
        return audio

    @staticmethod
    def _raise_for_api(response: httpx.Response, endpoint: str) -> None:
        if response.is_success:
            return
        detail = response.text.strip().replace("\n", " ")[:800]
        raise SarvamAPIError(f"{endpoint} returned HTTP {response.status_code}: {detail}")

    async def close(self) -> None:
        if self._ws_stt is not None:
            await self._ws_stt.close()
        await self._http.aclose()

    async def __aenter__(self) -> SarvamClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()


def apply_sermon_glossary(
    translated_text: str,
    language: str,
    glossary: dict[str, dict[str, Any]],
) -> str:
    """Normalize configured devotional vocabulary after machine translation.

    The Mayura API has no user-defined glossary field. This deterministic pass
    therefore normalizes known alternate target forms and catches Telugu terms
    that a model may have left untranslated. Longer source phrases run first.
    """
    result = translated_text
    for source_term, entry in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
        canonical = entry.get(language)
        if not isinstance(canonical, str) or not canonical:
            continue
        if source_term in result:
            result = result.replace(source_term, canonical)
        aliases = entry.get(f"{language}_aliases", [])
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias and alias != canonical:
                    result = result.replace(alias, canonical)
    return result


class TTSProvider(Protocol):
    async def synthesize(self, text: str, language_code: str) -> bytes: ...


class EdgeTTSProvider:
    """Free Edge neural voices, converted to mono PCM WAV by ffmpeg."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def synthesize(self, text: str, language_code: str) -> bytes:
        if language_code == "hi-IN":
            voice = self.settings.edge_hindi_voice
        elif language_code == "en-IN":
            voice = self.settings.edge_english_voice
        else:
            raise ValueError("Edge TTS target must be hi-IN or en-IN")
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("Install edge-tts for the free TTS fallback: pip install edge-tts") from exc
        if shutil.which(self.settings.ffmpeg) is None:
            raise RuntimeError(f"ffmpeg executable {self.settings.ffmpeg!r} was not found")
        audio = bytearray()
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data", b""))
        if not audio:
            raise RuntimeError(f"Edge TTS returned no audio for voice {voice!r}")
        result = await asyncio.to_thread(
            subprocess.run,
            [
                self.settings.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "mp3",
                "-i",
                "pipe:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.settings.tts_sample_rate),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ],
            input=bytes(audio),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace")[-1_000:]
            raise RuntimeError(f"ffmpeg could not convert Edge TTS audio: {message}")
        return result.stdout
