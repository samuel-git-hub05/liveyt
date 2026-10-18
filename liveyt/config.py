"""Configuration, defaults, and audio-device discovery for LiveYT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional when importing the library
    load_dotenv = None


# Extend this dictionary for the vocabulary used by your church or denomination.
# Values are canonical translation terms; aliases are normalized after translation.
SERMON_GLOSSARY: dict[str, dict[str, Any]] = {
    "దేవుడు": {
        "hi": "परमेश्वर",
        "en": "God",
        "hi_aliases": ["ईश्वर"],
        "en_aliases": [],
    },
    "ప్రభువు": {
        "hi": "प्रभु",
        "en": "the Lord",
        "hi_aliases": [],
        "en_aliases": [],
    },
    "యేసు క్రీస్తు": {
        "hi": "यीशु मसीह",
        "en": "Jesus Christ",
        "hi_aliases": ["ईसा मसीह"],
        "en_aliases": [],
    },
    "యేసు": {
        "hi": "यीशु",
        "en": "Jesus",
        "hi_aliases": ["ईसा"],
        "en_aliases": [],
    },
    "పరిశుద్ధాత్మ": {
        "hi": "पवित्र आत्मा",
        "en": "the Holy Spirit",
        "hi_aliases": [],
        "en_aliases": ["Holy Ghost"],
    },
    "ప్రార్థన": {
        "hi": "प्रार्थना",
        "en": "prayer",
        "hi_aliases": [],
        "en_aliases": [],
    },
    "సువార్త": {
        "hi": "सुसमाचार",
        "en": "the Gospel",
        "hi_aliases": [],
        "en_aliases": [],
    },
    "ఆమేన్": {
        "hi": "आमीन",
        "en": "Amen",
        "hi_aliases": [],
        "en_aliases": [],
    },
    "కృప": {
        "hi": "अनुग्रह",
        "en": "grace",
        "hi_aliases": [],
        "en_aliases": [],
    },
    "రక్షణ": {
        "hi": "उद्धार",
        "en": "salvation",
        "hi_aliases": [],
        "en_aliases": [],
    },
}


@dataclass(slots=True)
class Settings:
    """Runtime settings. Secrets are read only from the environment / .env."""

    sample_rate: int = 16_000
    output_sample_rate: int = 48_000
    background_gain: float = 0.15
    voiceover_gain: float = 1.0
    vad_backend: str = "silero"
    vad_device: str = "cpu"
    min_phrase_seconds: float = 2.0
    max_phrase_seconds: float = 4.0
    hard_phrase_seconds: float = 8.0
    vad_silence_ms: int = 420
    live_latency_seconds: float = 10.0
    sarvam_api_key: str | None = None
    sarvam_base_url: str = "https://api.sarvam.ai"
    stt_model: str = "saaras:v3"
    translation_model: str = "mayura:v1"
    tts_model: str = "bulbul:v4-flash"
    tts_sample_rate: int = 24_000
    hindi_speaker: str = "shubh_hi_devotional"
    english_speaker: str = "girish_en_devotional"
    edge_hindi_voice: str = "hi-IN-MadhurNeural"
    edge_english_voice: str = "en-IN-PrabhatNeural"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    yt_dlp: str = "yt-dlp"
    glossary: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(SERMON_GLOSSARY))

    @classmethod
    def from_environment(cls, *, env_file: str | Path | None = ".env") -> Settings:
        if load_dotenv is not None and env_file is not None:
            load_dotenv(dotenv_path=Path(env_file), override=False)
        return cls(
            sarvam_api_key=os.getenv("SARVAM_API_KEY") or None,
            ffmpeg=os.getenv("FFMPEG_PATH", "ffmpeg"),
            ffprobe=os.getenv("FFPROBE_PATH", "ffprobe"),
            yt_dlp=os.getenv("YT_DLP_PATH", "yt-dlp"),
            vad_device=os.getenv("VAD_DEVICE", "cpu").strip().lower(),
            hindi_speaker=os.getenv("SARVAM_HINDI_SPEAKER", "shubh_hi_devotional"),
            english_speaker=os.getenv("SARVAM_ENGLISH_SPEAKER", "girish_en_devotional"),
            edge_hindi_voice=os.getenv("EDGE_HINDI_VOICE", "hi-IN-MadhurNeural"),
            edge_english_voice=os.getenv("EDGE_ENGLISH_VOICE", "en-IN-PrabhatNeural"),
        )

    def require_sarvam_key(self) -> str:
        if not self.sarvam_api_key:
            raise RuntimeError(
                "SARVAM_API_KEY is required for the Sarvam provider. Set it in the environment "
                "or in a local .env file (see .env.example)."
            )
        return self.sarvam_api_key


def discover_audio_devices() -> list[dict[str, Any]]:
    """Return sounddevice/PortAudio devices, including VB-Cable playback endpoints."""
    try:
        import sounddevice as sd

        return [
            {
                "index": index,
                "name": str(device["name"]),
                "inputs": int(device["max_input_channels"]),
                "outputs": int(device["max_output_channels"]),
                "default_samplerate": int(device["default_samplerate"]),
            }
            for index, device in enumerate(sd.query_devices())
        ]
    except Exception:
        return discover_pyaudio_devices()


def discover_pyaudio_devices() -> list[dict[str, Any]]:
    """Return devices with the indices PyAudio uses for direct microphone capture."""
    try:
        import pyaudio

        api = pyaudio.PyAudio()
        found: list[dict[str, Any]] = []
        try:
            for index in range(api.get_device_count()):
                device = api.get_device_info_by_index(index)
                found.append(
                    {
                        "index": index,
                        "name": str(device.get("name", "")),
                        "inputs": int(device.get("maxInputChannels", 0)),
                        "outputs": int(device.get("maxOutputChannels", 0)),
                        "default_samplerate": int(device.get("defaultSampleRate", 0)),
                    }
                )
        finally:
            api.terminate()
        return found
    except Exception:
        return []


def glossary_from_json(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load an optional JSON glossary, merging it over built-in terms."""
    import json
    from copy import deepcopy

    glossary = deepcopy(SERMON_GLOSSARY)
    if path is None:
        return glossary
    glossary_path = Path(path)
    with glossary_path.open("r", encoding="utf-8") as source:
        loaded = json.load(source)
    if not isinstance(loaded, dict):
        raise ValueError("Glossary JSON must be an object keyed by Telugu source terms.")
    for source_term, translations in loaded.items():
        if not isinstance(source_term, str) or not isinstance(translations, dict):
            raise ValueError("Each glossary entry must map a Telugu string to an object.")
        if not isinstance(translations.get("hi"), str) or not isinstance(translations.get("en"), str):
            raise ValueError(f"Glossary entry {source_term!r} must include string 'hi' and 'en' values.")
        glossary[source_term] = translations
    return glossary
