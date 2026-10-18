import json

from liveyt.config import Settings, glossary_from_json
from liveyt.sarvam_client import apply_sermon_glossary


def test_defaults_have_sarvam_and_edge_voices(monkeypatch):
    monkeypatch.delenv("SARVAM_HINDI_SPEAKER", raising=False)
    monkeypatch.delenv("SARVAM_ENGLISH_SPEAKER", raising=False)
    settings = Settings.from_environment(env_file=None)
    assert settings.stt_model == "saaras:v3"
    assert settings.translation_model == "mayura:v1"
    assert settings.tts_model == "bulbul:v4-flash"
    assert settings.background_gain == 0.15
    assert settings.voiceover_gain == 1.0
    assert settings.edge_hindi_voice == "hi-IN-MadhurNeural"
    assert settings.edge_english_voice == "en-IN-PrabhatNeural"


def test_glossary_normalizes_target_aliases_and_telugu_terms():
    settings = Settings()
    hindi = apply_sermon_glossary("ईश्वर और ईसा मसीह के साथ దేవుడు", "hi", settings.glossary)
    assert "परमेश्वर" in hindi
    assert "यीशु मसीह" in hindi
    assert "దేవుడు" not in hindi


def test_json_glossary_extends_builtin_terms(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(
        json.dumps({"కొత్త పదం": {"hi": "नया शब्द", "en": "new term", "hi_aliases": []}}, ensure_ascii=False),
        encoding="utf-8",
    )
    glossary = glossary_from_json(path)
    assert glossary["దేవుడు"]["hi"] == "परमेश्वर"
    assert glossary["కొత్త పదం"]["en"] == "new term"
