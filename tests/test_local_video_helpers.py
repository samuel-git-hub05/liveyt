import numpy as np
import pytest

from liveyt.audio import pcm16_wav, wav_to_float32
from liveyt.config import Settings
from liveyt.local_video_dubber import (
    DubbedSegment,
    LocalVideoDubber,
    TranscriptSegment,
    _looks_like_cuda_oom,
)


def test_transcript_segment_timestamp_and_text():
    segment = TranscriptSegment(1.0, 2.5, " ప్రార్థన")
    assert segment.start == 1.0
    assert segment.end == 2.5
    assert segment.text.strip() == "ప్రార్థన"


def test_cuda_oom_detection():
    assert _looks_like_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not _looks_like_cuda_oom(RuntimeError("invalid audio file"))


def test_local_video_ducks_only_when_that_language_has_voice(tmp_path, monkeypatch):
    input_file = tmp_path / "sermon.mp4"
    input_file.touch()
    settings = Settings(sample_rate=16_000, background_gain=0.15)
    dubber = LocalVideoDubber(settings, input_file=input_file)
    monkeypatch.setattr(dubber, "_fit_audio", lambda samples, rate, target: samples[:target])
    voice_wav = pcm16_wav((np.full(16_000, 0.2) * 32767).astype("<i2").tobytes(), 16_000)
    items = [
        DubbedSegment(
            TranscriptSegment(1.0, 2.0, "ప్రార్థన"),
            "प्रार्थना",
            "",
            voice_wav,
            b"",
        )
    ]
    hindi_voice = tmp_path / "hindi.wav"
    hindi_gain = tmp_path / "hindi_gain.wav"
    english_voice = tmp_path / "english.wav"
    english_gain = tmp_path / "english_gain.wav"
    dubber._assemble_track(items, "hindi", 3.0, hindi_voice, hindi_gain)
    dubber._assemble_track(items, "english", 3.0, english_voice, english_gain)

    hindi_mask, _ = wav_to_float32(hindi_gain.read_bytes())
    english_mask, _ = wav_to_float32(english_gain.read_bytes())
    assert hindi_mask[0] == pytest.approx(1.0, abs=0.001)
    assert hindi_mask[16_000] == pytest.approx(0.15, abs=0.001)
    assert hindi_mask[-1] == pytest.approx(1.0, abs=0.001)
    assert np.allclose(english_mask, 1.0, atol=0.001)


def test_video_dubber_rejects_invalid_engine(tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    try:
        LocalVideoDubber(Settings(), input_file=source, engine="unknown")
    except ValueError as exc:
        assert "engine" in str(exc)
    else:
        raise AssertionError("invalid engine should fail fast")
