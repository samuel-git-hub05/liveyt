import numpy as np
import pytest

from liveyt.audio import LiveMixTimeline, pcm16_wav, resample_audio, wav_to_float32


def test_pcm16_wav_round_trip_mono():
    source = np.asarray([0.0, 0.25, -0.5, 0.99], dtype=np.float32)
    wav_data = pcm16_wav((source * 32767).astype("<i2").tobytes(), 16_000)
    decoded, rate = wav_to_float32(wav_data)
    assert rate == 16_000
    assert decoded.shape == source.shape
    assert np.allclose(decoded, source, atol=0.001)


def test_wav_decoder_downmixes_stereo():
    stereo = np.asarray([[0.5, -0.5], [0.25, 0.25]], dtype=np.float32)
    pcm = (stereo * 32767).astype("<i2").tobytes()
    wav_data = pcm16_wav(pcm, 24_000, channels=2)
    decoded, rate = wav_to_float32(wav_data)
    assert rate == 24_000
    assert decoded.shape == (2,)
    assert abs(float(decoded[0])) < 0.001
    assert decoded[1] == pytest.approx(0.25, abs=0.001)


def test_resample_audio_keeps_expected_duration():
    source = np.sin(np.linspace(0, 2 * np.pi, 16_000, endpoint=False)).astype(np.float32)
    converted = resample_audio(source, 16_000, 48_000)
    assert len(converted) == 48_000
    assert converted.dtype == np.float32


def test_live_mix_has_15_percent_telugu_and_full_voice_overlay():
    timeline = LiveMixTimeline(
        sample_rate=16_000,
        background_gain=0.15,
        latency_seconds=1.0,
        block_samples=1_600,
    )
    original = (np.full(16_000, 16_384, dtype="<i2")).tobytes()  # about +0.5
    assert timeline.add_original(0, original) == []
    timeline.add_overlay("hindi", 0, np.full(16_000, 0.25, dtype=np.float32), 16_000)
    blocks = timeline.add_original(16_000, original)
    assert len(blocks) == 10
    assert blocks[0].hindi[0] == pytest.approx(0.325, abs=0.002)
    assert blocks[0].english[0] == pytest.approx(0.075, abs=0.002)
