import numpy as np

from liveyt.vad_ingestor import EnergyVADDetector, SpeechSegment, VADPhraseSegmenter


def _pcm(value: float, samples: int) -> bytes:
    return (np.full(samples, value, dtype=np.float32) * 32767).astype("<i2").tobytes()


def test_energy_vad_segments_on_silence_and_preserves_phrase():
    detector = EnergyVADDetector(rms_threshold=0.02)
    segmenter = VADPhraseSegmenter(
        backend="energy",
        detector=detector,
        silence_ms=200,
        min_speech_ms=100,
        max_phrase_seconds=4,
        hard_phrase_seconds=8,
    )
    data = _pcm(0.15, 3 * 16_000) + _pcm(0.0, 8_000)
    phrases = []
    for offset in range(0, len(data), 3_200):
        phrases.extend(segmenter.feed(data[offset : offset + 3_200]))
    phrases.extend(segmenter.flush())
    assert len(phrases) == 1
    assert phrases[0].start_sample == 0
    assert 3.0 <= phrases[0].duration_seconds < 3.4
    assert phrases[0].pcm_s16le.startswith(_pcm(0.15, 16))


def test_pause_reset_keeps_global_source_timeline():
    segmenter = VADPhraseSegmenter(
        backend="energy",
        detector=EnergyVADDetector(rms_threshold=0.02),
        silence_ms=100,
        min_speech_ms=100,
        max_phrase_seconds=4,
        hard_phrase_seconds=8,
    )
    segmenter.reset(start_sample=32_000)
    data = _pcm(0.15, 2 * 16_000) + _pcm(0.0, 4_096)
    phrases = segmenter.feed(data)
    assert len(phrases) == 1
    assert phrases[0].start_sample == 32_000


def test_short_natural_phrases_join_into_two_to_four_second_window():
    segmenter = VADPhraseSegmenter(
        backend="energy",
        detector=EnergyVADDetector(),
        min_phrase_seconds=2.0,
        max_phrase_seconds=4.0,
        hard_phrase_seconds=8.0,
    )
    first = SpeechSegment(0, 16_000, _pcm(0.1, 16_000))
    second = SpeechSegment(18_000, 34_000, _pcm(0.1, 16_000))
    assert segmenter._aggregate_short_phrase(first, force=False) == []
    joined = segmenter._aggregate_short_phrase(second, force=False)
    assert len(joined) == 1
    assert joined[0].start_sample == 0
    assert joined[0].end_sample == 34_000
    assert 2.0 <= joined[0].duration_seconds <= 4.0


def test_noise_under_min_speech_is_dropped():
    segmenter = VADPhraseSegmenter(
        backend="energy",
        detector=EnergyVADDetector(rms_threshold=0.02),
        silence_ms=100,
        min_speech_ms=240,
        max_phrase_seconds=4,
        hard_phrase_seconds=8,
    )
    phrase = segmenter.feed(_pcm(0.15, 100) + _pcm(0.0, 2_000))
    assert phrase == []
