# Architecture and operating behavior

## Live mode

1. `YoutubeAudioSource` starts yt-dlp and pipes its audio container into ffmpeg; `DirectMicAudioSource` captures mono PCM through PyAudio. Both yield timestamped 100 ms, 16 kHz mono signed-PCM frames.
2. `VADPhraseSegmenter` evaluates 512-sample frames with Silero VAD, keeps pre-roll, closes phrases on silence, and joins short adjacent utterances toward the 2–4 second target. A 0.8 second join window prevents short utterances from remaining buffered indefinitely. It only uses the 8-second hard boundary when speech remains continuous.
3. A single persistent Saaras v3 WebSocket transcriber serializes STT turns. It reconnects with exponential backoff (0.5 seconds up to 12 seconds, with jitter), does not retry explicit auth/quota closes, and optionally falls back to the Saaras REST endpoint. REST mode is also selectable directly.
4. Hindi and English Mayura requests are issued concurrently. Both Bulbul requests (or both Edge TTS requests) are also issued concurrently.
5. `LiveMixTimeline` buffers source samples for `--latency` seconds, schedules each synthesized phrase at its original sample timestamp, keeps Telugu at 0.15 gain, and adds the translated waveform at 1.0 gain. The renderer produces separate Hindi and English mixes. VB-Cable writes consume both mixes. Optional ffmpeg RTMP output adds a black 720p H.264 slate and AAC audio; it carries only the selected Hindi or English mix and does not relay source video.

Live dubbing necessarily has latency. Set `--latency` longer than the worst expected STT + two translations + two TTS calls. If a voice arrives after its sample window has already passed, late leading samples are trimmed and the event is logged rather than playing out of sync. Monitor this warning during a rehearsal.

## Local video mode

1. ffmpeg extracts the first audio stream as mono 16 kHz PCM WAV; ffprobe records the source duration.
2. faster-whisper materializes timestamped Telugu segments in CUDA `int8_float16` (or CPU `int8` when CUDA is unavailable). One ASR model worker is used. The Whisper object is deleted and `torch.cuda.empty_cache()` is called before translation.
3. `--engine local` loads NLLB-200 distilled 600M on CPU and translates each Telugu segment into Hindi and English in sequence. `--engine cloud` instead runs the two Mayura calls concurrently. This is the only local-video mode that uses Sarvam translation.
4. Edge neural TTS synthesizes Hindi and English concurrently per segment. `ffmpeg atempo` changes playback speed and the resulting clip is padded/trimmed to precisely the segment's original timestamp window.
5. A gain WAV contains 0.15 during every dubbed segment and 1.0 elsewhere. ffmpeg multiplies the original audio by this envelope, mixes the full-level voiceover, copies the video stream (`-c:v copy`) and encodes the new audio as AAC.

## Glossary behavior

Sarvam Mayura's public text-translation request has no custom glossary field. LiveYT therefore performs a deterministic normalization pass after translation: it replaces exact Telugu terms that remain untranslated and configured aliases in the target language. This improves consistency but cannot guarantee the translation model uses a term that it did not return. The same normalization applies to NLLB output.

## Secrets

`SARVAM_API_KEY` is read from the environment or `.env`. RTMP URLs are not included in application log messages, but are necessarily passed to the ffmpeg child process. Do not paste full RTMP URLs into public logs or tickets.
