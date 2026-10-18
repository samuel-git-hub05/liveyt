# LiveYT — Telugu Sermon Dubber

LiveYT is a Python 3.11+ application for translating Telugu prayer sermons into spoken Hindi and English. It supports YouTube audio, a direct microphone/hall mixer, and local video dubbing. The live path uses Sarvam Saaras v3, Mayura and Bulbul (or Edge TTS); the default local-video path uses faster-whisper, NLLB and Edge's neural voices.

> **Important limitations / costs**
>
> * Live Sarvam STT, translation and Sarvam TTS are cloud services and may incur charges. Live mode requires `SARVAM_API_KEY`.
> * `local_video --engine local` makes no paid Sarvam API calls. It downloads/uses the NLLB model locally and uses Microsoft's Edge neural TTS endpoint through `edge-tts` (internet required; not an offline voice model). Check model/voice terms for your use case. NLLB-200-distilled-600M is distributed under a non-commercial research license; do not use it commercially unless you have appropriate rights.
> * `local_video --engine cloud` sends transcript text to Sarvam Mayura and may incur API charges. STT remains local and TTS uses Edge.
> * A 6 GB GPU cannot be promised to stay below a fixed VRAM ceiling on every Windows driver/model combination. The default `small` Whisper model uses `int8_float16` on CUDA; large-v3-turbo is optional, is only selected when the preflight free-VRAM check permits it, and falls back to small/CPU on detected CUDA OOM. NLLB runs on CPU after Whisper is unloaded. The code calls `torch.cuda.empty_cache()` after ASR. For the most reliable <4.5 GiB target, leave `--whisper-model small` selected and close GPU-heavy apps.
> * Video is stream-copied (`-c:v copy`) for speed. The source video codec must be compatible with MP4; otherwise ffmpeg reports an error rather than silently re-encoding it.

## Folder structure

```text
liveyt/
├── main.py                      # CLI and interactive TUI entry point
├── liveyt/
│   ├── __init__.py
│   ├── config.py                # Settings, device discovery, devotional glossary
│   ├── audio.py                 # PCM/WAV, delayed live mixer, VB-Cable and RTMP sinks
│   ├── vad_ingestor.py          # yt-dlp/ffmpeg, PyAudio, Silero VAD, extraction
│   ├── sarvam_client.py         # Saaras WS/REST, Mayura REST, Bulbul and Edge TTS
│   ├── live_pipeline.py         # Concurrent live dual-language pipeline
│   └── local_video_dubber.py    # Sequential CUDA ASR, NLLB/cloud translation, muxing
├── tests/
├── docs/
├── scripts/
├── LICENSE
├── .github/workflows/ci.yml
├── requirements.txt             # Small shared Python dependencies
├── requirements-live.txt        # Live-mode dependencies
├── requirements-local-video.txt # Local-video dependencies
├── requirements-dev.txt
├── glossary.example.json
├── .env.example
└── pyproject.toml
```

## Windows 11 setup

Open PowerShell. Install Python 3.11 and ffmpeg (winget package names can vary by Windows image; Chocolatey is an alternative):

```powershell
winget install --exact --id Python.Python.3.11
winget install --exact --id Gyan.FFmpeg.Shared
# Or, if you use Chocolatey: choco install ffmpeg-full -y
```

Close/reopen PowerShell and verify:

```powershell
py -3.11 --version
ffmpeg -version
ffprobe -version
```

If `ffmpeg`/`ffprobe` are not on `PATH`, add the folder containing both executables to Windows `PATH`, or set `FFMPEG_PATH` and `FFPROBE_PATH` in `.env`.

### CUDA/PyTorch

Install a PyTorch + torchaudio build matching your installed NVIDIA driver/CUDA runtime **before** installing the optional requirements. The example below is for the CUDA 12.8 wheel channel; if that channel is not offered for your Python/driver combination, use the current Windows/Pip command from the [PyTorch install selector](https://pytorch.org/get-started/locally/):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`faster-whisper` uses CTranslate2 rather than PyTorch for inference. A CUDA-enabled PyTorch check alone does **not** guarantee CTranslate2 can load CUDA: install the CUDA/cuDNN runtime DLL versions supported by the installed CTranslate2 build and ensure they are on `PATH`. If CTranslate2 reports that CUDA libraries are missing, the application continues with CPU Whisper when no CTranslate2 CUDA device is found.

### Python dependencies

Install only the mode(s) you need:

```powershell
# Shared HTTP, config and WebSocket dependencies
python -m pip install -r requirements.txt

# Live YouTube/microphone/VB-Cable use
python -m pip install -r requirements-live.txt

# Local MP4/MKV dubbing (also installs faster-whisper, Transformers and NLLB runtime deps)
python -m pip install -r requirements-local-video.txt

# Install the liveyt console command (optional; python main.py also works)
python -m pip install -e .

# Optional development/test tooling
python -m pip install -r requirements-dev.txt
```

`PyAudio` normally installs a Windows wheel from PyPI. If pip asks for a compiler, update pip and retry; do not download unofficial binaries from untrusted sites. The requirements deliberately do not pin a PyTorch CUDA build so that your existing CUDA installation is not replaced with an incompatible wheel.

### VB-CABLE A+B and OBS

1. Download **VB-CABLE A+B** from the official [VB-Audio Virtual Cable page](https://vb-audio.com/Cable/). Use the A+B package (not only the single Cable A package).
2. Extract and locate the official installer in PowerShell:

   ```powershell
   Start-Process "https://vb-audio.com/Cable/"
   $zip = Read-Host "Full path to the downloaded VBCABLE_A_B_Driver_Pack45.zip"
   $dest = Join-Path $PWD "vbcable-ab"
   Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
   Get-ChildItem $dest -Recurse -Filter "*Setup*.exe" | Select-Object -ExpandProperty FullName
   ```

3. If the archive includes `VBCable_AB_PackSetup.exe`, run that combined setup **as Administrator**. If it contains separate Cable A and Cable B driver-pack archives, extract both and run each included `VBCABLE_Setup_x64.exe` as Administrator. Reboot Windows after installation. You can launch the combined installer from PowerShell with `Start-Process -FilePath (Join-Path $dest "VBCable_AB_PackSetup.exe") -Verb RunAs -Wait` if that is the filename in your download.
4. In Windows Sound settings, leave each cable's **Input** endpoint available as a playback device. Python writes audio to playback endpoints named approximately `CABLE-A Input` and `CABLE-B Input`; OBS receives recording endpoints named approximately `CABLE-A Output` and `CABLE-B Output`.
5. Run `python main.py --list-devices`. The `sounddevice` list is for VB-Cable playback outputs; the separately printed `PyAudio` indices are for `--device` microphone capture. Names or unique fragments can be passed to either selector.
6. In OBS, add two **Audio Input Capture** sources and choose Cable A Output for Hindi and Cable B Output for English. Route/monitor them to your broadcast mix as desired. Avoid monitoring the same cable back into itself.

VB-CABLE is a separate driver download/install; its Windows drivers need administrator approval and a reboot, so there is no reliable, supported `winget` command for installing the official A+B drivers silently. Install only the official package.

### API key

For live Sarvam mode, copy `.env.example` to `.env` and add your Sarvam subscription key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Never commit or share `.env`, a Sarvam key, or an RTMP stream key. Local-video mode with `--engine local` does not need a Sarvam key.

## Run

Use `python main.py` without `--mode` for a small interactive TUI, or supply the mode on the command line. `--list-devices` works without a mode.

### Mode 1 — YouTube live/recorded stream

```powershell
python main.py --mode youtube --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

Live audio is decoded by `yt-dlp | ffmpeg` to 16 kHz mono PCM. Silero VAD prefers silence/phrase boundaries around 2–4 seconds; it waits for a natural pause rather than cutting every four seconds. Continuous speech has an 8-second safety boundary. The ten-second default mix latency gives the dual translations/voiceovers time to complete before the corresponding Telugu audio reaches the outputs; adjust it with `--latency` if needed.

### Mode 2 — Direct mic / hall mixer input

```powershell
python main.py --mode direct --device 3
# Or select a device by a unique name fragment:
python main.py --mode direct --device "USB Audio"
# Omit --device to use Windows' default input device:
python main.py --mode direct
```

`P` then Enter toggles translation pause/resume (the Telugu background continues at 15%); `Q` then Enter stops. On Windows the single keys `P` and `Q` are also detected by the console. Use `--no-keyboard` for a non-interactive/headless run. Use `--no-stt-fallback` to disable REST fallback after the Saaras WebSocket reconnect policy is exhausted.

### Mode 3 — Local video to two dubbed MP4s

```powershell
python main.py --mode local_video --file "C:\sermons\prayer.mp4" --engine local
```

Outputs are `output_hindi.mp4` and `output_english.mp4` beside the source (or in `--output-dir`). Default `small` Whisper transcription runs in CUDA `int8_float16` when CTranslate2 detects CUDA; it automatically uses CPU/int8 otherwise. Once timestamped Telugu segments are materialized, Whisper is unloaded and the CUDA cache is cleared. Hindi and English translation is performed sequentially with the local NLLB-200 distilled 600M model on CPU. Edge TTS uses `hi-IN-MadhurNeural` and `en-IN-PrabhatNeural` by default. Each voice clip is fitted to the Whisper segment window with ffmpeg `atempo`; ffmpeg builds a gain envelope (15% Telugu under speech, 100% translated voice) and muxes AAC audio while copying the video stream.

The first local NLLB run downloads the model (several GB) from Hugging Face. If the NLLB model's license is unsuitable for your intended use, choose another compatible model with `--translation-model` or use the cloud engine:

```powershell
# Cloud Mayura translation may incur Sarvam API charges; ASR is still local.
python main.py --mode local_video --file "C:\sermons\prayer.mkv" --engine cloud
```

## Useful options

```text
--stt-transport websocket|rest   Saaras transport in live modes (default: websocket)
--stt-fallback / --no-stt-fallback
--tts-provider sarvam|edge       Live voice provider (default: Sarvam)
--vad silero|energy              Silero is recommended; energy is a simple fallback
--hindi-device / --english-device VB-Cable playback endpoint names/indices
--no-playback                    RTMP-only/headless live output
--rtmp-url rtmp(s)://...         Optional RTMP output with a black 720p slate
--rtmp-language hindi|english    Track sent to the one RTMP destination
--latency 10                     Live source delay in seconds
--max-inflight 2                 Concurrent phrase translations
--glossary glossary.example.json JSON overrides/extends the built-in glossary
--whisper-model small|large-v3-turbo
--output-dir PATH                Local-video output folder
```

### Glossary

The built-in Telugu-to-Hindi/English glossary is in `liveyt/config.py`. Create a JSON file with the same structure as `glossary.example.json`, then pass `--glossary path.json`. Mayura does not expose a custom glossary input field, so LiveYT normalizes listed Telugu terms left in translations and configured target-language aliases after translation; this cannot force an API model to choose a term it did not produce. NLLB output gets the same normalization.

### RTMP

RTMP output is optional and carries one chosen language (Hindi **or** English) plus 15% original Telugu. The headless ffmpeg sink supplies a black 720p H.264 slate with AAC audio so the RTMP destination receives a video-bearing FLV stream; it does not forward the original YouTube video. Use OBS with the VB-Cable routes when you need the source video or a designed slate. Provide a complete RTMP(S) URL, including the stream key, through `--rtmp-url`; the application redacts it from its logs.

## Verify / test

```powershell
python main.py --help
python main.py --list-devices
python -m pytest
```

Tests exercise audio conversion, VAD phrase boundaries, glossary normalization, and timestamped audio mixing without contacting Sarvam or requiring a GPU/audio device. For live use, test the device routes and stream latency with a short, private test stream before a public five-hour broadcast.
