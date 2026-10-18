#!/usr/bin/env python3
"""Command-line and TUI entry point for LiveYT."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from liveyt import __version__
from liveyt.config import Settings, glossary_from_json
from liveyt.live_pipeline import LivePipeline, list_live_devices, validate_live_dependencies
from liveyt.local_video_dubber import LocalVideoDubber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liveyt",
        description=(
            "Translate Telugu prayer sermons into synchronized Hindi and English voiceovers. "
            "Run without --mode in a terminal for the interactive menu."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("youtube", "direct", "local_video"), help="Execution mode")
    parser.add_argument("--url", help="YouTube live or recorded video URL (--mode youtube)")
    parser.add_argument("--device", help="Microphone device index or name (--mode direct)")
    parser.add_argument("--file", help="Local MP4/MKV video path (--mode local_video)")
    parser.add_argument(
        "--engine",
        choices=("local", "cloud"),
        default="local",
        help="Local NLLB or Sarvam translation for local_video",
    )
    parser.add_argument(
        "--whisper-model",
        choices=("small", "large-v3-turbo"),
        default="small",
        help="faster-whisper model; small is the safer 6 GB GPU choice",
    )
    parser.add_argument(
        "--translation-model",
        default="facebook/nllb-200-distilled-600M",
        help="Hugging Face translation model for local_video",
    )
    parser.add_argument("--output-dir", help="Output directory for the two dubbed MP4 files")
    parser.add_argument("--glossary", help="Optional JSON glossary overlay keyed by Telugu terms")

    parser.add_argument(
        "--stt-transport",
        choices=("websocket", "rest"),
        default="websocket",
        help="Sarvam Saaras transport for live mode",
    )
    parser.add_argument(
        "--stt-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back from exhausted STT WebSocket reconnects to REST",
    )
    parser.add_argument(
        "--tts-provider",
        choices=("sarvam", "edge"),
        help="Live TTS provider; local video always uses Edge neural voices",
    )
    parser.add_argument(
        "--vad",
        choices=("silero", "energy"),
        default="silero",
        help="Live voice detector (Silero recommended)",
    )
    parser.add_argument(
        "--hindi-device", default="CABLE-A Input", help="VB-Cable A playback device index/name"
    )
    parser.add_argument(
        "--english-device", default="CABLE-B Input", help="VB-Cable B playback device index/name"
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Do not open VB-Cable/sounddevice outputs (useful with RTMP-only output)",
    )
    parser.add_argument(
        "--rtmp-url", help="Optional RTMP(S) destination with black 720p slate; keep stream keys private"
    )
    parser.add_argument(
        "--rtmp-language",
        choices=("hindi", "english"),
        default="english",
        help="Which translated mix to send to RTMP",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=10.0,
        help="Live mix delay in seconds to allow translation/TTS to finish",
    )
    parser.add_argument("--max-inflight", type=int, default=2, help="Maximum concurrent phrase jobs")
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=0.0,
        help="Stop live mode after this many seconds (0 means no limit)",
    )
    parser.add_argument(
        "--keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable P pause/resume and Q stop controls in a TTY",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List input/output audio devices and exit"
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def select_mode_tui(args: argparse.Namespace) -> argparse.Namespace:
    print("\nLiveYT — Telugu Sermon Dubbing")
    print("  1) YouTube live/recorded stream → Hindi + English live audio")
    print("  2) Direct microphone/hall mixer → Hindi + English live audio")
    print("  3) Local video → output_hindi.mp4 + output_english.mp4")
    selected = input("Choose mode [1-3]: ").strip()
    mapping = {"1": "youtube", "2": "direct", "3": "local_video"}
    if selected not in mapping:
        raise ValueError("Choose 1, 2, or 3")
    args.mode = mapping[selected]
    if args.mode == "youtube" and not args.url:
        args.url = input("YouTube URL: ").strip()
    elif args.mode == "direct" and args.device is None:
        args.device = input("Microphone device index/name (blank for system default): ").strip() or None
    elif args.mode == "local_video" and not args.file:
        args.file = input("Input MP4/MKV path: ").strip()
        engine = input("Translation engine [local/cloud] (local): ").strip().lower()
        if engine:
            args.engine = engine
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode is None:
        if not sys.stdin.isatty():
            parser.print_help()
            raise SystemExit(2)
        select_mode_tui(args)
    if args.mode == "youtube" and not args.url:
        parser.error("--mode youtube requires --url <YOUTUBE_URL>")
    if args.mode == "direct" and args.device is not None and str(args.device).strip() == "":
        args.device = None
    if args.mode == "local_video" and not args.file:
        parser.error("--mode local_video requires --file <INPUT_MP4_OR_MKV>")
    if args.latency < 0:
        parser.error("--latency must be zero or greater")
    if args.max_inflight < 1:
        parser.error("--max-inflight must be at least 1")
    if args.max_runtime < 0:
        parser.error("--max-runtime must be zero or greater")
    if args.rtmp_url and not args.rtmp_url.lower().startswith(("rtmp://", "rtmps://")):
        parser.error("--rtmp-url must begin with rtmp:// or rtmps://")
    if args.mode == "local_video" and args.tts_provider not in (None, "edge"):
        parser.error("Local video mode uses Edge TTS to avoid paid TTS API calls; choose --tts-provider edge")


async def _run_live(settings: Settings, args: argparse.Namespace) -> None:
    await LivePipeline(settings, args).run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_devices:
        list_live_devices()
        return 0
    _validate_args(parser, args)
    settings = Settings.from_environment()
    settings.glossary = glossary_from_json(args.glossary)
    settings.vad_backend = args.vad

    if args.mode in {"youtube", "direct"}:
        if args.tts_provider is None:
            args.tts_provider = "sarvam"
        try:
            validate_live_dependencies(args, settings)
            asyncio.run(_run_live(settings, args))
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Interrupted by user")
            return 130
        except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
            logging.getLogger(__name__).error("Live run failed: %s", exc)
            return 1
        return 0

    if args.mode == "local_video":
        try:
            if args.engine == "cloud":
                logging.getLogger(__name__).warning(
                    "Cloud translation calls Sarvam Mayura and may incur API charges; "
                    "use --engine local for no paid API calls."
                )
            dubber = LocalVideoDubber(
                settings,
                input_file=Path(args.file),
                engine=args.engine,
                whisper_model=args.whisper_model,
                output_dir=args.output_dir,
                translation_model=args.translation_model,
                glossary=settings.glossary,
            )
            hindi_path, english_path = asyncio.run(dubber.run())
            print(f"Hindi video:   {hindi_path}")
            print(f"English video: {english_path}")
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Interrupted by user")
            return 130
        except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
            logging.getLogger(__name__).error("Local video dubbing failed: %s", exc)
            return 1
        return 0

    parser.error("Choose --mode youtube, direct, or local_video")
    return 2


if __name__ == "__main__":
    sys.exit(main())
