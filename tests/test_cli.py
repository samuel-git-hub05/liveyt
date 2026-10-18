from main import build_parser


def test_cli_accepts_all_execution_modes():
    parser = build_parser()
    assert parser.parse_args(["--mode", "youtube", "--url", "https://youtu.be/example"]).mode == "youtube"
    assert parser.parse_args(["--mode", "direct", "--device", "2"]).device == "2"
    local = parser.parse_args(["--mode", "local_video", "--file", "sermon.mp4"])
    assert local.engine == "local"
    assert local.whisper_model == "small"


def test_cli_defaults_to_websocket_vad_and_parallel_phrases():
    args = build_parser().parse_args(["--mode", "direct"])
    assert args.stt_transport == "websocket"
    assert args.vad == "silero"
    assert args.max_inflight == 2
