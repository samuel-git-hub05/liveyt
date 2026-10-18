import asyncio
import base64
import json

import httpx

from liveyt.audio import pcm16_wav
from liveyt.config import Settings
from liveyt.sarvam_client import SarvamClient, SpeechWebSocketTranscriber


def test_sarvam_rest_endpoints_use_expected_contracts():
    wav_data = pcm16_wav(b"\x00\x00" * 512, 16_000)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/speech-to-text":
            return httpx.Response(200, json={"transcript": "దేవుడు ప్రార్థన"})
        if request.url.path == "/translate":
            payload = json.loads(request.content)
            translated = "ईश्वर की प्रार्थना" if payload["target_language_code"] == "hi-IN" else "prayer to God"
            return httpx.Response(200, json={"translated_text": translated})
        if request.url.path == "/text-to-speech":
            return httpx.Response(200, json={"audios": [base64.b64encode(wav_data).decode()]})
        return httpx.Response(404)

    async def run():
        client = SarvamClient(Settings(sarvam_api_key="test-key"))
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://api.sarvam.ai",
            headers={"api-subscription-key": "test-key"},
            transport=httpx.MockTransport(handler),
        )
        try:
            transcript = await client.transcribe_rest(b"\x00\x00" * 512)
            hindi = await client.translate(transcript, "hi-IN")
            english = await client.translate(transcript, "en-IN")
            audio = await client.synthesize(hindi, "hi-IN")
            return transcript, hindi, english, audio
        finally:
            await client.close()

    transcript, hindi, english, audio = asyncio.run(run())
    assert transcript == "దేవుడు ప్రార్థన"
    assert hindi == "परमेश्वर की प्रार्थना"
    assert english == "prayer to God"
    assert audio.startswith(b"RIFF")
    assert [request.url.path for request in calls] == [
        "/speech-to-text",
        "/translate",
        "/translate",
        "/text-to-speech",
    ]
    assert "saaras:v3" in calls[0].content.decode(errors="ignore")


def test_saaras_websocket_phrase_sends_wav_and_flush():
    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.closed = False

        async def send(self, message):
            self.sent.append(message)

        async def recv(self):
            return json.dumps({"type": "data", "data": {"transcript": "తెలుగు ప్రార్థన"}})

        async def close(self):
            self.closed = True

    async def run():
        socket = FakeSocket()
        transcriber = SpeechWebSocketTranscriber(
            api_key="test-key",
            base_url="https://api.sarvam.ai",
            model="saaras:v3",
        )

        async def fake_connect():
            transcriber._socket = socket
            return socket

        transcriber._connect = fake_connect
        try:
            result = await transcriber.transcribe(b"\x00\x00" * 512)
        finally:
            await transcriber.close()
        return result, socket

    result, socket = asyncio.run(run())
    assert result == "తెలుగు ప్రార్థన"
    audio_message = json.loads(socket.sent[0])
    decoded_wav = base64.b64decode(audio_message["audio"]["data"])
    assert decoded_wav.startswith(b"RIFF")
    assert audio_message["audio"]["encoding"] == "audio/wav"
    assert json.loads(socket.sent[1]) == {"type": "flush"}
    assert socket.closed
