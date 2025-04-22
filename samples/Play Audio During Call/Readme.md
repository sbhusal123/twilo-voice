# Sending Audio Bytes To Play Through Twilio WebSocket

We can send the audio to the twilio websocket, the json structure looks like below:

```json
    dt = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": b64_payload
            }
    }
```

Here:
- ``stream_sid`` is the stream id => ``data["start"]["streamSid"]`` 
- ``base64_payload`` is a binary audio payload. It must have
    - Frame rate => 8000
    - Channels => 1
    - Format of ulaw

We can built a base64 payload with following function:

```python
def mp3_to_mulaw_b64(path: str) -> str:
    """
    Read an MP3 file, convert it to 8kHz mono μ‑law, and return
    a base64-encoded payload (no headers).
    """
    # 1. Decode MP3 and normalize to 8kHz mono, 16-bit PCM
    audio = AudioSegment.from_file(path) \
                        .set_frame_rate(8000) \
                        .set_channels(1) \
                        .set_sample_width(2)

    # 2. Extract raw 16-bit PCM data
    raw_pcm = audio.raw_data

    # 3. μ‑law encode (audioop.lin2ulaw expects sample_width=2 for 16-bit)
    mulaw_pcm = audioop.lin2ulaw(raw_pcm, audio.sample_width)

    # 4. Base64 encode and return ASCII string
    return base64.b64encode(mulaw_pcm).decode('ascii')
```

## Implementing With FastAPI

```python
"""
Sending a bytes of an audio file to play to twilio websocket
"""


# fastapi
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse

import json

import audioop

from pydub import AudioSegment
import base64

app = FastAPI()



def mp3_to_mulaw_b64(path: str) -> str:
    """
    Read an MP3 file, convert it to 8kHz mono μ‑law, and return
    a base64-encoded payload (no headers).
    """
    # 1. Decode MP3 and normalize to 8kHz mono, 16-bit PCM
    audio = AudioSegment.from_file(path) \
                        .set_frame_rate(8000) \
                        .set_channels(1) \
                        .set_sample_width(2)

    # 2. Extract raw 16-bit PCM data
    raw_pcm = audio.raw_data

    # 3. μ‑law encode (audioop.lin2ulaw expects sample_width=2 for 16-bit)
    mulaw_pcm = audioop.lin2ulaw(raw_pcm, audio.sample_width)

    # 4. Base64 encode and return ASCII string
    return base64.b64encode(mulaw_pcm).decode('ascii')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Whisper + Ollama server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Respond with TwiML to route call to WebSocket."""
    from twilio.twiml.voice_response import VoiceResponse, Connect

    response = VoiceResponse()
    response.say("Please wait while we connect your call to the AI voice assistant.")
    response.pause(length=1)
    response.say("Okay, you can start talking now.")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")

    audio_file_path = 'sample-3s.mp3'

    await websocket.accept()


    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                b64_payload = mp3_to_mulaw_b64(audio_file_path)

                dt = json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": b64_payload
                    }
                })

                await websocket.send_text(dt)
        
    except Exception as e:
        print(f"Exception occured: {e.__class__}")
        traceback.print_exc()
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)

```

So, on ``data["event"] == "start"`` we send the payload to be played with a proper json format.

```python
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                b64_payload = mp3_to_mulaw_b64(audio_file_path)

                dt = json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": b64_payload
                    }
                })
```

