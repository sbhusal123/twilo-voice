# Constructing an Audio File Out of Chunks Recieved

Now we can construct the audio files from the audio bytes stored in the variable.


```python
"""
Decoding audio chunks recieved from websockets
"""


# fastapi
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse

# foo
import json
import base64

# audio encodings
import ffmpeg

app = FastAPI()

audio_buffer = bytearray()


def save_ulaw_to_wav(audio_data: bytes, out_path: str = "test.wav") -> str:
    try:
        process = (
            ffmpeg
            .input('pipe:0', format='mulaw', ar='8000', ac='1')
            .output(out_path, format='wav')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        process.stdin.write(audio_data)
        process.stdin.close()

        out, err = process.communicate()
        if process.returncode != 0:
            print("[FFmpeg Error]", err.decode())
            return None

        print(f"WAV saved to {out_path}")
        return out_path
    except Exception as ex:
        print(f"[Error] {ex}")
        return None

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

    global audio_buffer
    audio_buffer.clear()

    await websocket.accept()

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data["event"] == "start":
                audio_buffer.clear()


            elif data["event"] == "media":
                audio_bytes = base64.b64decode(data['media']['payload'])
                audio_buffer.extend(audio_bytes)

            elif data["event"] == "stop":
                print("Strem Stopped..")
                if audio_buffer:
                    save_ulaw_to_wav(audio_buffer)
                await websocket.close()

    except Exception as e:
        print(f"Exception occured::: {e.__class__} :: e")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)

```

- On start of event, clears audio buffer.
- On any media event, media is base64 decoded and appeneded to buffer
- At the end, file is written to temporary file and media is created.

**Following is the spec of the audio bytes recieved from twilio websocket:**

**i. Audio Encoding:**

- Format: μ-Law (also called mu-Law or mulaw)

- File extension: The raw audio is typically not a .wav file but a raw byte stream.

**ii.Sample Rate:**

- Rate: 8000 Hz

- This means each second of audio consists of 8000 audio samples (since it's typically 8-bit).

**iii. Channels:**

- Mono: The audio stream is mono (single channel).

**iv. Audio Frame Size:**

- 8-bit: Each audio sample is 1 byte (8 bits) — since μ-Law encoding uses 8-bit samples.

> μ-Law Encoding: This encoding is commonly used for telephony (low bandwidth, speech encoding).``

```python
import ffmpeg

def save_ulaw_to_wav(audio_data: bytes, out_path: str = "test.wav") -> str:
    try:
        process = (
            ffmpeg
            .input('pipe:0', format='mulaw', ar='8000', ac='1')
            .output(out_path, format='wav')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        # Write raw audio bytes to stdin
        process.stdin.write(audio_data)
        process.stdin.close()

        # Wait for process to complete and capture stderr
        out, err = process.communicate()
        if process.returncode != 0:
            print("[FFmpeg Error]", err.decode())
            return None

        print(f"WAV saved to {out_path}")
        return out_path
    except Exception as ex:
        print(f"[Error] {ex}")
        return None
```
