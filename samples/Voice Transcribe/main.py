import json
import traceback
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse

from utils import get_transcribed_text_from_stream, \
    stream_tts_to_twilio, reset_audio_buffer

import ollama

import audioop
from pydub import AudioSegment
import base64

# Initialize FastAPI
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
    """
    Respond with TwiML to connect the call to our WebSocket media-stream endpoint.
    """
    from twilio.twiml.voice_response import VoiceResponse, Connect

    response = VoiceResponse()
    response.say("Please wait, connecting to the AI voice assistant.")
    response.pause(1)
    response.say("You are now connected to AI assiatant. You can start speaking after the ring")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


async def play_sound_file(file_path, data, websocket):
    stream_sid = data["streamSid"]
    b64_payload = mp3_to_mulaw_b64(file_path)

    dt = json.dumps({
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": b64_payload
        }
    })

    await websocket.send_text(dt)


import asyncio
lock = asyncio.Lock()
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    Accepts Twilio Media Stream messages, decodes µ-law to PCM,
    applies noise reduction, VAD, and transcribes with Whisper.
    """
    reset_audio_buffer()
    await websocket.accept()


    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                print("🚀 Stream started")
                reset_audio_buffer()

            elif event == "media":
                text = await get_transcribed_text_from_stream(data)
                # if there is a text
                if text:
                    ai_response = ollama.chat(
                        model='llama3:8b',
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a super smart voice call assistant"
                                " You will be given a text from the audio of what user said."
                                " Your response text will be converted into audio, so you need "
                                "to make sure number of words in your response is not more than 15 words."
                                "Do not make up anything, just answer the users answer in 15-20 words."
                            },
                            {
                                "role": "user",
                                "content": text
                            }
                        ]
                    )
                    await stream_tts_to_twilio(
                        ai_reply=ai_response["message"]["content"],
                        websocket=websocket,
                        stream_sid=data["streamSid"]
                    )                

            elif event == "stop":
                print("🛑 Stream stopped by Twilio")
                await websocket.close()

    except Exception as exc:
        print(f"❌ Exception in media stream: {exc}")
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
