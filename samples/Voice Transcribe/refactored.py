from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse

import json
import traceback
import base64
import audioop
import time


from silero_vad import load_silero_vad,  get_speech_timestamps


import nltk

from utils import (
    call_ollama_chat_llama3,
    transcribe_pcm_ulaw,
    pcm_bytes_to_tensor,
    stream_tts_to_twilio,
    is_complete_sentence
)

nltk.download('punkt_tab')

vad_model = load_silero_vad()


app = FastAPI()

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
    response.say("Connected to voice assistant, start speaking after the ring.")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    Accepts Twilio Media Stream messages, decodes µ-law to PCM,
    applies noise reduction, VAD, and transcribes with Whisper.
    """
    await websocket.accept()
    audio_bytes = bytearray()
    vad_audio_bytes = bytearray()

    
    speech_pause_seconds_threshold = 2
    speech_pause_seconds = 0

    vad_start_timestamp = 0
    vad_end_timestamp = 0
    vad_audio_length_ms = 500

    
    message_history = []

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                pass

            elif event == "media":
                payload = data["media"]["payload"]
                payload = base64.b64decode(payload)

                vad_audio_bytes.extend(payload)
                audio_bytes.extend(payload)
                
                vad_end_timestamp = int(data["media"]["timestamp"])

                # process every 500ms chunk to check if voice is present
                if vad_end_timestamp - vad_start_timestamp >= vad_audio_length_ms:

                    diff = vad_end_timestamp - vad_start_timestamp
                    pcm_bytes = audioop.ulaw2lin(vad_audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,
                    )

                    print("speech_timestamps::", speech_timestamps)

                    
                    if speech_timestamps:
                        speech_pause_seconds = 0
                    else:
                        speech_pause_seconds += diff/1000

                    vad_audio_bytes.clear()
                    vad_start_timestamp = vad_end_timestamp
                

                if speech_pause_seconds >= speech_pause_seconds_threshold:
                    print("Speech paused for 2 seconds.")
                    speech_pause_seconds = 0
                    
                    pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,
                    )

                    # if there is speech in audio process buffer
                    if speech_timestamps:
                        print("Yes audio found..")

                        start = time.time()
                        transcribed_text = transcribe_pcm_ulaw(audio_bytes)
                        end = time.time()
                        print(f"Text transcribe took: {end-start:5f} seconds")
                        audio_bytes.clear()

                        print("Transcribed Text=>", transcribed_text)

                        if transcribed_text:
                            message_history.append({
                                "role": "user", "content": transcribed_text
                            })

                            start = time.time()

                            full_response = ""
                            buffer = ""

                            
                            async for part in call_ollama_chat_llama3(message_history):
                                print("Part::", part)

                                buffer += part
                                full_response += part

                                if is_complete_sentence(buffer):
                                    await stream_tts_to_twilio(
                                        ai_reply=buffer,
                                        websocket=websocket,
                                        stream_sid=data["streamSid"]
                                    )
                                    buffer = ""

                                
                            message_history.append({
                                "role": "assistant", "content": full_response
                            })

                            print("Response:::", full_response)
                    else:
                        print("No audio to process")

            elif event == "stop":
                await websocket.close()

    except Exception as exc:
        print(f"❌ Exception in media stream: {exc}")
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)

