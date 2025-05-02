from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse

import json
import traceback
import base64
import audioop
import torch
import numpy as np


from silero_vad import load_silero_vad,  get_speech_timestamps


import torchaudio.transforms as T

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch

resampler = T.Resample(orig_freq=8000, new_freq=16000)

vad_model = load_silero_vad()


def pcm_bytes_to_tensor(pcm_bytes):
    np_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    waveform =  torch.tensor(np_array, dtype=torch.float32).unsqueeze(0) / 32768.0
    return resampler(waveform)

def transcribe_pcm_ulaw(pcm_ulaw_bytes):
    pcm_bytes = audioop.ulaw2lin(pcm_ulaw_bytes, 2)
    waveform = pcm_bytes_to_tensor(pcm_bytes)  # float32, [1, N], 16kHz
    result = asr_pipeline(waveform.squeeze(0).numpy(), chunk_length_s=10)
    return result["text"]

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)


asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1,
)



# Initialize FastAPI
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

    speech_pause_seconds = 0

    vad_start_timestamp = 0
    vad_end_timestamp = 0

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

                is_audio = False

                # process every 500ms chunk to check if voice is present
                if vad_end_timestamp - vad_start_timestamp >= 500:
                    pcm_bytes = audioop.ulaw2lin(vad_audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
                    )

                    print("speech_timestamps::", speech_timestamps)

                    
                    if speech_timestamps:
                        is_audio = True
                        speech_pause_seconds = 0
                    else:
                        is_audio = False
                        speech_pause_seconds += 20/1000

                    vad_audio_bytes.clear()
                    vad_start_timestamp = vad_end_timestamp

                if is_audio:
                    speech_pause_seconds = 0
                else:
                    speech_pause_seconds += 20/1000
                

                if speech_pause_seconds >= 2:
                    print("Speech paused for 2 seconds.")
                    speech_pause_seconds = 0
                    
                    pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
                    )

                    if speech_timestamps:
                        print("Yes audio found..")                        

                        result = transcribe_pcm_ulaw(audio_bytes)

                        print("Transcribed text = ", result)

                        audio_bytes.clear()
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
