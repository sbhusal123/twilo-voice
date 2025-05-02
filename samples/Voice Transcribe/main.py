from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse

import json
import traceback
import base64
import audioop
import torch
import numpy as np


from silero_vad import load_silero_vad,  get_speech_timestamps


import torchaudio

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch

from pydub import AudioSegment
from io import BytesIO

import edge_tts
import io

import wave

import ollama


resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

vad_model = load_silero_vad()


async def text_to_pcm_bytes(text: str) -> bytes:
    tts = edge_tts.Communicate(text, voice="en-US-JennyNeural")
    pcm_data = bytearray()

    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            pcm_data.extend(chunk["data"])

    return bytes(pcm_data)


def pcm16_to_mulaw(pcm_bytes: bytes, sample_rate=16000) -> bytes:
    # 1. Wrap PCM bytes into WAV format in-memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)

    # 2. Load WAV from buffer
    waveform, _ = torchaudio.load(buffer)

    # 3. Resample to 8000 Hz
    resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)(waveform)

    # 4. Convert to int16 PCM
    int16_pcm = (resampled.squeeze().numpy() * 32768).astype(np.int16).tobytes()

    # 5. Encode to μ-law
    return audioop.lin2ulaw(int16_pcm, 2)

def pcm_bytes_to_tensor(pcm_bytes):
    np_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    waveform =  torch.tensor(np_array, dtype=torch.float32).unsqueeze(0) / 32768.0
    return resampler(waveform)

def transcribe_pcm_ulaw(pcm_ulaw_bytes):
    pcm_bytes = audioop.ulaw2lin(pcm_ulaw_bytes, 2)
    waveform = pcm_bytes_to_tensor(pcm_bytes)  # float32, [1, N], 16kHz
    result = asr_pipeline(waveform.squeeze(0).numpy(), chunk_length_s=10)
    return result["text"]


async def stream_tts_to_twilio(ai_reply: str, websocket, stream_sid: str):
    communicate = edge_tts.Communicate(text=ai_reply, voice="en-US-GuyNeural")
    audio_chunks = []

    # Collect TTS MP3 data
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    
    mp3_data = b"".join(audio_chunks)

    # Convert MP3 to 8kHz µ-law PCM using pydub
    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3") \
                       .set_frame_rate(8000) \
                       .set_channels(1) \
                       .set_sample_width(2)

    raw_pcm = audio.raw_data

    mulaw_pcm = audioop.lin2ulaw(raw_pcm, audio.sample_width)

    
    payload = base64.b64encode(mulaw_pcm).decode('ascii')
    message = json.dumps({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
    })
    await websocket.send_text(message)

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

system_prompt = """
You are a highly knowledgeable AI assistant. Always answer clearly, confidently, and naturally,
but keep your response short and meaningful. Do not list points or use bullet forms.

Your answers will be converted to speech, so use proper punctuation like commas, periods, and question marks.
Avoid long sentences or excessive detail. Speak in a natural, conversational tone.
Limit your response to 1–2 short sentences only.

If you dont know anything, respond im unsure about that.
"""


async def call_ollama_chat_llama3(messages: str) -> str:
    msg = [{
        'role': 'system',
        'content': system_prompt
    }]
    
    msg.extend(messages)

    # Send to Ollama with llama3 or any other model you have
    response = await ollama.AsyncClient().chat(
        model='llama3:8b',
        messages=msg,
        stream=False,
        options={"temperature": 0.1}
    )

    return response['message']['content']


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

                is_audio = False

                # process every 500ms chunk to check if voice is present
                if vad_end_timestamp - vad_start_timestamp >= vad_audio_length_ms:
                    pcm_bytes = audioop.ulaw2lin(vad_audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,
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
                    # note that: twilio sends us back with 20ms audio
                    speech_pause_seconds += 20/1000
                

                if speech_pause_seconds >= speech_pause_seconds_threshold:
                    print("Speech paused for 2 seconds.")
                    speech_pause_seconds = 0
                    
                    pcm_bytes = audioop.ulaw2lin(audio_bytes, 2)
                    speech_timestamps = get_speech_timestamps(
                        pcm_bytes_to_tensor(pcm_bytes),
                        vad_model,
                        return_seconds=True,
                    )

                    if speech_timestamps:
                        print("Yes audio found..")

                        transcribed_text = transcribe_pcm_ulaw(audio_bytes)
                        audio_bytes.clear()

                        print("Transcribed Text=>", transcribed_text)

                        if transcribed_text:
                            message_history.append({
                                "role": "user", "content": transcribed_text
                            })

                            # now call ollama model from here
                            text = await call_ollama_chat_llama3(message_history)
            
                            message_history.append({
                                "role": "assistant", "content": text
                            })

                            await stream_tts_to_twilio(
                                ai_reply=text,
                                websocket=websocket,
                                stream_sid=data["streamSid"]
                            )

                            print("Response:::", text)
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
