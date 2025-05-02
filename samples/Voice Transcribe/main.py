from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, HTMLResponse

import json
import traceback
import base64
import audioop
import torch
import numpy as np
import time


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

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab')




resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

vad_model = load_silero_vad()

def is_complete_sentence(text):
    text = text.strip()
    sentences = sent_tokenize(text)

    if len(sentences) != 1:
        return False  # More than one sentence, not suitable

    sentence = sentences[0]
    words = word_tokenize(sentence)

    # Return True only if ends with punctuation and has >1 word
    return sentence.endswith(('.', '!', '?')) and len([w for w in words if w.isalnum()]) > 1


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
    CHUNK_MS = 10  # 20 milliseconds
    SAMPLE_RATE = 8000  # 8kHz
    BYTES_PER_SAMPLE = 1  # µ-law = 8 bits = 1 byte
    communicate = edge_tts.Communicate(text=ai_reply, voice="en-US-GuyNeural")
    audio_chunks = []

    # Collect TTS MP3 data
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    
    mp3_data = b"".join(audio_chunks)

    # Convert MP3 to 8kHz mono PCM (16-bit)
    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3") \
                       .set_frame_rate(SAMPLE_RATE) \
                       .set_channels(1) \
                       .set_sample_width(2)  # 16-bit linear PCM

    raw_pcm = audio.raw_data

    # Convert to µ-law (1 byte per sample)
    mulaw_pcm = audioop.lin2ulaw(raw_pcm, 2)  # 2 bytes = 16-bit input

    # Calculate chunk size in bytes for 20 ms at 8kHz and 1 byte/sample
    bytes_per_chunk = int(SAMPLE_RATE * CHUNK_MS / 1000 * BYTES_PER_SAMPLE)

    # Send in chunks
    for i in range(0, len(mulaw_pcm), bytes_per_chunk):
        chunk = mulaw_pcm[i:i + bytes_per_chunk]
        payload = base64.b64encode(chunk).decode("ascii")

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
Limit your response to 1–2 short sentences only. Do not make sentences longer.

If you dont know anything, respond im unsure about that.
"""


async def call_ollama_chat_llama3(messages: list):
    msg = [{
        'role': 'system',
        'content': system_prompt
    }]
    
    msg.extend(messages)

    # Enable streaming
    response_stream = await ollama.AsyncClient().chat(
        model='llama3:8b',
        messages=msg,
        stream=True,
        options={"temperature": 0.1}
    )

    # Stream each chunk
    async for chunk in response_stream:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']


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

    
    speech_pause_seconds_threshold = 3
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
