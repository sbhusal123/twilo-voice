import ollama

import torchaudio
import torch
import numpy as np
import audioop

import edge_tts
from io import BytesIO
from pydub import AudioSegment
import base64
import json

from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration

from nltk.tokenize import sent_tokenize, word_tokenize


system_prompt = """
You are a highly knowledgeable AI assistant. Always answer clearly, confidently, and naturally,
but keep your response short and meaningful. Do not list points or use bullet forms.

Your answers will be converted to speech, so use proper punctuation like commas, periods, and question marks.
Avoid long sentences or excessive detail. Speak in a natural, conversational tone.
Limit your response to one to two short sentences only. Do not make sentences longer.

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


resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

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


    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    
    mp3_data = b"".join(audio_chunks)


    audio = AudioSegment.from_file(BytesIO(mp3_data), format="mp3") \
                       .set_frame_rate(SAMPLE_RATE) \
                       .set_channels(1) \
                       .set_sample_width(2)

    raw_pcm = audio.raw_data
    mulaw_pcm = audioop.lin2ulaw(raw_pcm, 2)
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


def is_complete_sentence(text):
    text = text.strip()
    sentences = sent_tokenize(text)

    if len(sentences) != 1:
        return False

    sentence = sentences[0]
    words = word_tokenize(sentence)

    # Return True only if ends with punctuation and has >1 word
    return sentence.endswith(('.', '!', '?')) and len([w for w in words if w.isalnum()]) > 1
