import json
import base64
import audioop
import webrtcvad

import base64
import audioop
import numpy as np
import librosa
import whisper
import numpy as np
import noisereduce as nr
import scipy.signal

from pydub import AudioSegment
from io import BytesIO


import ollama
import edge_tts
import asyncio
import base64

# Setup
sample_rate = 16000
bytes_per_sample = 2  # 16-bit PCM
vad = webrtcvad.Vad(3)
audio_buffer = bytearray()

def reset_audio_buffer():
    global audio_buffer
    audio_buffer.clear()

whisper_model = whisper.load_model("base")  # or "tiny", "small", etc.

def transcribe_mulaw_audio(mulaw_bytes: bytes, input_sample_rate: int = 8000, target_sample_rate: int = 16000) -> str:
    """
    Decodes µ-law bytes, resamples to 16 kHz, and transcribes using Whisper.
    
    Args:
        mulaw_bytes (bytes): µ-law encoded audio (bytearray or bytes)
        input_sample_rate (int): Original sample rate (Twilio sends 8000 Hz)
        target_sample_rate (int): Target rate for Whisper (default: 16000 Hz)
    
    Returns:
        str: Transcribed text
    """
    # Decode µ-law to 16-bit PCM (2 bytes per sample)
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)

    # Convert to numpy int16 array
    pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Convert to float32 for resampling
    float_audio = pcm_np.astype(np.float32) / 32768.0

    # Resample to 16kHz
    resampled = librosa.resample(float_audio, orig_sr=input_sample_rate, target_sr=target_sample_rate)

    # Transcribe with Whisper
    result = whisper_model.transcribe(resampled, language="en")

    return result['text']

def has_speech(pcm_data, sample_rate=16000, frame_duration_ms=30):
    frame_len = int(sample_rate * frame_duration_ms / 1000) * 2
    frames = [
        pcm_data[i:i + frame_len]
        for i in range(0, len(pcm_data), frame_len)
        if len(pcm_data[i:i + frame_len]) == frame_len
    ]
    speech_frames = sum(vad.is_speech(f, sample_rate) for f in frames)
    return speech_frames > 0  # at least 1 speech frame

def remove_noise_from_pcm(pcm_data: bytes, sample_rate: int = 16000, reduce_stationary=True) -> bytes:
    """
    Removes background noise from 16-bit PCM audio before VAD/STT.

    Enhancements:
    - Handles empty or short audio safely
    - Applies high-pass filter to remove low-frequency rumble
    - Optionally applies noise reduction for stationary + transient noise

    Args:
        pcm_data (bytes): Raw PCM audio
        sample_rate (int): Input sample rate (default 16kHz)
        reduce_stationary (bool): Whether to reduce stationary noise too

    Returns:
        bytes: Denoised PCM audio
    """
    if not pcm_data:
        return pcm_data  # Empty input guard

    # Convert to float32 NumPy array
    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)

    # Normalize to [-1.0, 1.0]
    audio_np /= 32768.0

    # --- High-pass filter to remove low-frequency hum ---
    sos = scipy.signal.butter(10, 80, 'hp', fs=sample_rate, output='sos')  # 80 Hz
    audio_filtered = scipy.signal.sosfilt(sos, audio_np)

    # --- Noise reduction ---
    reduced_audio = nr.reduce_noise(
        y=audio_filtered,
        sr=sample_rate,
        stationary=reduce_stationary,
        prop_decrease=1.0  # Full reduction
    )

    # Re-normalize and convert back to PCM int16
    reduced_int16 = np.clip(reduced_audio * 32768.0, -32768, 32767).astype(np.int16)

    return reduced_int16.tobytes()

async def send_audio_to_websocket(websocket, audio_bytes: bytes, chunk_size: int = 3200):
    """Split and send audio as base64-encoded WebSocket media messages."""
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        message = {
            "event": "media",
            "media": {
                "payload": base64.b64encode(chunk).decode("utf-8")
            }
        }
        await websocket.send_text(json.dumps(message))
        await asyncio.sleep(0.1)  # simulate real-time pacing

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

async def get_transcribed_text_from_stream(data):
    # Decode µ-law and buffer audio
    pcm_bytes = base64.b64decode(data["media"]["payload"])
    audio_buffer.extend(pcm_bytes)

    # Configuration
    duration_check_s = 5  # seconds for VAD
    whisper_window_s = 3  # seconds for Whisper
    chunk_size = sample_rate * bytes_per_sample

    # Enough audio buffered?
    if len(audio_buffer) < chunk_size * whisper_window_s:
        return

    # Extract latest VAD window
    latest_window = audio_buffer[-chunk_size * duration_check_s:]
    denoised_window = remove_noise_from_pcm(latest_window)

    if has_speech(denoised_window):
        print("✅ Speech detected. Transcribing...")

        # Prepare last 5s of audio for transcription
        to_transcribe = audio_buffer[-chunk_size * whisper_window_s:]
        audio_buffer.clear()  # Clear buffer after use

        text = transcribe_mulaw_audio(to_transcribe)
        print(f"📝 Transcript: {text}")

        return text
            
    else:
        return ""
