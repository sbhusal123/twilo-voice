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

# Globals (or class-level if you're using a class)
speech_state = "silent"
silence_duration_ms = 0
silence_threshold_ms = 2000  # ms of silence to detect end of speech
speech_buffer = bytearray()
audio_buffer = bytearray()  # For VAD analysis only

# Configuration (adjust based on your audio stream)
sample_rate = 8000  # 8kHz for telephony
bytes_per_sample = 2  # 16-bit audio
duration_check_s = 0.5  # Duration of VAD chunk in seconds (200ms)

# Function
async def get_transcribed_text_from_stream(data):    
    global speech_state, silence_duration_ms, speech_buffer, audio_buffer

    # 1. Decode base64 µ-law audio payload
    pcm_bytes = base64.b64decode(data["media"]["payload"])

    # 2. Add to general buffer for VAD
    audio_buffer.extend(pcm_bytes)

    # 3. Calculate chunk size for VAD (e.g., 200ms chunk)
    chunk_size = int(sample_rate * bytes_per_sample * duration_check_s)

    # 4. Wait until there's enough audio to run VAD
    if len(audio_buffer) < chunk_size:
        print("🔴 Length of audoi bytes is not sufficient.")
        return

    # 5. Extract and denoise the latest VAD chunk
    latest_chunk = audio_buffer[-chunk_size:]
    denoised_chunk = remove_noise_from_pcm(latest_chunk)  # Your custom function

    # 6. Run VAD
    if has_speech(denoised_chunk):  # Your custom or WebRTC VAD-based function
        if speech_state == "silent":
            print("🟢 Start of speech detected.")
            speech_state = "speaking"
            speech_buffer = bytearray()  # Clear previous buffer
        silence_duration_ms = 0
        speech_buffer.extend(pcm_bytes)  # Append the current chunk
    else:
        if speech_state == "speaking":
            silence_duration_ms += duration_check_s * 1000
            if silence_duration_ms >= silence_threshold_ms:
                print("🔴 End of speech detected.")
                speech_state = "silent"
                silence_duration_ms = 0

                # Transcribe the complete speech segment
                to_transcribe = bytes(speech_buffer)
                speech_buffer.clear()

                text = transcribe_mulaw_audio(to_transcribe)  # Your custom function
                print(f"📝 Transcript: {text}")
                return text

    return ""
