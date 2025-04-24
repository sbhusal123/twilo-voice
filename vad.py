import webrtcvad

# Initialize WebRTC VAD with aggressiveness level 3 (most aggressive)
vad = webrtcvad.Vad(3)

# Constants
SAMPLE_RATE = 8000  # 8kHz is common for mu-law audio
FRAME_DURATION_MS = 30  # 30ms frame duration
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Number of samples per frame

# Function to convert mu-law to PCM
def mu_law_to_pcm(mu_law_bytes):
    pcm_bytes = audioop.ulaw2lin(mu_law_bytes, 2)
    return pcm_bytes

# Function to detect speech and process audio chunks
def detect_speech_in_encoded_audio(audio_chunk):
    # Decode the mu-law audio chunk into PCM (16-bit linear)
    pcm_audio = mu_law_to_pcm(audio_chunk)

    # Process the PCM audio in 30ms frames and detect speech
    num_frames = len(pcm_audio) // FRAME_SIZE
    frames = [pcm_audio[i * FRAME_SIZE:(i + 1) * FRAME_SIZE] for i in range(num_frames)]

    for frame in frames:
        # Check if speech is detected in the frame
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        if is_speech:
            print("Speech detected!")
        else:
            print("Silence detected.")

from pydub import AudioSegment
import audioop
import base64

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
    return base64.b64encode(mulaw_pcm)


dd = mp3_to_mulaw_b64('sample.mp3')

# Detect speech in the encoded audio chunk
detect_speech_in_encoded_audio(dd)
