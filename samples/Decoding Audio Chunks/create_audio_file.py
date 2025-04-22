import ffmpeg

def convert_ulaw_bytes_to_wav_file(audio_bytes: bytes, output_path="output.wav"):
    try:
        process = (
            ffmpeg
            .input('pipe:0', format='mulaw', ar='8000', ac='1')
            .output(output_path, format='wav')
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

        # Write raw audio bytes to stdin
        process.stdin.write(audio_bytes)
        process.stdin.close()

        # Wait for process to complete and capture stderr
        out, err = process.communicate()
        if process.returncode != 0:
            print("[FFmpeg Error]", err.decode())
            return None

        print(f"WAV saved to {output_path}")
        return output_path
    except Exception as ex:
        print(f"[Error] {ex}")
        return None


# Suppose you have a bytearray from Twilio or a file
with open("raw_audio.raw", "rb") as f:
    audio_buffer = bytearray(f.read())

# Now convert it to .wav
convert_ulaw_bytes_to_wav_file(audio_buffer, "converted.wav")

