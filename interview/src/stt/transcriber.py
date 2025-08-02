import ffmpeg
import whisper
import os

model = whisper.load_model("base")  # 또는 "medium", "large"

def convert_to_wav(input_path: str, output_path: str):
    ffmpeg.input(input_path).output(
        output_path,
        format='wav',
        acodec='pcm_s16le',
        ac=1,
        ar='16000'
    ).overwrite_output().run()

def transcribe_audio(wav_path: str):
    result = model.transcribe(wav_path)
    return result["text"]

def stt_from_path(input_path: str) -> str:
    ext = input_path.split('.')[-1].lower()
    output_path = input_path.replace(f".{ext}", ".wav")

    if ext != "wav":
        convert_to_wav(input_path, output_path)
    else:
        output_path = input_path

    return transcribe_audio(output_path)