import os
import subprocess
from faster_whisper import WhisperModel

# 🔹 모델 로드 (CPU or GPU 선택)
# device="cuda" 사용 시 GPU 가속 가능 (CUDA 설치되어 있어야 함)
model = WhisperModel("base", device="cpu")  # 또는 "cuda"

def convert_to_wav(input_path: str, output_path: str):
    #로컬 실행용
    #ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"  # 실제 ffmpeg.exe 경로
    command = [
        #로컬 실행용
        #ffmpeg_path,
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        output_path,
        "-y"
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("❌ ffmpeg가 시스템 경로에 없습니다. PATH를 확인하거나 절대경로로 지정하세요.")

def transcribe_audio(wav_path: str) -> str:
    """
    Faster-Whisper로 STT 수행하여 텍스트 반환
    """
    segments, _ = model.transcribe(wav_path)

    # Segment 별 텍스트 합치기
    transcript = " ".join(segment.text for segment in segments)
    return transcript.strip()

def stt_from_path(input_path: str) -> str:
    """
    입력 파일 경로에서 텍스트 변환 결과 반환
    - 비-WAV는 wav로 변환 후 진행
    """
    ext = input_path.split('.')[-1].lower()
    output_path = input_path.replace(f".{ext}", ".wav")

    if ext != "wav":
        convert_to_wav(input_path, output_path)
    else:
        output_path = input_path

    return transcribe_audio(output_path)