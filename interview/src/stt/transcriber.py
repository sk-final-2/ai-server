import subprocess
import numpy as np
from faster_whisper import WhisperModel

# -----------------------------
# 1) Whisper 모델 로드 (CPU 기준)
# -----------------------------
model = WhisperModel(
    "small",            # 모델 크기 (tiny / base / small / medium / large-v2 가능)
    device="cpu",       # GPU 쓸 경우 "cuda"
    compute_type="int8" # CPU는 int8이 가장 효율적
)

# -----------------------------
# 2) 오디오 로더 (ffmpeg → numpy)
# -----------------------------
def load_audio_as_numpy(input_path: str, sr: int = 16000) -> np.ndarray:
    """
    오디오/영상(mp3, mp4, wav 등)을 ffmpeg로 디코딩해서
    float32 numpy array로 변환 (mono, 16kHz).
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-f", "f32le",      # raw float32 PCM
        "-ac", "1",         # mono
        "-ar", str(sr),     # 16kHz
        "pipe:1"            # stdout으로 출력
    ]

    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # 로그 숨김
        check=True
    )
    audio = np.frombuffer(proc.stdout, np.float32)
    return audio

# -----------------------------
# 3) STT 변환
# -----------------------------
def transcribe_audio(input_path: str):
    """
    입력 파일(mp3/mp4/wav 등)을 numpy array로 변환 후 STT 실행.
    """
    audio = load_audio_as_numpy(input_path)

    segments, _ = model.transcribe(
        audio,
        beam_size=5,
        best_of=5,
        patience=1,
        vad_filter=True,
        temperature=0.0
    )

    results = []
    for seg in segments:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })

    transcript = " ".join([r["text"] for r in results])
    return transcript, results

# -----------------------------
# 4) 외부 호출용 (통합 함수)
# -----------------------------
def stt_from_path(input_path: str):
    """
    일반 오디오/비디오(mp3, mp4, wav 등) → STT 수행 후 결과 반환
    """
    transcript, segments = transcribe_audio(input_path)
    return transcript, segments
