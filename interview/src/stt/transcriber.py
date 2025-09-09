import subprocess
import numpy as np
from faster_whisper import WhisperModel

# ✅ CPU + medium 모델 (형 노트북 VRAM 6GB 기준)
model = WhisperModel(
    "small",            # 모델 크기 (tiny / base / small / medium / large-v2 가능)
    device="cpu",       # GPU 쓸 경우 "cuda"
    compute_type="int8" # CPU는 int8이 가장 효율적
)

# -----------------------------
# 오디오 로더 (ffmpeg → numpy)
# -----------------------------
def load_audio_as_numpy(input_path: str, sr: int = 16000) -> np.ndarray:
    command = [
        "ffmpeg",
        "-i", input_path,
        "-f", "f32le",  # raw float32 PCM
        "-ac", "1",     # mono
        "-ar", str(sr), # 16kHz
        "pipe:1"
    ]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return np.frombuffer(proc.stdout, np.float32)

# -----------------------------
# STT 실행 (final 모드 고정)
# -----------------------------
def transcribe_audio(input_path: str, language: str = None):
    audio = load_audio_as_numpy(input_path)

    # ✅ 항상 정확도 최우선 옵션
    beam_size, patience = 10, 2

    # 언어 매핑
    LANG_MAP = {
        "KOREAN": "ko",
        "ENGLISH": "en"
    }
    lang_opt = LANG_MAP.get(language, None)  # None이면 auto detect

    segments, _ = model.transcribe(
        audio,
        beam_size=beam_size,
        patience=patience,
        language=lang_opt,  # ✅ 여기서 ISO 코드 사용
        vad_filter=True,
        temperature=0.0
    )

    results = [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments]
    transcript = " ".join([r["text"] for r in results])
    return transcript, results
# -----------------------------
# 외부 호출용 (API는 그대로)
# -----------------------------
def stt_from_path(input_path: str, language: str = None):
    transcript, segments = transcribe_audio(input_path, language=language)
    return transcript, segments