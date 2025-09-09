from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
import cv2
import numpy as np
from collections import Counter, defaultdict
import shutil
import os
import uuid

app = FastAPI()

# (참고) DeepFace 감정 라벨: angry, disgust, fear, happy, sad, surprise, neutral

# 라벨/순위별 감점 가중치 (원하는 대로 조절)
# 예: angry/disgust/fear는 강하게, sad는 중간, surprise는 약하게, neutral/happy는 0
PENALTY_RULES = {
    "angry":   {"top1": 1.0, "top2": 0.0},
    "disgust": {"top1": 1.0, "top2": 0.0},
    "fear":    {"top1": 1.0, "top2": 0.0},
    "sad":     {"top1": 1.0, "top2": 0.0},
    "surprise":{"top1": 1.0, "top2": 0.0},
    "neutral": {"top1": 0.0, "top2": 0.0},
    "happy":   {"top1": 0.0, "top2": 0.0},
}
_DEFAULT_RULE = {"top1": 0.0, "top2": 0.0}

emotion_kor_map = {
    "sad": "슬픔", "angry": "화남", "surprise": "놀람",
    "disgust": "혐오", "fear": "두려움", "neutral": "무표정", "happy": "행복"
}

def _kor(label: str) -> str:
    return emotion_kor_map.get(label, label)

def _rank_weight(label: str, rank: str) -> float:
    """rank는 'top1' 또는 'top2'"""
    return PENALTY_RULES.get(label, _DEFAULT_RULE).get(rank, 0.0)

def sec_to_hhmmss(sec: int) -> str:
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze")
async def analyze_emotion(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...),
):
    # 1) 파일 저장
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) 영상 열기
    cap = cv2.VideoCapture(temp_filename)
    if not cap.isOpened():
        os.remove(temp_filename)
        return JSONResponse(content={"error": "영상 열기 실패"}, status_code=400)

    # 총 길이(초) 계산 시도
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    # 3) 분석(초당 3프레임 샘플링)
    frame_interval = 1 / 3  # 1초당 3프레임
    per_second_pairs = defaultdict(list)  # {sec: [(top1, top2), ...]}
    prev_timestamp = -frame_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp - prev_timestamp >= frame_interval:
            prev_timestamp = timestamp
            second = int(timestamp)

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="mtcnn",  # 필요 시 retinaface로 교체 가능
                    enforce_detection=False
                )
                emotions = result[0]["emotion"]
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                top1, top2 = sorted_emotions[0][0], sorted_emotions[1][0]
                per_second_pairs[second].append((top1, top2))
            except Exception:
                continue

    # 영상 리소스 정리
    cap.release()
    os.remove(temp_filename)

    # fps 정보를 못 얻었을 때 총 길이 추정 (마지막 초 + 1)
    if fps > 0 and frame_count > 0:
        duration_seconds = max(1, int(round(frame_count / fps)))
    else:
        duration_seconds = max(per_second_pairs.keys(), default=0) + 1
        duration_seconds = max(1, duration_seconds)

    # 4) 초당 기본 감점 단위
    per_second_unit = 100.0 / duration_seconds

    # 5) 초별 대표 Top-1/Top-2 선정 → 가중치 합으로 감점
    emotion_top1_seconds = Counter()  # 설명용: 초 단위 top1 카운트
    total_deduction = 0.0
    penalty_timestamps = []

    for second, pairs in per_second_pairs.items():
        if not pairs:
            continue

        # 각 초의 top1/top2 최빈값(대표 감정) 산출
        c1 = Counter([p[0] for p in pairs]).most_common()
        c2 = Counter([p[1] for p in pairs]).most_common()

        top1_rep = c1[0][0] if c1 else None
        top2_rep = None
        if c2:
            # top2가 top1과 같으면 다음 후보를 찾아 중복 감점 방지
            for lab, _ in c2:
                if lab != top1_rep:
                    top2_rep = lab
                    break

        # 초별 가중치 합
        w1 = _rank_weight(top1_rep, "top1") if top1_rep else 0.0
        w2 = _rank_weight(top2_rep, "top2") if top2_rep else 0.0
        sec_weight = w1 + w2

        # 감점 누적
        deduction = per_second_unit * sec_weight
        total_deduction += deduction

        # 설명/통계
        if top1_rep:
            emotion_top1_seconds[top1_rep] += 1
        if deduction > 0:
            reason = f"표정 감지: {_kor(top1_rep)}" if top1_rep else "표정 감지"
            if w2 > 0 and top2_rep:
                reason += f"/{_kor(top2_rep)}"
            penalty_timestamps.append({
                "time": sec_to_hhmmss(second),
                "reason": reason
            })

    final_score = max(0.0, 100.0 - total_deduction)
    final_score = round(final_score, 2)

    # 6) 설명 메시지 (Top-1 기준 초수 집계)
    parts = [f"{_kor(emo)} {cnt}초" for emo, cnt in emotion_top1_seconds.items()]
    if parts:
        message = (
            f"표정 감지 분석 결과: {', '.join(parts)}로 인해 "
            f"총 감점 {round(total_deduction, 2)}점, 최종 점수는 {final_score}점입니다."
        )
    else:
        message = f"표정 감지 분석 결과: 감정 인식 실패, 최종 점수는 {final_score}점입니다."

    return {
        "interviewId": interviewId,
        "seq": seq,
        "score": final_score,
        "text": message,
        "timestamp": penalty_timestamps,
        "meta": {
            "durationSeconds": duration_seconds,
            "perSecondUnit": round(per_second_unit, 4),
            "rules": PENALTY_RULES
        }
    }
