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

# 감정별 감점 기준
emotion_penalty_map = {
    "sad": 1,
    "angry": 2,
    "surprise": 2,
    "disgust": 2,
    "fear": 3,
    "neutral": 0,
    "happy": 0
}
# 영어 감정을 한글로 매핑
emotion_kor_map = {
    "sad": "슬픔",
    "angry": "화남",
    "surprise": "놀람",
    "disgust": "혐오",
    "fear": "두려움",
    "neutral": "무표정",
    "happy": "행복"
}

def sec_to_hhmmss(sec: int) -> str:
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

@app.post("/analyze")
async def analyze_emotion(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...),
):
    # 1. 파일 저장
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 영상 열기
    cap = cv2.VideoCapture(temp_filename)
    if not cap.isOpened():
        os.remove(temp_filename)
        return JSONResponse(content={"error": "영상 열기 실패"}, status_code=400)

    frame_interval = 1 / 3  # 1초당 3프레임
    per_second_emotions = defaultdict(list)
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
                    detector_backend="mtcnn",
                    enforce_detection=False
                )
                emotions = result[0]["emotion"]
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                top1, top2 = sorted_emotions[0][0], sorted_emotions[1][0]
                per_second_emotions[second].append((top1, top2))
            except Exception:
                continue

    cap.release()
    os.remove(temp_filename)

    # 3. 초당 감정 집계 → 1초에 대표 감정 1개만 반영
    emotion_counter = Counter()
    total_penalty = 0
    penalty_timestamps = []  # 감점 발생 초 기록

    for second, pairs in per_second_emotions.items():
        top1_list = [top1 for top1, _ in pairs]
        if not top1_list:
            continue

        most_common_emotion, _ = Counter(top1_list).most_common(1)[0]
        emotion_counter[most_common_emotion] += 1
        penalty = emotion_penalty_map.get(most_common_emotion, 0)
        total_penalty += penalty

        # 감점이 1점 이상인 경우에만 timestamp에 기록
        if penalty > 0:
            penalty_timestamps.append({
                "time": sec_to_hhmmss(second),
                "reason": "표정 감지"
            })

    # 4. 최종 점수 계산
    final_score = max(0, 100 - total_penalty)

    # 5. 설명 메시지 생성 (모든 감정 포함)
    all_emotion_parts = []
    for emo, count in emotion_counter.items():
        kor_name = emotion_kor_map.get(emo, emo)
        all_emotion_parts.append(f"{kor_name} {count}초")

    if all_emotion_parts:
        message = (
            f"표정 감지 분석 결과: {', '.join(all_emotion_parts)}로 인해 "
            f"감점 {total_penalty}점, 점수는 {round(final_score, 2)}점입니다!"
        )
    else:
        message = f"표정 감지 분석 결과: 감정 인식 실패, 점수는 {round(final_score, 2)}점입니다!"

    # 6. 결과 반환 (timestamp 추가)
    return {
        "interviewId": interviewId,
        "seq": seq,
        "score": round(final_score, 2),
        "text": message,
        "timestamp": penalty_timestamps
    }
