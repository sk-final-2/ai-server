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

@app.post("/analyze")
async def analyze_emotion(
    file: UploadFile = File(...),
    interviewID: int = Form(...),
    seq: int = Form(...)
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
                print(f"[{second}초] 감정 분석 결과 → 1위: {top1}, 2위: {top2}")
            except Exception as e:
                continue

    cap.release()
    os.remove(temp_filename)

    # 3. 감정 분석 후 점수 계산
    penalty = 0
    for sec, top_pairs in per_second_emotions.items():
        flat_emotions = {emo for pair in top_pairs for emo in pair}
        if not ("neutral" in flat_emotions and "happy" in flat_emotions):
            penalty += 1


    max_seconds = len(per_second_emotions)
    final_score = max(0, 100 - (penalty / max_seconds) * 100)

    # 4. 결과 반환
    return {
        "interviewID": interviewID,
        "seq": seq,
        "score": round(final_score, 2)
    }
