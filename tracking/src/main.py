from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from src.blink_detection.blink_detection import BlinkCounterVideo
from src.gaze_detection.gaze_detection import GazeDirectionVideo
from src.hand_detection.hand_detection import FaceTouchDetectorVideo
from src.head_detection.head_detection import HeadPoseVideo
from src.blink_detection.FaceMeshModule import FaceMeshGenerator
import tempfile
import shutil
import os

app = FastAPI()

def measure_center(cap, model, predictor_func, calibrate_func, label=""):
    total_pitch = total_yaw = total_roll = 0
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * 3)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = predictor_func(landmarks)
            total_pitch += pitch
            total_yaw += yaw
            total_roll += roll
            count += 1

    face_mesh.close()

    if count > 0:
        calibrate_func(
            total_pitch / count,
            total_yaw / count,
            total_roll / count
        )


def run_all_analyses(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "영상 열기 실패"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    generator = FaceMeshGenerator()
    blink = BlinkCounterVideo()
    gaze = GazeDirectionVideo()
    face_touch = FaceTouchDetectorVideo()
    head_pose = HeadPoseVideo()
    face_touch.set_fps(fps)

    measure_center(cap, head_pose, head_pose._predict_pose, head_pose.calibrate_center, label="Head Pose")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    measure_center(cap, gaze, gaze.predict_head_pose, gaze.calibrate_center, label="Gaze")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t_sec = frame_idx / fps  # ← 현재 영상 진행 초
        frame_idx += 1

        frame, landmarks_xy = generator.create_face_mesh(frame, draw=False)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        if not face_results.multi_face_landmarks:
            continue

        landmarks_obj = face_results.multi_face_landmarks[0].landmark

        hand_results = hands.process(rgb)
        hand_lms = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        blink.process(landmarks_xy, t_sec)
        gaze.process(landmarks_obj, w, h, t_sec)
        face_touch.process(landmarks_xy, hand_lms, w, h, t_sec)
        head_pose.process(landmarks_obj, t_sec)

    cap.release()
    # 리소스 정리
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()

    # 모듈별 결과 수집
    blink_res = blink.get_result()
    gaze_res  = gaze.get_result()
    head_res  = head_pose.get_result()
    hand_res  = face_touch.get_result()

    # 텍스트 요약
    def summarize(name, r):
        if r["reasons"]:
            return f"{name}: {', '.join(r['reasons'])}로 인해 감점 {r['penalty']}점, 점수는 {r['score']}점입니다!"
        return f"{name}: 감점 없이 만점입니다! 점수는 {r['score']}점입니다!"

    text_summary = "\n".join([
        summarize("눈 깜빡임 감지 분석 결과", blink_res),
        summarize("시선처리 감지 분석 결과",   gaze_res),
        summarize("고개 각도 감지 분석 결과", head_res),
        summarize("손 움직임 감지 분석 결과", hand_res),
    ])

    # 타임스탬프 병합
    timestamps = []
    timestamps.extend(blink_res.get("events", []))
    timestamps.extend(gaze_res.get("events", []))
    timestamps.extend(head_res.get("events", []))
    timestamps.extend(hand_res.get("events", []))

    # 정렬(시간순)
    def key_ts(e):  # "MM:SS" → 초로 변환
        mm, ss = e["time"].split(":")
        return int(mm) * 60 + int(ss)
    timestamps.sort(key=key_ts)

    return {
        "text": text_summary.strip(),
        "blinkScore": blink_res["score"],
        "eyeScore":   gaze_res["score"],
        "headScore":  head_res["score"],
        "handScore":  hand_res["score"],
        "timestamp":  timestamps
    }, None


@app.post("/tracking")
async def analyze_tracking(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        # 1. 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # 2. 분석 실행
        result, error = run_all_analyses(temp_path)
        os.remove(temp_path)

        if error:
            return JSONResponse(content={"error": error}, status_code=400)

        # 3. 응답 형식 통일
        return {
            "interviewId": interviewId,
            "seq": seq,
            "text": result["text"],
            "blinkScore": result["blinkScore"],
            "eyeScore": result["eyeScore"],
            "headScore": result["headScore"],
            "handScore": result["handScore"],
            "timestamp": result["timestamp"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})