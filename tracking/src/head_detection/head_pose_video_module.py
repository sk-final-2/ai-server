import cv2
import numpy as np
import mediapipe as mp
import pickle
import os

# === 모델 및 Mediapipe 설정 ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# === 허용 오차 ===
PITCH_TOLERANCE = 0.03
YAW_TOLERANCE = 0.1
ROLL_TOLERANCE = 0.1

# === 학습 시 사용한 랜드마크 인덱스 ===
landmark_indices = [1, 33, 61, 199, 263, 291, 362]

def predict_pose(landmarks):
    """선택된 랜드마크로 pitch/yaw/roll 예측"""
    coords = []
    for idx in landmark_indices:
        lm = landmarks[idx]
        coords.extend([lm.x, lm.y])
    coords = np.array(coords).reshape(1, -1)
    return model.predict(coords)[0]  # [pitch, yaw, roll]

def measure_center_values(cap, duration_sec=3):
    """영상 처음 duration_sec 동안 pitch/yaw/roll 평균 측정"""
    total_pitch, total_yaw, total_roll = 0, 0, 0
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * duration_sec)

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pitch, yaw, roll = predict_pose(face_landmarks.landmark)
                total_pitch += pitch
                total_yaw += yaw
                total_roll += roll
                count += 1

    if count > 0:
        return (total_pitch / count, total_yaw / count, total_roll / count)
    else:
        return (0.0, 0.0, 0.0)  # 얼굴 인식 실패 시

def analyze_video(video_path: str) -> dict:
    """영상 분석 → 정면 유지율 기반 점수 계산"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "영상 열기 실패"}

    # 1. 기준값 자동 측정
    pitch_center, yaw_center, roll_center = measure_center_values(cap, duration_sec=3)

    # 2. 다시 처음부터 읽기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_interval = 1 / 3  # 1초당 3프레임 분석
    prev_timestamp = -frame_interval
    total_frames = 0
    forward_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp - prev_timestamp >= frame_interval:
            prev_timestamp = timestamp
            total_frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pitch, yaw, roll = predict_pose(face_landmarks.landmark)

                    pitch_diff = pitch - pitch_center
                    yaw_diff = yaw - yaw_center
                    roll_diff = roll - roll_center

                    if (abs(pitch_diff) < PITCH_TOLERANCE and
                        abs(yaw_diff) < YAW_TOLERANCE and
                        abs(roll_diff) < ROLL_TOLERANCE):
                        forward_frames += 1

    cap.release()

    # 점수 계산
    score = round((forward_frames / total_frames) * 100, 2) if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "forward_frames": forward_frames,
        "score": score,
        "pitch_center": round(pitch_center, 3),
        "yaw_center": round(yaw_center, 3),
        "roll_center": round(roll_center, 3)
    }
