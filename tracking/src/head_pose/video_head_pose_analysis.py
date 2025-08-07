import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import Counter

# === 모델 경로 설정 ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# 모델 로드
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# 허용 오차
PITCH_TOLERANCE = 0.01
YAW_TOLERANCE = 0.02
ROLL_TOLERANCE = 0.02

# 학습 시 사용한 랜드마크 인덱스
landmark_indices = [1, 33, 61, 199, 263, 291, 362]

def predict_pose(landmarks):
    coords = []
    for idx in landmark_indices:
        lm = landmarks[idx]
        coords.extend([lm.x, lm.y])
    coords = np.array(coords).reshape(1, -1)
    return model.predict(coords)[0]  # [pitch, yaw, roll]

def measure_center_from_video(cap, duration_sec=3):
    """영상 처음 duration_sec 동안 평균값 계산"""
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
        return (0.0, 0.0, 0.0)

# 감점 이유 영어 → 한국어 매핑
REASON_TRANSLATIONS = {
    "Looking Up": "고개 숙임",
    "Head Down": "위쪽 응시",
    "Looking Right": "왼쪽 응시",
    "Looking Left": "오른쪽 응시",
    "Head Tilt Right": "왼쪽 기울임",
    "Head Tilt Left": "오른쪽 기울임",
    "Not Facing Forward": "정면 응시 아님"
}

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 영상 열기 실패")
        return

    # 1. 기준값 측정
    pitch_center, yaw_center, roll_center = measure_center_from_video(cap, duration_sec=3)
    print(f"[INFO] Center: Pitch={pitch_center:.3f}, Yaw={yaw_center:.3f}, Roll={roll_center:.3f}")

    # 2. 다시 처음부터 분석
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = 0
    forward_frames = 0
    penalty_count = 0        # 감점 횟수 (10점씩 차감)
    penalty_reasons_list = []
    prev_status = "Facing Forward"
    prev_reason = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                    status_text = "Facing Forward"
                    forward_frames += 1
                    color = (0, 255, 0)
                    prev_reason = None
                else:
                    # 감점 사유 판별
                    if yaw_diff > YAW_TOLERANCE:
                        reason = "Looking Right"
                    elif yaw_diff < -YAW_TOLERANCE:
                        reason = "Looking Left"
                    elif pitch_diff > PITCH_TOLERANCE:
                        reason = "Looking Up"
                    elif pitch_diff < -PITCH_TOLERANCE:
                        reason = "Head Down"
                    elif roll_diff > ROLL_TOLERANCE:
                        reason = "Head Tilt Right"
                    elif roll_diff < -ROLL_TOLERANCE:
                        reason = "Head Tilt Left"
                    else:
                        reason = "Not Facing Forward"

                    status_text = "Not Facing Forward"
                    color = (0, 0, 255)

                    # 정면 → 위반 변경 시, 그리고 이전 위반 사유와 다를 때만 기록
                    if prev_status == "Facing Forward" or reason != prev_reason:
                        penalty_reasons_list.append(reason)
                        penalty_count += 1
                        prev_reason = reason

                prev_status = status_text

                # 영상 디스플레이용 텍스트
                cv2.putText(frame, f"Pitch: {pitch:.2f} | Yaw: {yaw:.2f} | Roll: {roll:.2f}",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, status_text, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        total_frames += 1
        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 최종 점수 계산 (최소 0점)
    head_pose_score = max(0, 100 - (penalty_count * 5))

    # 사유별 카운트
    reason_counts = Counter(penalty_reasons_list)
    reason_text_parts = [
        f"{REASON_TRANSLATIONS.get(reason, reason)} {count}회"
        for reason, count in reason_counts.items()
    ]
    reasons_kor_text = ", ".join(reason_text_parts) if reason_counts else "위반 없음"

    # 최종 사유 문장
    if reason_counts:
        head_pose_text = f"{reasons_kor_text}의 이유로 {100 - head_pose_score}점 감점되었습니다."
    else:
        head_pose_text = "위반한 점이 없습니다."

    print(f"[RESULT] Score: {head_pose_score}, Text: {head_pose_text}")
    return {
        "head_pose_score": head_pose_score,
        "head_pose_text": head_pose_text
    }

if __name__ == "__main__":
    video_path = "test.mp4"
    analyze_video(video_path)
