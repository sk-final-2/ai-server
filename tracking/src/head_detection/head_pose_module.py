import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time

# 현재 파일 기준 model.pkl 경로
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# 모델 로드
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 학습 시 사용한 랜드마크 인덱스
landmark_indices = [1, 33, 61, 199, 263, 291, 362]

# Mediapipe Face Mesh 초기화
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

# 랜드마크 → 예측
def predict_pose(landmarks):
    coords = []
    for idx in landmark_indices:
        lm = landmarks[idx]
        coords.extend([lm.x, lm.y])  # 학습 때 z값 안 썼음
    coords = np.array(coords).reshape(1, -1)
    return model.predict(coords)[0]  # [pitch, yaw, roll]

# === 자동 기준값 측정 ===
def measure_center_from_cam(duration_sec=3):
    print(f"[INFO] Measuring head pose for {duration_sec} seconds... Please face the camera.")
    total_pitch, total_yaw, total_roll = 0, 0, 0
    count = 0
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    while time.time() - start_time < duration_sec:
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

        cv2.putText(frame, "Measuring center... Face forward", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Head Pose Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        return (total_pitch / count, total_yaw / count, total_roll / count)
    else:
        return (0.0, 0.0, 0.0)

# === 메인 실행 ===
PITCH_CENTER, YAW_CENTER, ROLL_CENTER = measure_center_from_cam(3)
print(f"[INFO] Center values: Pitch={PITCH_CENTER:.3f}, Yaw={YAW_CENTER:.3f}, Roll={ROLL_CENTER:.3f}")

# 점수 계산용 카운터
total_frames = 0
forward_frames = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1  # 총 프레임 수 카운트

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pitch, yaw, roll = predict_pose(face_landmarks.landmark)

            # 디버그용 출력
            text = f"Pitch: {pitch:.2f} | Yaw: {yaw:.2f} | Roll: {roll:.2f}"
            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # 현재 상태 판정
            pitch_diff = pitch - PITCH_CENTER
            yaw_diff = yaw - YAW_CENTER
            roll_diff = roll - ROLL_CENTER

            if (abs(pitch_diff) < PITCH_TOLERANCE and
                abs(yaw_diff) < YAW_TOLERANCE and
                abs(roll_diff) < ROLL_TOLERANCE):
                status_text = "Facing Forward"
                color = (0, 255, 0)
                forward_frames += 1  # 정면 유지 카운트
            else:
                if pitch_diff > PITCH_TOLERANCE:
                    status_text = "Looking Up"
                elif pitch_diff < -PITCH_TOLERANCE:
                    status_text = "Head Down"
                elif yaw_diff > YAW_TOLERANCE:
                    status_text = "Looking Right"
                elif yaw_diff < -YAW_TOLERANCE:
                    status_text = "Looking Left"
                elif roll_diff > ROLL_TOLERANCE:
                    status_text = "Head Tilt Right"
                elif roll_diff < -ROLL_TOLERANCE:
                    status_text = "Head Tilt Left"
                else:
                    status_text = "Unknown Direction"
                color = (0, 0, 255)

            # 상태 표시
            cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            # 경고 메시지
            if status_text != "Facing Forward":
                cv2.putText(frame, "Please face the camera", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

    # 점수 계산 (정면 유지율)
    score = round((forward_frames / total_frames) * 100, 2) if total_frames > 0 else 0
    cv2.putText(frame, f"Score: {score}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Head Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()

print(f"[RESULT] Score: {score}% (Forward {forward_frames} / Total {total_frames})")
