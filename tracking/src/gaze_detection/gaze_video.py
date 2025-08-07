import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import Counter
import pickle
import os

class GazeDirectionVideo:
    # 눈 좌우 끝과 홍채 중심 인덱스
    LEFT_EYE_LANDMARKS = [33, 133]
    RIGHT_EYE_LANDMARKS = [362, 263]
    LEFT_IRIS = [468]
    RIGHT_IRIS = [473]

    # head_pose 학습 랜드마크
    HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 263, 291, 362]

    # 최종 출력용 번역
    REASON_TRANSLATIONS = {
        "LOOK_LEFT": "오른쪽 응시",
        "LOOK_RIGHT": "왼쪽 응시",
        "LOOK_UP": "아래쪽 응시",
        "LOOK_DOWN": "위쪽 응시"
    }

    def __init__(self, video_path, penalty_per_violation=10, stable_frames_required=5):
        self.video_path = video_path
        self.penalty_per_violation = penalty_per_violation
        self.stable_frames_required = stable_frames_required
        self.violations = []

        # Mediapipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # head_pose 모델 로드
        BASE_DIR = os.path.dirname(__file__)
        MODEL_PATH = os.path.join(BASE_DIR, "..", "head_pose", "model.pkl")
        MODEL_PATH = os.path.abspath(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            self.head_pose_model = pickle.load(f)

        # head_pose 허용 오차
        self.PITCH_TOL = 0.05
        self.YAW_TOL = 0.08
        self.ROLL_TOL = 0.08

        # 캘리브레이션 기본값
        self.PITCH_CENTER = 0.0
        self.YAW_CENTER = 0.0
        self.ROLL_CENTER = 0.0

    # ===== Head Pose =====
    def predict_head_pose(self, landmarks):
        coords = []
        for idx in self.HEAD_POSE_LANDMARKS:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y])
        coords = np.array(coords).reshape(1, -1)
        return self.head_pose_model.predict(coords)[0]  # [pitch, yaw, roll]

    def is_head_forward(self, pitch, yaw, roll):
        return (
            abs(pitch - self.PITCH_CENTER) < self.PITCH_TOL and
            abs(yaw - self.YAW_CENTER) < self.YAW_TOL and
            abs(roll - self.ROLL_CENTER) < self.ROLL_TOL
        )

    def calibrate_center_from_video(self, cap, duration_sec=3):
        """3초 동안 고개 중심값 캘리브레이션"""
        print(f"[INFO] Calibrating head pose for {duration_sec} seconds... Please face forward.")
        total_pitch, total_yaw, total_roll = 0, 0, 0
        count = 0
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        max_frames = int(fps * duration_sec)

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    pitch, yaw, roll = self.predict_head_pose(lm.landmark)
                    total_pitch += pitch
                    total_yaw += yaw
                    total_roll += roll
                    count += 1
            cv.putText(frame, "Calibrating...", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.imshow("Gaze Calibration", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

        if count > 0:
            self.PITCH_CENTER = total_pitch / count
            self.YAW_CENTER = total_yaw / count
            self.ROLL_CENTER = total_roll / count
        else:
            print("[WARN] Calibration failed. Using default values.")

        print(f"[INFO] Center calibrated: Pitch={self.PITCH_CENTER:.3f}, "
              f"Yaw={self.YAW_CENTER:.3f}, Roll={self.ROLL_CENTER:.3f}")

    # ===== Gaze Direction =====
    def get_gaze_direction(self, landmarks, w, h):
        left_iris = np.array([landmarks[self.LEFT_IRIS[0]].x * w,
                              landmarks[self.LEFT_IRIS[0]].y * h])
        left_eye_left = np.array([landmarks[self.LEFT_EYE_LANDMARKS[0]].x * w,
                                  landmarks[self.LEFT_EYE_LANDMARKS[0]].y * h])
        left_eye_right = np.array([landmarks[self.LEFT_EYE_LANDMARKS[1]].x * w,
                                   landmarks[self.LEFT_EYE_LANDMARKS[1]].y * h])

        right_iris = np.array([landmarks[self.RIGHT_IRIS[0]].x * w,
                               landmarks[self.RIGHT_IRIS[0]].y * h])
        right_eye_left = np.array([landmarks[self.RIGHT_EYE_LANDMARKS[0]].x * w,
                                   landmarks[self.RIGHT_EYE_LANDMARKS[0]].y * h])
        right_eye_right = np.array([landmarks[self.RIGHT_EYE_LANDMARKS[1]].x * w,
                                    landmarks[self.RIGHT_EYE_LANDMARKS[1]].y * h])

        left_ratio = (np.linalg.norm(left_iris - left_eye_left) /
                      np.linalg.norm(left_eye_right - left_eye_left))
        right_ratio = (np.linalg.norm(right_iris - right_eye_left) /
                       np.linalg.norm(right_eye_right - right_eye_left))

        avg_ratio = (left_ratio + right_ratio) / 2

        iris_y_avg = (left_iris[1] + right_iris[1]) / 2
        eye_y_center = (left_eye_left[1] + left_eye_right[1] +
                        right_eye_left[1] + right_eye_right[1]) / 4

        if avg_ratio < 0.42:
            return "LOOK_LEFT"
        elif avg_ratio > 0.58:
            return "LOOK_RIGHT"
        elif iris_y_avg < eye_y_center - 3:
            return "LOOK_UP"
        elif iris_y_avg > eye_y_center + 3:
            return "LOOK_DOWN"
        else:
            return "FORWARD"

    # ===== Main Run =====
    def run(self):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {self.video_path}")
            return

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 1. 캘리브레이션
        self.calibrate_center_from_video(cap, duration_sec=3)

        # 2. 영상 처음부터 재시작
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        # FPS 샘플링 설정 (1초에 1프레임만 분석)
        fps = cap.get(cv.CAP_PROP_FPS) or 30
        frame_interval = int(fps / 1)
        frame_count = 0

        is_in_violation = False
        current_violation_type = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pitch, yaw, roll = self.predict_head_pose(face_landmarks.landmark)

                    if self.is_head_forward(pitch, yaw, roll):
                        gaze_dir = self.get_gaze_direction(face_landmarks.landmark, w, h)

                        # 새로운 위반 시작 순간
                        if gaze_dir != "FORWARD" and (not is_in_violation or gaze_dir != current_violation_type):
                            self.violations.append(gaze_dir)
                            is_in_violation = True
                            current_violation_type = gaze_dir

                        # 위반 종료 순간
                        if gaze_dir == "FORWARD":
                            is_in_violation = False
                            current_violation_type = None

                        # 화면 표시
                        cv.putText(frame, gaze_dir, (30, 60),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255) if gaze_dir != "FORWARD" else (0, 255, 0), 2)
                    else:
                        cv.putText(frame, "Head not forward", (30, 60),
                                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv.imshow("Gaze Direction Detection", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()

        # 결과 출력 (한글 변환)
        reason_counts = Counter(self.violations)
        reason_text_parts = [
            f"{count}초 동안의 {self.REASON_TRANSLATIONS.get(reason, reason)}"
            for reason, count in reason_counts.items()
        ]
        reasons_kor_text = ", ".join(reason_text_parts) if reason_counts else "위반 없음"

        total_penalty = len(self.violations) * self.penalty_per_violation
        final_score = max(0, 100 - total_penalty)

        print(f"[RESULT] Score: {final_score}, Text: {reasons_kor_text}로 인해 {total_penalty}점 감점되었습니다.")


if __name__ == "__main__":
    video_path = "testttttt.mp4"
    gaze_detector = GazeDirectionVideo(video_path, penalty_per_violation=10, stable_frames_required=5)
    gaze_detector.run()
