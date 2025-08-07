import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import Counter

class HeadPoseVideo:
    REASON_TRANSLATIONS = {
        "Looking Up": "고개 숙임",
        "Head Down": "위쪽 응시",
        "Looking Right": "왼쪽 응시",
        "Looking Left": "오른쪽 응시",
        "Head Tilt Right": "왼쪽 기울임",
        "Head Tilt Left": "오른쪽 기울임",
        "Not Facing Forward": "정면 응시 아님"
    }

    LANDMARK_IDX = [1, 33, 61, 199, 263, 291, 362]

    def __init__(self, video_path, pitch_tol=0.01, yaw_tol=0.02, roll_tol=0.02):
        self.video_path = video_path
        self.PITCH_TOL = pitch_tol
        self.YAW_TOL = yaw_tol
        self.ROLL_TOL = roll_tol
        self.penalty_reasons_list = []

        # 모델 로드
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Mediapipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def _predict_pose(self, landmarks):
        coords = []
        for idx in self.LANDMARK_IDX:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y])
        coords = np.array(coords).reshape(1, -1)
        return self.model.predict(coords)[0]  # pitch, yaw, roll

    def _measure_center(self, cap, duration_sec=3):
        total_pitch = total_yaw = total_roll = 0
        count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        max_frames = int(fps * duration_sec)

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    pitch, yaw, roll = self._predict_pose(lm.landmark)
                    total_pitch += pitch
                    total_yaw += yaw
                    total_roll += roll
                    count += 1

        return (total_pitch / count, total_yaw / count, total_roll / count) if count > 0 else (0, 0, 0)

    def _process(self, cap):
        pitch_center, yaw_center, roll_center = self._measure_center(cap, duration_sec=3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_status = "Facing Forward"
        prev_reason = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks:
                    pitch, yaw, roll = self._predict_pose(lm.landmark)
                    pitch_diff = pitch - pitch_center
                    yaw_diff = yaw - yaw_center
                    roll_diff = roll - roll_center

                    if (abs(pitch_diff) < self.PITCH_TOL and
                        abs(yaw_diff) < self.YAW_TOL and
                        abs(roll_diff) < self.ROLL_TOL):
                        status_text = "Facing Forward"
                        prev_reason = None
                        color = (0, 255, 0)
                    else:
                        if yaw_diff > self.YAW_TOL:
                            reason = "Looking Right"
                        elif yaw_diff < -self.YAW_TOL:
                            reason = "Looking Left"
                        elif pitch_diff > self.PITCH_TOL:
                            reason = "Looking Up"
                        elif pitch_diff < -self.PITCH_TOL:
                            reason = "Head Down"
                        elif roll_diff > self.ROLL_TOL:
                            reason = "Head Tilt Right"
                        elif roll_diff < -self.ROLL_TOL:
                            reason = "Head Tilt Left"
                        else:
                            reason = "Not Facing Forward"

                        status_text = "Not Facing Forward"
                        color = (0, 0, 255)

                        if prev_status == "Facing Forward" or reason != prev_reason:
                            self.penalty_reasons_list.append(reason)
                            prev_reason = reason

                    prev_status = status_text

                    cv2.putText(frame, f"Pitch: {pitch:.2f} | Yaw: {yaw:.2f} | Roll: {roll:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Head Pose Analysis", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def _calculate_result(self):
        penalty = len(self.penalty_reasons_list) * 5
        score = max(0, 100 - penalty)
        reason_counts = Counter(self.penalty_reasons_list)
        reason_text = ", ".join([f"{self.REASON_TRANSLATIONS.get(r, r)} {c}회" for r, c in reason_counts.items()])
        return score, penalty, reason_text or "위반 없음"

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("[ERROR] 영상 열기 실패")
            return
        self._process(cap)
        score, penalty, reasons_kor_text = self._calculate_result()
        if penalty > 0:
            print(f"고개 각도 감지 분석 결과: {reasons_kor_text}로 인해 {penalty}점 감점, 점수는 {score}점입니다!")
        else:
            print("고개 각도 감지 분석 결과: 위반한 점이 없습니다. 점수는 100점입니다!")

    def run_and_get_result(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0, 0, []
        self._process(cap)
        score, penalty, _ = self._calculate_result()
        return score, penalty, self.penalty_reasons_list

if __name__ == "__main__":
    video_path = "test.mp4"
    analyzer = HeadPoseVideo(video_path)
    analyzer.run()
