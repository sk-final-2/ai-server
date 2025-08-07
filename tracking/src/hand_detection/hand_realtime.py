import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import Counter

class FaceTouchDetectorLive:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    def __init__(self, penalty_per_violation=10, camera_index=0):
        self.penalty_per_violation = penalty_per_violation
        self.violations = []
        self.camera_index = camera_index

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_face_touch(self, face_landmarks, hand_landmarks, w, h):
        """
        얼굴 랜드마크와 손 랜드마크가 일정 거리 이하이면 True 반환
        """
        face_points = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark]
        )

        for hand in hand_landmarks:
            hand_points = np.array(
                [[lm.x * w, lm.y * h] for lm in hand.landmark]
            )

            # 손 포인트들과 얼굴 포인트 사이의 거리 계산
            for hp in hand_points:
                distances = np.linalg.norm(face_points - hp, axis=1)
                if np.any(distances < 40):  # 40px 이내면 얼굴 터치로 판단
                    return True
        return False

    def run(self):
        cap = cv.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[ERROR] 웹캠을 열 수 없습니다.")
            return

        is_in_violation = False
        total_frames = 0

        print("[INFO] 실시간 얼굴 터치 감지 시작 (ESC로 종료)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb)
            hand_results = self.hands.process(rgb)

            touched = False
            if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                if self.detect_face_touch(face_lm, hand_results.multi_hand_landmarks, w, h):
                    touched = True

            if touched:
                if not is_in_violation:
                    self.violations.append("FACE_TOUCH")
                    is_in_violation = True
                cv.putText(frame, "Face Touch Detected!", (30, 60),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                is_in_violation = False

            total_penalty = len(self.violations) * self.penalty_per_violation
            score = max(0, 100 - total_penalty)

            # 점수 표시
            cv.putText(frame, f"Score: {score}", (30, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv.imshow("Live Face Touch Detection", frame)
            key = cv.waitKey(1)
            if key == 27:  # ESC 누르면 종료
                break

            total_frames += 1

        cap.release()
        cv.destroyAllWindows()

        print(f"[RESULT] 총 {len(self.violations)}회 위반으로 {len(self.violations)*self.penalty_per_violation}점 감점되었습니다.")


if __name__ == "__main__":
    detector = FaceTouchDetectorLive(penalty_per_violation=10, camera_index=0)
    detector.run()
