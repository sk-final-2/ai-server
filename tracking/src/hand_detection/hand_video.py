import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import deque

class FaceTouchDetectorVideo:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    def __init__(self, video_path, penalty_per_violation=10):
        self.video_path = video_path
        self.penalty_per_violation = penalty_per_violation
        self.touch_frame_count = 0
        self.penalized_sections = 0
        self.penalty_sections = deque(maxlen=5)  # 최근 감점 시간 체크용

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
        face_points = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark]
        )

        for hand in hand_landmarks:
            hand_points = np.array(
                [[lm.x * w, lm.y * h] for lm in hand.landmark]
            )

            for hp in hand_points:
                distances = np.linalg.norm(face_points - hp, axis=1)
                if np.any(distances < 40):  # 40px 이내이면 터치로 간주
                    return True
        return False

    def run(self):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {self.video_path}")
            return

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS) or 30

        touch_frames = 0
        no_touch_frames = 0
        frame_threshold = int(fps * 0.5)  # 0.5초 이상 터치 시 감점 (약 15프레임)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb)
            hand_results = self.hands.process(rgb)

            if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                touched = self.detect_face_touch(face_lm, hand_results.multi_hand_landmarks, w, h)

                if touched:
                    touch_frames += 1
                    no_touch_frames = 0
                    cv.putText(frame, "Face Touching...", (30, 60),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if touch_frames >= frame_threshold:
                        if len(self.penalty_sections) == 0 or self.penalty_sections[-1] != "penalized":
                            self.penalized_sections += 1
                            self.penalty_sections.append("penalized")
                else:
                    if touch_frames > 0:
                        self.penalty_sections.append("non-penalized")
                    touch_frames = 0
                    no_touch_frames += 1
            else:
                touch_frames = 0
                no_touch_frames += 1

            # 점수 표시
            penalty = self.penalized_sections * self.penalty_per_violation
            score = max(0, 100 - penalty)
            cv.putText(frame, f"Score: {score}", (30, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv.imshow("Face Touch Detection", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()

        print(f"손 움직임 감지 분석 결과: {self.penalized_sections}회의 불필요한 손 움직임으로 인해 감점 {penalty}점, 점수는 {score}점입니다!")


    def run_and_get_result(self):
        self.run()
        penalty = self.penalized_sections * self.penalty_per_violation
        score = max(0, 100 - penalty)
        return score, penalty, self.penalized_sections

if __name__ == "__main__":
    video_path = "test.mp4"
    detector = FaceTouchDetectorVideo(video_path, penalty_per_violation=10)
    detector.run()
