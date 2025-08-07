import numpy as np
from collections import deque

class FaceTouchDetectorVideo:
    def __init__(self, penalty_per_violation=10, fps=30):
        self.penalty_per_violation = penalty_per_violation
        self.touch_frame_count = 0
        self.penalized_sections = 0
        self.penalty_sections = deque(maxlen=5)  # 최근 감점 기록 저장
        self.touch_frames = 0
        self.frame_threshold = int(fps * 0.5)  # 0.5초 이상 터치 시 감점

    def set_fps(self, fps):
        self.frame_threshold = int(fps * 0.5)

    def detect_face_touch(self, face_landmarks, hand_landmarks, w, h):
        if not face_landmarks or not hand_landmarks:
            return False

        face_points = np.array([[pt[0], pt[1]] for pt in face_landmarks.values()])
        for hand in hand_landmarks:
            hand_points = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])
            for hp in hand_points:
                distances = np.linalg.norm(face_points - hp, axis=1)
                if np.any(distances < 40):  # 40px 이내면 터치로 간주
                    return True
        return False

    def process(self, face_landmarks, hand_landmarks, w, h):
        if self.detect_face_touch(face_landmarks, hand_landmarks, w, h):
            self.touch_frames += 1
            if self.touch_frames >= self.frame_threshold:
                if len(self.penalty_sections) == 0 or self.penalty_sections[-1] != "penalized":
                    self.penalized_sections += 1
                    self.penalty_sections.append("penalized")
        else:
            self.touch_frames = 0

    def get_result(self):
        penalty = self.penalized_sections * self.penalty_per_violation
        score = max(0, 100 - penalty)
        reasons = [f"얼굴 터치 {self.penalized_sections}회"] if self.penalized_sections > 0 else []
        return {
            "score": score,
            "penalty": penalty,
            "reasons": reasons
        }
