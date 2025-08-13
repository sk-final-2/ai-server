import numpy as np
from collections import deque
from src.utils.common import sec_to_timestamp
class FaceTouchDetectorVideo:
    def __init__(self, penalty_per_violation=10, fps=30):
        self.penalty_per_violation = penalty_per_violation
        self.frame_threshold = int(fps * 0.5)  # 0.5초 이상 터치 시 감점

        self.touch_frames = 0
        self.in_touch = False  # 연속 터치 구간 중복 카운트 방지

        self.penalized_sections = 0
        self.events = []  # [{"time":"MM:SS","reason":"손 움직임"}]

    def set_fps(self, fps):
        self.frame_threshold = int(fps * 0.5)

    def detect_face_touch(self, face_landmarks, hand_landmarks, w, h) -> bool:
        if not face_landmarks or not hand_landmarks:
            return False

        face_points = np.array([[pt[0], pt[1]] for pt in face_landmarks.values()])
        for hand in hand_landmarks:
            hand_points = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])
            dmin = np.min(np.linalg.norm(face_points[None, :, :] - hand_points[:, None, :], axis=2), axis=1)
            if np.any(dmin < 40):
                return True
        return False

    def process(self, face_landmarks, hand_landmarks, w, h, t_sec: float):
        touching = self.detect_face_touch(face_landmarks, hand_landmarks, w, h)

        if touching:
            self.touch_frames += 1
            # 아직 터치 구간으로 기록되지 않았고, 0.5초 이상 연속 터치면 1회 인정
            if not self.in_touch and self.touch_frames >= self.frame_threshold:
                self.in_touch = True
                self.penalized_sections += 1
                self.events.append({
                    "time": sec_to_timestamp(t_sec),
                    "reason": "손 움직임"
                })

        else:
            # 터치 끊기면 상태 초기화(다음 터치 구간을 새로 카운트)
            self.touch_frames = 0
            self.in_touch = False

    def get_result(self):
        penalty = self.penalized_sections * self.penalty_per_violation
        score = max(0, 100 - penalty)
        reasons = [f"얼굴 터치 {self.penalized_sections}회"] if self.penalized_sections > 0 else []
        return {
            "score": score,
            "penalty": penalty,
            "reasons": reasons,
            "events": self.events
        }
