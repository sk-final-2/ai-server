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
        """
        1단계: 얼굴/손 바운딩박스가 겹치는지 빠르게 검사 (겹치지 않으면 즉시 False)
        2단계: 겹칠 때만 세밀 최소거리 계산
        """
        if not face_landmarks or not hand_landmarks:
            return False
        
        # --- 얼굴 bbox (pad로 여유를 줘서 민감도 확보) ---
        xs = [pt[0] for pt in face_landmarks.values()]
        ys = [pt[1] for pt in face_landmarks.values()]
        fx1, fy1, fx2, fy2 = min(xs), min(ys), max(xs), max(ys)
        pad = 30
        fx1 -= pad; fy1 -= pad; fx2 += pad; fy2 += pad

        face_points = None

        for hand in hand_landmarks:
            # 손 bbox
            hx = [lm.x * w for lm in hand.landmark]
            hy = [lm.y * h for lm in hand.landmark]
            hx1, hy1, hx2, hy2 = min(hx), min(hy), max(hx), max(hy)

            # 빠른 충돌 검사: bbox가 안 겹치면 스킵
            if hx2 < fx1 or hx1 > fx2 or hy2 < fy1 or hy1 > fy2:
                continue

            # 여기서만 세밀 최소거리 계산
            if face_points is None:
                face_points = np.array([[pt[0], pt[1]] for pt in face_landmarks.values()])  # (F,2)

            hand_points = np.array([[x, y] for x, y in zip(hx, hy)], dtype=float)  # (H,2)

            # 모든 조합 최소거리 (H x F)
            diffs = face_points[None, :, :] - hand_points[:, None, :]
            dists = np.linalg.norm(diffs, axis=2)
            dmin = np.min(dists)

            if dmin < 40:
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
