import numpy as np
import pickle
import os
from collections import Counter
from src.utils.common import sec_to_timestamp
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

    def __init__(self, pitch_tol=0.01, yaw_tol=0.02, roll_tol=0.02, penalty_per_violation=5, cooldown_secs=2.0):
        self.PITCH_TOL = pitch_tol
        self.YAW_TOL = yaw_tol
        self.ROLL_TOL = roll_tol
        self.penalty_per_violation = penalty_per_violation
        self.cooldown_secs = cooldown_secs

        self.pitch_center = 0
        self.yaw_center = 0
        self.roll_center = 0
        self.calibrated = False

        self.prev_reason = None  # 연속 중복 방지용
        self.last_event_time_sec = -1e9  # 마지막 이벤트 발생 시각(초)

        self.penalty_reasons_list = []
        self.events = []  # [{"time":"MM:SS","reason":"고개 움직임"}]

        # 모델 로드
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def calibrate_center(self, pitch, yaw, roll):
        self.pitch_center = pitch
        self.yaw_center = yaw
        self.roll_center = roll
        self.calibrated = True

    def _predict_pose(self, landmarks):
        coords = []
        for idx in self.LANDMARK_IDX:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y])
        coords = np.array(coords).reshape(1, -1)
        return self.model.predict(coords)[0]  # pitch, yaw, roll

    def process(self, landmarks, t_sec: float):
        if not self.calibrated:
            return

        pitch, yaw, roll = self._predict_pose(landmarks)
        pitch_diff = pitch - self.pitch_center
        yaw_diff = yaw - self.yaw_center
        roll_diff = roll - self.roll_center

        if abs(pitch_diff) < self.PITCH_TOL and abs(yaw_diff) < self.YAW_TOL and abs(roll_diff) < self.ROLL_TOL:
            self.prev_reason = None  # 정면 응시 중이면 리셋
            return

        # 위반 사유 판단
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

        # 같은 사유 연속 방지 + 쿨다운 적용
        if self.prev_reason != reason and (t_sec - self.last_event_time_sec) >= self.cooldown_secs:
            self.penalty_reasons_list.append(reason)
            self.prev_reason = reason
            self.last_event_time_sec = t_sec
            self.events.append({
                "time": sec_to_timestamp(t_sec),
                "reason": "고개 움직임"
            })

    def get_result(self):
        if not self.penalty_reasons_list:
            return {"score": 100, "penalty": 0, "reasons": [], "events": self.events}
    
        reason_counts = Counter(self.penalty_reasons_list)
        penalty = sum(reason_counts.values()) * self.penalty_per_violation
        reasons = [
            f"{self.REASON_TRANSLATIONS.get(reason, reason)} {count}회"
            for reason, count in reason_counts.items()
        ]
        return {
            "score": max(0, 100 - penalty),
            "penalty": penalty,
            "reasons": reasons,
            "events": self.events
        }
