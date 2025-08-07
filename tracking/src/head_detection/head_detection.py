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

    def __init__(self, pitch_tol=0.01, yaw_tol=0.02, roll_tol=0.02):
        self.PITCH_TOL = pitch_tol
        self.YAW_TOL = yaw_tol
        self.ROLL_TOL = roll_tol
        self.penalty_reasons_list = []

        self.pitch_center = 0
        self.yaw_center = 0
        self.roll_center = 0
        self.calibrated = False
        self.prev_reason = None  # 연속 중복 방지용

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

    def process(self, landmarks):
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

        # ✅ 같은 사유 중복 감지 방지
        if self.prev_reason != reason:
            self.penalty_reasons_list.append(reason)
            self.prev_reason = reason

    def get_result(self):
        penalty = len(self.penalty_reasons_list) * 5
        score = max(0, 100 - penalty)
        reason_counts = Counter(self.penalty_reasons_list)
        reasons = [
            f"{self.REASON_TRANSLATIONS.get(reason, reason)} {count}회"
            for reason, count in reason_counts.items()
        ]
        return {
            "score": score,
            "penalty": penalty,
            "reasons": reasons if reasons else ["위반 없음"]
        }
