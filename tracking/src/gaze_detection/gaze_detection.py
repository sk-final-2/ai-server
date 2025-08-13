import cv2 as cv
import mediapipe as mp
import numpy as np
from collections import Counter
from src.blink_detection.FaceMeshModule import FaceMeshGenerator
import pickle
import os
from src.utils.common import sec_to_timestamp
class GazeDirectionVideo:
    LEFT_EYE_LANDMARKS = [33, 133]
    RIGHT_EYE_LANDMARKS = [362, 263]
    LEFT_IRIS = [468]
    RIGHT_IRIS = [473]
    HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 263, 291, 362]

    REASON_TRANSLATIONS = {
        "LOOK_LEFT": "오른쪽 응시",
        "LOOK_RIGHT": "왼쪽 응시",
        "LOOK_UP": "아래쪽 응시",
        "LOOK_DOWN": "위쪽 응시"
    }

    def __init__(self, penalty_per_violation=10, stable_frames_required=5, cooldown_secs=2.0):
        self.penalty_per_violation = penalty_per_violation
        self.stable_frames_required = stable_frames_required
        self.cooldown_secs = cooldown_secs
        self.violations = []
        self.events = []

        # 방향 안정화/중복 방지용 상태
        self.current_dir = "FORWARD"
        self.consec_count = 0
        self.last_event_time_sec = -1e9  # 마지막 이벤트 발생 시각(초)

        BASE_DIR = os.path.dirname(__file__)
        MODEL_PATH = os.path.join(BASE_DIR, "..", "head_detection", "model.pkl")
        MODEL_PATH = os.path.abspath(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            self.head_pose_model = pickle.load(f)

        self.PITCH_TOL = 0.05
        self.YAW_TOL = 0.08
        self.ROLL_TOL = 0.08

        self.PITCH_CENTER = 0.0
        self.YAW_CENTER = 0.0
        self.ROLL_CENTER = 0.0

    def calibrate_center(self, pitch, yaw, roll):
        self.PITCH_CENTER = pitch
        self.YAW_CENTER = yaw
        self.ROLL_CENTER = roll

    def predict_head_pose(self, landmarks):
        coords = []
        for idx in self.HEAD_POSE_LANDMARKS:
            lm = landmarks[idx]
            coords.extend([lm.x, lm.y])
        coords = np.array(coords).reshape(1, -1)
        return self.head_pose_model.predict(coords)[0]

    def is_head_forward(self, pitch, yaw, roll):
        return (
            abs(pitch - self.PITCH_CENTER) < self.PITCH_TOL and
            abs(yaw - self.YAW_CENTER) < self.YAW_TOL and
            abs(roll - self.ROLL_CENTER) < self.ROLL_TOL
        )

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

    def process(self, landmarks, w, h, t_sec: float):
        pitch, yaw, roll = self.predict_head_pose(landmarks)

        # '정면'일 때만 시선판정 (고개가 돌아가면 시선 무시)
        if not self.is_head_forward(pitch, yaw, roll):
            # 정면 아니면 상태 초기화(원하면 유지해도 됨)
            self.current_dir = "FORWARD"
            self.consec_count = 0
            return

        gaze_dir = self.get_gaze_direction(landmarks, w, h)

        # 현재 방향 안정화(연속 stable_frames_required 프레임 유지 시 '확정')
        if gaze_dir == self.current_dir:
            self.consec_count += 1
        else:
            self.current_dir = gaze_dir
            self.consec_count = 1

        # 이벤트/위반 기록 조건:
        # - 정면 아님
        # - 방향이 stable_frames_required 프레임 이상 유지
        # - 마지막 이벤트 이후 cooldown_secs 경과
        if (self.current_dir != "FORWARD" and
            self.consec_count >= self.stable_frames_required and
            (t_sec - self.last_event_time_sec) >= self.cooldown_secs):

            self.violations.append(self.current_dir)  # 점수 계산용
            self.last_event_time_sec = t_sec

            self.events.append({
                "time": sec_to_timestamp(t_sec),
                "reason": f"시선처리"
            })

    def get_result(self):
        if not self.violations:
            return {"score": 100, "penalty": 0, "reasons": [], "events": self.events}

        reason_counts = Counter(self.violations)
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
