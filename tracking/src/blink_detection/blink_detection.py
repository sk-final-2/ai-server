import cv2 as cv
import numpy as np
from collections import deque, Counter
from src.utils.common import sec_to_timestamp

class BlinkCounterVideo:
    # 눈의 EAR(Eye Aspect Ratio) 계산에 사용할 랜드마크 인덱스
    RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    # 눈 윤곽을 표시하기 위한 랜드마크 인덱스
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, ear_threshold=0.3, consec_frames=3, blink_limit_10s=10, penalty_per_excess=10):
        self.ear_threshold = ear_threshold # EAR 임계값 (이 값보다 작으면 눈이 감겼다고 판단)
        self.consec_frames = consec_frames # 눈이 감겼다고 판정하기 위한 최소 연속 프레임 수
        self.blink_counter = 0
        self.frame_counter = 0

        # 영상 시간(초)로만 관리
        self.blink_times = deque() # t_sec 값들
        self.blink_limit_10s = blink_limit_10s # 10초 동안 허용되는 최대 깜빡임 횟수
        self.penalty_per_excess = penalty_per_excess # 위반 시 감점 점수
        self.blink_violations = [] # 위반 기록 저장
        # self.last_violation_time = 0  # 직전 위반 체크 시각
        self.last_violation_time = -1e9   # 마지막 위반 기록된 t_sec
        self.events = []                  # [{"time": float, "reason": str}, ...]

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        """
        눈의 EAR(Eye Aspect Ratio) 계산
        A, B: 세로 거리
        C: 가로 거리
        EAR = (A + B) / (2.0 * C)
        """
        A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
        B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
        C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
        return (A + B) / (2.0 * C)

    def update_blink_count(self, ear, t_sec: float):
        """
        EAR 값 기반으로 눈 깜빡임 횟수 업데이트
        """
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                self.blink_times.append(t_sec)
            self.frame_counter = 0

    def calculate_score_and_text(self):
        if not self.blink_violations:
            return 100, "", 0, 0  # 점수, 이유 텍스트, 감점, 총 위반 수

        reason_counts = Counter(self.blink_violations)
        reason_text_parts = [
            f"{reason} {count}회"
            for reason, count in reason_counts.items()
        ]
        reasons_kor_text = ", ".join(reason_text_parts)

        total_violations = sum(reason_counts.values())
        penalty = total_violations * self.penalty_per_excess
        blink_score = max(0, 100 - penalty)

        return blink_score, reasons_kor_text, penalty, total_violations

    def process(self, landmarks, t_sec: float):
        right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, landmarks)
        left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, landmarks)
        ear = (right_ear + left_ear) / 2.0

        self.update_blink_count(ear, t_sec)

        while self.blink_times and (t_sec - self.blink_times[0]) > 10:
            self.blink_times.popleft()

        # 최근 10초 깜빡임 과다 → 10초 쿨다운으로 1회만 기록
        if len(self.blink_times) > self.blink_limit_10s and (t_sec - self.last_violation_time) > 10:
            self.blink_violations.append("과도한 깜빡임")
            self.last_violation_time = t_sec
            self.events.append({
                "time": sec_to_timestamp(t_sec),
                "reason": "눈 깜빡임"
            })

    def get_result(self):
        if not self.blink_violations:
            return {"score": 100, "penalty": 0, "reasons": [], "events": self.events}

        reason_counts = Counter(self.blink_violations)
        penalty = len(self.blink_violations) * self.penalty_per_excess
        return {
            "score": max(0, 100 - penalty),
            "penalty": penalty,
            "reasons": [f"{reason} {count}회" for reason, count in reason_counts.items()],
            "events": self.events
        }