import cv2
import mediapipe as mp
import numpy as np

from blink_detection.blink_detection import BlinkCounterVideo
from gaze_detection.gaze_detection import GazeDirectionVideo
from hand_detection.hand_detection import FaceTouchDetectorVideo
from head_detection.head_detection import HeadPoseVideo
from blink_detection.FaceMeshModule import FaceMeshGenerator

def measure_center(cap, model, predictor_func, calibrate_func, label=""):
    total_pitch = total_yaw = total_roll = 0
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * 3)  # 3초

    print(f"[INFO] {label} 기준값 캘리브레이션 중...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            pitch, yaw, roll = predictor_func(landmarks)
            total_pitch += pitch
            total_yaw += yaw
            total_roll += roll
            count += 1

    face_mesh.close()

    if count > 0:
        calibrate_func(
            total_pitch / count,
            total_yaw / count,
            total_roll / count
        )
        print(f"[INFO] {label} 캘리브레이션 완료")
    else:
        print(f"[WARN] {label} 캘리브레이션 실패")

def run_all_analyses(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 분석기 초기화
    generator = FaceMeshGenerator()
    blink = BlinkCounterVideo()
    gaze = GazeDirectionVideo()
    face_touch = FaceTouchDetectorVideo()
    head_pose = HeadPoseVideo()

    face_touch.set_fps(fps)

    # 캘리브레이션
    measure_center(cap, head_pose, head_pose._predict_pose, head_pose.calibrate_center, label="Head Pose")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    measure_center(cap, gaze, gaze.predict_head_pose, gaze.calibrate_center, label="Gaze")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # face mesh 검출기
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 손 검출기
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 튜플 기반 랜드마크 (시각화용)
        frame, landmarks_xy = generator.create_face_mesh(frame, draw=False)

        # 객체 기반 랜드마크 (머신러닝 모델 입력용)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        if not face_results.multi_face_landmarks:
            continue
        landmarks_obj = face_results.multi_face_landmarks[0].landmark

        # 손 랜드마크
        hand_results = hands.process(rgb)
        hand_lms = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else None

        # 분석기별 처리
        blink.process(landmarks_xy)  # 튜플 (x, y)
        gaze.process(landmarks_obj, w, h)  # landmark 객체
        face_touch.process(landmarks_xy, hand_lms, w, h)  # 튜플
        head_pose.process(landmarks_obj)  # landmark 객체

    cap.release()
    cv2.destroyAllWindows()

    total_score = 0
    num_criteria = 0

    for analyzer, description in [
        (blink, "눈 깜빡임 감지 분석 결과"),
        (gaze, "시선처리 감지 분석 결과"),
        (head_pose, "고개 각도 감지 분석 결과"),
        (face_touch, "손 움직임 감지 분석 결과"),
    ]:
        result = analyzer.get_result()
        score = result['score']
        penalty = result['penalty']
        reasons = result['reasons']

        total_score += score
        num_criteria += 1

        if reasons:
            reasons_text = ", ".join(reasons)
            print(f"{description}: {reasons_text}로 인해 감점 {penalty}점, 점수는 {score}점입니다!\n")
        else:
            print(f"{description}: 감점 없이 만점입니다! 점수는 {score}점입니다!\n")
    
    # 평균 점수
    if num_criteria > 0:
        average_score = round(total_score / num_criteria)
        print(f"tracking 평균 점수는 {average_score}점입니다!")

if __name__ == "__main__":
    video_path = "test.mp4"
    run_all_analyses(video_path)
