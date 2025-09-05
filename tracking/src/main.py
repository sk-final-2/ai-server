from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
from src.blink_detection.blink_detection import BlinkCounterVideo
from src.gaze_detection.gaze_detection import GazeDirectionVideo
from src.hand_detection.hand_detection import FaceTouchDetectorVideo
from src.head_detection.head_detection import HeadPoseVideo
import tempfile
import shutil
import os
import uuid
import subprocess

app = FastAPI()

# webm → mp4 변환 유틸
def transcode_to_mp4(src_path: str) -> str:
    dst_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-r", "30", "-an",  # 오디오 제거
        dst_path
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        print("[FFMPEG_ERR]", (cp.stderr or "")[:500])
        raise RuntimeError("ffmpeg 변환 실패")
    return dst_path

def probe_needs_transcode(path: str, sample_frames: int = 10) -> bool:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release(); return True
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # 메타 이상
    if fps > 120 or fps <= 0 or frames_meta < 0:
        cap.release(); return True
    # POS_MSEC 증가 여부 체크
    prev = -1.0; progressed = False
    for _ in range(sample_frames):
        ok, _ = cap.read()
        if not ok: break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        if t > prev:
            progressed = True
        prev = t
    cap.release()
    return not progressed

def measure_center(video_path, model, predictor_func, calibrate_func, label=""):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[CALIB:{label}] 영상 열기 실패")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    total_pitch = total_yaw = total_roll = 0.0
    count = 0
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if start_time is None:
            start_time = t_sec

        # 3초까지만 사용
        if t_sec - start_time > 3.0:
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
    cap.release()

    if count > 0:
        calibrate_func(
            total_pitch / count,
            total_yaw / count,
            total_roll / count
        )
    print(f"[CALIB:{label}] frames_used={count}, duration~{(t_sec-start_time):.2f}s")

def run_all_analyses(video_path):
    cap = cv2.VideoCapture(video_path)
    print("[OPEN]", "path:", video_path, "opened:", cap.isOpened())
    if not cap.isOpened():
        return None, "영상 열기 실패"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("[META]", f"size=({w}x{h}) fps={fps} frames_meta={frames_meta}")

    ok, frame0 = cap.read()
    print("[FIRST_READ]", "ok:", ok, "frame_none:", frame0 is None)
    if not ok or frame0 is None:
        return None, "프레임 읽기 실패(코덱/파일 문제 가능)"

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 모듈 생성
    blink = BlinkCounterVideo()
    gaze = GazeDirectionVideo()
    face_touch = FaceTouchDetectorVideo()
    head_pose = HeadPoseVideo()
    face_touch.set_fps(30.0)  # 기본값 30으로 두고 POS_MSEC으로 시간 계산

    # 센터 보정 (고개/시선)  ← 이건 영상 경로로 새 cap 열어서
    measure_center(video_path, head_pose, head_pose._predict_pose, head_pose.calibrate_center, "HeadPose")
    measure_center(video_path, gaze, gaze.predict_head_pose, gaze.calibrate_center, "Gaze")

    # 기존 cap 폐기하고 메인 루프용으로 새 cap 열기
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "영상 재오픈 실패"

    # 솔루션 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hands_every = 3
    last_hand_res = None

    frame_idx = 0
    last_sec = -1
    max_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # POS_MSEC 기반 시간
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms <= 0:   # fallback
            t_sec = frame_idx / 30.0
        else:
            t_sec = timestamp_ms / 1000.0
        max_time = max(max_time, t_sec)

        # FaceMesh 처리
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        if not face_results.multi_face_landmarks:
            continue

        landmarks_obj = face_results.multi_face_landmarks[0].landmark
        landmarks_xy = {i: (landmarks_obj[i].x * w, landmarks_obj[i].y * h) for i in range(len(landmarks_obj))}

        if (frame_idx % hands_every) == 0:
            last_hand_res = hands.process(rgb)
        hand_lms = (last_hand_res.multi_hand_landmarks if (last_hand_res and last_hand_res.multi_hand_landmarks) else None)

        blink.process(landmarks_xy, t_sec)
        gaze.process(landmarks_obj, w, h, t_sec)
        face_touch.process(landmarks_xy, hand_lms, w, h, t_sec)
        head_pose.process(landmarks_obj, t_sec)

    cap.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()

    # duration은 POS_MSEC 기반으로 계산
    total_secs = max(1.0, round(max_time))
    print("[DURATION]", "max_time:", max_time, "used total_secs:", total_secs)

    # 단위 감점 세팅
    for m in (blink, gaze, face_touch, head_pose):
        if hasattr(m, "set_video_duration"):
            m.set_video_duration(total_secs, decimals=1)

    blink_res = blink.get_result()
    gaze_res  = gaze.get_result()
    head_res  = head_pose.get_result()
    hand_res  = face_touch.get_result()

    def summarize(name, r):
        if r["reasons"]:
            return f"{name}: {', '.join(r['reasons'])}로 인해 감점 {r['penalty']:.1f}점, 점수는 {r['score']:.1f}점입니다!"
        return f"{name}: 감점 없이 만점입니다! 점수는 {r['score']:.1f}점입니다!"

    text_summary = "\n".join([
        summarize("눈 깜빡임 감지 분석 결과", blink_res),
        summarize("시선처리 감지 분석 결과",   gaze_res),
        summarize("고개 각도 감지 분석 결과", head_res),
        summarize("손 움직임 감지 분석 결과", hand_res),
    ])

    timestamps = []
    timestamps.extend(blink_res.get("events", []))
    timestamps.extend(gaze_res.get("events", []))
    timestamps.extend(head_res.get("events", []))
    timestamps.extend(hand_res.get("events", []))
    def key_ts(e):
        mm, ss = e["time"].split(":")
        return int(mm) * 60 + int(ss)
    timestamps.sort(key=key_ts)

    return {
        "text": text_summary.strip(),
        "blinkScore": blink_res["score"],
        "eyeScore":   gaze_res["score"],
        "headScore":  head_res["score"],
        "handScore":  hand_res["score"],
        "timestamp":  timestamps
    }, None


@app.post("/tracking")
async def analyze_tracking(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        # 1) 원본 저장 (확장자 유지)
        orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
        src_path = f"temp_{uuid.uuid4()}{orig_ext}"
        with open(src_path, "wb") as f: shutil.copyfileobj(file.file, f)

        use_path = src_path
        # 2) 메타/타임스탬프 이상하면 무조건 변환
        if probe_needs_transcode(src_path):
            print("[TRANSCODE] abnormal meta/pos_msec → convert to h264 mp4")
            use_path = transcode_to_mp4(src_path)

        # 3) 분석
        result, error = run_all_analyses(use_path)

        # 4) 정리
        for p in {src_path, use_path}:
            try:
                if p and os.path.exists(p): os.remove(p)
            except: pass

        if error:
            return JSONResponse(content={"error": error}, status_code=400)

        # 3. 응답 형식 통일
        return {
            "interviewId": interviewId,
            "seq": seq,
            "text": result["text"],
            "blinkScore": result["blinkScore"],
            "eyeScore": result["eyeScore"],
            "headScore": result["headScore"],
            "handScore": result["handScore"],
            "timestamp": result["timestamp"]
        }

    except Exception as e:
        print("[TRACKING_ERR]", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})