import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime

# ▶ MediaPipe POSE 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ▶ 영상 경로 입력 받기
video_path = "videos/movebright2.MP4"

if not os.path.exists(video_path):
    print("❗ 입력한 파일 경로가 존재하지 않습니다.")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval_ms = int(1000 / fps) if fps > 0 else 16

# ▶ ROI 기준점 저장용 딕셔너리
ref_positions = {part: {'ref': None} for part in
                 ['hand_left', 'hand_right', 'foot_left', 'foot_right']}

# ▶ 설정값
MOVEMENT_THRESHOLD = 40
log_filename = f"movement_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
recorded_entries = set()

# ▶ 메인 루프
with open(log_filename, 'w', encoding='utf-8') as f:
    # f.write("격한 움직임 감지 시각 목록\n")
    frame_count = 0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = {
                'hand_left': lm[mp_pose.PoseLandmark.LEFT_WRIST],
                'hand_right': lm[mp_pose.PoseLandmark.RIGHT_WRIST],
                'foot_left': lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
                'foot_right': lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            }
            event_detected = False
            active_parts = []

            for key, lm_pt in coords.items():
                if lm_pt.visibility < 0.3:
                    continue

                cx, cy = int(lm_pt.x * w), int(lm_pt.y * h)
                current = np.array([cx, cy], dtype=float)
                buf = ref_positions[key]

                # ▶ ROI 시각화
                roi_size = 20
                color = (255, 0, 0) if 'hand' in key else (0, 0, 255)
                cv2.rectangle(frame, (cx - roi_size, cy - roi_size), (cx + roi_size, cy + roi_size), color, 2)
                cv2.putText(frame, 'hand' if 'hand' in key else 'foot',
                            (cx - roi_size, cy - roi_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if buf['ref'] is None:
                    buf['ref'] = current

                dist = np.linalg.norm(current - buf['ref'])
                midpoint = (buf['ref'] + current) / 2
                center_dist = np.linalg.norm(current - midpoint)
                if center_dist > MOVEMENT_THRESHOLD:
                    event_detected = True
                    active_parts.append(key)
                    buf['ref'] = current

            if event_detected:
                timestamp_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                minutes = timestamp_sec // 60
                seconds = timestamp_sec % 60
                timestamp = f"{minutes:02d}:{seconds:02d}"

                record = f"{timestamp}"
                if record not in recorded_entries:
                    print(f"⚠️ {timestamp} 격한 움직임: {', '.join(active_parts)}")
                    f.write(record + "\n")
                    recorded_entries.add(record)

        # GUI 출력 제거
        # cv2.imshow("Video", frame)

        elapsed = (time.time() - start_time) * 1000
        wait_time = max(1, int(frame_interval_ms - elapsed))
        # GUI 관련 대기 제거
        # if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        #     break

    cap.release()
    # GUI 창 정리 코드 제거
    # cv2.destroyAllWindows()

print(f"\n✅ 총 {len(recorded_entries)}건 저장됨 → {log_filename}")
