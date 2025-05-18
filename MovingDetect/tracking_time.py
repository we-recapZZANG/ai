import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO
# ▶ YOLOv8 Pose 모델 로드 (GPU 사용 가능)
model = YOLO("yolov8n-pose.pt")
# ▶ 영상 경로
video_path = "videos/move1.MP4"
if not os.path.exists(video_path):
    print("❗ 입력한 파일 경로가 존재하지 않습니다.")
    exit()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval_ms = int(1000 / fps) if fps > 0 else 16
# ▶ 기준 위치 저장용 딕셔너리
ref_positions = {part: {'ref': None} for part in ['hand_left', 'hand_right', 'foot_left', 'foot_right']}
# ▶ 설정값
MOVEMENT_THRESHOLD = 500
log_filename = f"movement_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
recorded_entries = set()
# ▶ Keypoint 인덱스 (COCO 기준 YOLOv8-pose: 17 keypoints)
kp_map = {
    'hand_left': 9,    # LEFT_WRIST
    'hand_right': 10,  # RIGHT_WRIST
    'foot_left': 15,   # LEFT_ANKLE
    'foot_right': 16   # RIGHT_ANKLE
}
# ▶ 메인 루프
with open(log_filename, 'w', encoding='utf-8') as f:
    f.write("격한 움직임 감지 시각 목록\n")
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        results = model(frame, verbose=False)[0]
        if results.keypoints is not None:
            keypoints = results.keypoints.data.cpu().numpy()  # (N, 17, 3)
            # 한 프레임에 여러 명이 잡힐 수 있지만 여기서는 첫 번째 사람만
            if len(keypoints) > 0:
                person_kp = keypoints[0]
                event_detected = False
                active_parts = []
                for key, idx in kp_map.items():
                    x, y, conf = person_kp[idx]
                    if conf < 0.3:
                        continue
                    cx, cy = int(x), int(y)
                    current = np.array([cx, cy], dtype=float)
                    buf = ref_positions[key]
                    # ROI 시각화
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
                    timestamp = f"{timestamp_sec}초"
                    record = f"{timestamp} - {', '.join(active_parts)}"
                    if record not in recorded_entries:
                        print(f"⚠️ {timestamp} 격한 움직임: {', '.join(active_parts)}")
                        f.write(record + "\n")
                        recorded_entries.add(record)

        elapsed = (time.time() - start_time) * 1000
        wait_time = max(1, int(frame_interval_ms - elapsed))

    cap.release()

print(f"\n✅ 총 {len(recorded_entries)}건 저장됨 → {log_filename}")

