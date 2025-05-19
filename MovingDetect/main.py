import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict, Any
import shutil
import tempfile

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# MediaPipe POSE 초기화 (tr10.py와 동일)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,  # 비디오 스트림에 적합하게 False로 설정
    model_complexity=2,       # 모델 복잡도
    min_detection_confidence=0.5, # 최소 감지 신뢰도
    min_tracking_confidence=0.5   # 최소 추적 신뢰도
)

# ▶ 설정값 (tr10.py와 동일)
MOVEMENT_THRESHOLD = 40  # 움직임 감지 임계값

def process_video(video_path: str) -> List[Dict[str, str]]:
    """
    비디오 파일을 분석하여 움직임 타임스탬프 목록을 반환합니다.
    tr10.py의 핵심 로직을 사용합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 비디오 파일을 열 수 없는 경우 예외 발생
        raise HTTPException(status_code=500, detail="비디오 파일을 열 수 없습니다.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # frame_interval_ms = int(1000 / fps) if fps > 0 else 16 # GUI가 없으므로 이 부분은 직접 사용되지 않음

    # ROI 기준점 저장용 딕셔너리
    ref_positions = {part: {'ref': None} for part in
                     ['hand_left', 'hand_right', 'foot_left', 'foot_right']}
    
    recorded_entries_details = [] # 감지된 타임스탬프 상세 정보 저장 (중복 제거 전)
    recorded_timestamps_set = set() # 고유한 타임스탬프 문자열 저장 (중복 제거용)

    frame_count = 0
    while cap.isOpened():
        # start_time = time.time() # GUI가 없으므로 프레임 간 지연 계산 불필요
        ret, frame = cap.read()
        if not ret:
            # 프레임이 더 이상 없으면 루프 종료
            break
        
        frame_count += 1
        h, w = frame.shape[:2]

        # BGR 이미지를 RGB로 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pose Process
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = {
                'hand_left': lm[mp_pose.PoseLandmark.LEFT_WRIST],
                'hand_right': lm[mp_pose.PoseLandmark.RIGHT_WRIST],
                'foot_left': lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
                'foot_right': lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            }
            event_detected_in_frame = False
            # active_parts_in_frame = [] # 현재 요청 형식에서는 사용되지 않음

            for key, lm_pt in coords.items():
                if lm_pt.visibility < 0.3: # 가시성이 낮은 랜드마크는 건너뛰기
                    continue

                cx, cy = int(lm_pt.x * w), int(lm_pt.y * h)
                current_pos = np.array([cx, cy], dtype=float)
                buf = ref_positions[key]

                # 기준점이 없으면 현재 위치를 기준점으로 설정
                if buf['ref'] is None:
                    buf['ref'] = current_pos

                # 현재 위치와 기준점 사이의 거리 계산 (tr10.py 로직)
                # dist = np.linalg.norm(current_pos - buf['ref']) # 직접 사용되지 않음
                midpoint = (buf['ref'] + current_pos) / 2
                center_dist = np.linalg.norm(current_pos - midpoint)

                if center_dist > MOVEMENT_THRESHOLD:
                    event_detected_in_frame = True
                    # active_parts_in_frame.append(key) # 현재 요청 형식에서는 사용되지 않음
                    buf['ref'] = current_pos # 기준점 업데이트

            if event_detected_in_frame:
                timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC) # 밀리초 단위 타임스탬프
                timestamp_sec = int(timestamp_msec / 1000)     # 초 단위로 변환
                
                minutes = timestamp_sec // 60
                seconds = timestamp_sec % 60
                formatted_timestamp = f"{minutes:02d}:{seconds:02d}" # MM:SS 형식

                # 중복되지 않은 타임스탬프만 추가
                if formatted_timestamp not in recorded_timestamps_set:
                    recorded_timestamps_set.add(formatted_timestamp)
                    recorded_entries_details.append({
                        "category": "move",
                        "timeStamp": formatted_timestamp
                    })
                    # print(f"⚠️ {formatted_timestamp} 격한 움직임 감지") # 서버 로그용 (선택 사항)
    
    cap.release() # 비디오 캡처 객체 해제
    # cv2.destroyAllWindows() # GUI가 없으므로 필요 없음

    return recorded_entries_details

@app.post("/upload_video/", response_model=Dict[str, List[Dict[str, str]]])
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    """
    비디오 파일을 업로드 받아 움직임을 분석하고, 감지된 타임스탬프를 JSON으로 반환합니다.
    - **uploaded_file**: 업로드할 비디오 파일 (mp4, avi, mov 등)
    """
    # 임시 파일로 저장하여 cv2에서 처리할 수 있도록 함
    # tempfile.NamedTemporaryFile은 컨텍스트를 벗어나면 자동으로 삭제됨
    tmp_path = None # tmp_path 초기화
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.filename)[1]) as tmp:
            shutil.copyfileobj(uploaded_file.file, tmp)
            tmp_path = tmp.name # 임시 파일 경로 저장
    finally:
        if uploaded_file.file: # 파일 객체가 존재하는지 확인 후 닫기
            uploaded_file.file.close() # 업로드된 파일 객체 닫기

    try:
        # 비디오 처리 함수 호출
        time_stamps = process_video(tmp_path)
    except HTTPException as e:
        # process_video에서 발생한 HTTP 예외는 그대로 전달
        raise e
    except Exception as e:
        # 기타 예외 발생 시 500 오류 반환
        # 실제 운영 환경에서는 더 구체적인 오류 로깅 및 처리가 필요할 수 있음
        print(f"Error processing video: {e}") # 서버 콘솔에 오류 로깅
        raise HTTPException(status_code=500, detail=f"비디오 처리 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 삭제
        if tmp_path and os.path.exists(tmp_path): # tmp_path가 None이 아니고 파일이 존재할 경우
            os.remove(tmp_path)
            
    return {"timeStamps": time_stamps}

# FastAPI 서버 실행 방법 (터미널에서):
# uvicorn main:app --reload
# 예: uvicorn main:app --host 0.0.0.0 --port 3000 --reload 
