import cv2
import time
import numpy as np
import torch # torch는 YOLO 모델 사용 시 내부적으로 필요할 수 있습니다.
from ultralytics import YOLO
import os
from datetime import datetime
import tempfile # 임시 파일 생성을 위해 추가
import shutil # 파일 복사를 위해 추가
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict # 타입 힌팅을 위해 추가

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# --- 설정값 ---
# 실제 운영 환경에서는 환경 변수나 설정 파일을 사용하는 것이 좋습니다.
MODEL_PATH_PRIMARY = "/mnt/hdd1/users/20011959_son/Capstone/proneDetect/runs/detect/train17/weights/best.pt"
ALTERNATIVE_MODEL_PATHS = [
    "/mnt/hdd1/users/20011959_son/Capstone/proneDetect/runs/detect/train15/weights/best.pt",
    "/mnt/hdd1/users/20011959_son/Capstone/proneDetect/runs/detect/train16/weights/best.pth",
    "/mnt/hdd1/users/20011959_son/Capstone/proneDetect/runs/detect/train15/weights/best.pth",
    "/mnt/hdd1/users/20011959_son/Capstone/proneDetect/yolov8n.pt"  # 기본 YOLOv8 모델
]
CONF_THRESHOLD = 0.4  # 감지 신뢰도 임계값

# --- 모델 로드 ---
model = None
CLASS_NAMES = {}

def load_model():
    """YOLO 모델을 로드하고 클래스 이름을 설정합니다."""
    global model, CLASS_NAMES
    
    selected_model_path = ""
    if os.path.exists(MODEL_PATH_PRIMARY):
        selected_model_path = MODEL_PATH_PRIMARY
    else:
        print(f"경고: 주 모델 파일이 존재하지 않습니다: {MODEL_PATH_PRIMARY}")
        for alt_path in ALTERNATIVE_MODEL_PATHS:
            if os.path.exists(alt_path):
                print(f"대체 모델 파일을 사용합니다: {alt_path}")
                selected_model_path = alt_path
                break
    
    if not selected_model_path:
        print("오류: 사용 가능한 모델 파일을 찾을 수 없습니다.")
        raise RuntimeError("사용 가능한 모델 파일을 찾을 수 없습니다. MODEL_PATH_PRIMARY 및 ALTERNATIVE_MODEL_PATHS를 확인하세요.")

    try:
        print(f"모델 로드 중: {selected_model_path}")
        model = YOLO(selected_model_path)
        CLASS_NAMES = model.names
        print(f"모델 로드 성공. 감지 클래스: {CLASS_NAMES}")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise RuntimeError(f"모델 로드 실패: {e}")

# 애플리케이션 시작 시 모델 로드
load_model()


def format_timestamp_from_seconds(seconds: float) -> str:
    """
    총 초를 MM:SS 형식의 문자열로 변환합니다.
    """
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def process_video_for_posedown(video_path: str) -> List[Dict[str, str]]:
    """
    비디오 파일을 처리하여 "PoseDOWN" 이벤트가 감지된 시간 목록을 반환합니다.

    Args:
        video_path (str): 처리할 비디오 파일의 경로.

    Returns:
        List[Dict[str, str]]: 각 감지 이벤트는 {"category": "PoseDown", "timeStamp": "MM:SS"} 형식.
    """
    if model is None:
        # 이 경우는 load_model()에서 이미 처리되어야 하지만, 안전장치로 추가합니다.
        raise RuntimeError("모델이 로드되지 않았습니다.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 비디오 파일 {video_path}를 열 수 없습니다.")
        # HTTP 예외를 발생시키기보다는 호출한 쪽에서 처리하도록 빈 리스트 또는 예외를 발생시킬 수 있습니다.
        # 여기서는 빈 리스트를 반환하거나, 특정 예외를 발생시켜 엔드포인트에서 처리하도록 할 수 있습니다.
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"경고: 비디오 FPS를 읽을 수 없습니다 ({video_path}). 기본값 30으로 설정합니다.")
        fps = 30  # FPS를 읽을 수 없는 경우 기본값

    detected_posedown_timestamps = []
    frame_number = 0
    # PoseDOWN 상태가 이전 프레임에서 감지되었는지 추적
    posedown_detected_in_previous_frame = False 
    
    print(f"영상 처리 시작: {video_path}, FPS: {fps}")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("비디오 스트림 종료 또는 프레임 읽기 실패.")
                break
            
            frame_number += 1
            # 현재 비디오 시간 (초)
            current_video_time_seconds = frame_number / fps

            # YOLOv8 모델로 감지 실행
            results = model(frame, conf=CONF_THRESHOLD, verbose=False) # verbose=False로 로그 최소화
            
            # 현재 프레임에서 PoseDOWN 감지 여부
            posedown_currently_detected = False
            
            for result in results: # 각 result는 이미지 하나에 대한 결과
                boxes = result.boxes.cpu().numpy() # GPU 사용 시 CPU로 데이터 이동
                for box in boxes:
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # 클래스 ID가 CLASS_NAMES에 있는지 확인
                    if class_id in CLASS_NAMES:
                        class_name = CLASS_NAMES[class_id]
                        # "PoseDOWN" 클래스이고 신뢰도 임계값을 넘는지 확인
                        if class_name == "PoseDOWN" and confidence >= CONF_THRESHOLD:
                            posedown_currently_detected = True
                            break # 현재 프레임에서 PoseDOWN 하나라도 감지되면 더 이상 박스 검사 안함
                if posedown_currently_detected:
                    break # PoseDOWN 감지 시 result 순회 중단
            
            # PoseDOWN 상태가 "시작"될 때 타임스탬프 기록
            # (이전 프레임에서는 감지X, 현재 프레임에서는 감지O)
            if posedown_currently_detected and not posedown_detected_in_previous_frame:
                timestamp_str = format_timestamp_from_seconds(current_video_time_seconds)
                detected_posedown_timestamps.append({
                    "category": "PoseDown", # 요청된 JSON 형식에 맞춤
                    "timeStamp": timestamp_str
                })
                print(f"PoseDown 감지: {timestamp_str} (프레임: {frame_number})")
            
            # 다음 프레임 처리를 위해 현재 감지 상태 업데이트
            posedown_detected_in_previous_frame = posedown_currently_detected

    except Exception as e:
        print(f"영상 처리 중 오류 발생: {e}")
        # 특정 오류를 발생시켜 엔드포인트에서 처리
        raise RuntimeError(f"영상 처리 중 내부 오류: {e}")
    finally:
        cap.release()
        print(f"영상 처리 완료: {video_path}")

    return detected_posedown_timestamps

@app.post("/detect_posedown/")
async def api_detect_posedown(file: UploadFile = File(...)):
    """
    업로드된 비디오 파일에서 "PoseDOWN" 자세를 감지합니다.
    감지된 각 "PoseDOWN" 시작 시점의 타임스탬프를 JSON 형식으로 반환합니다.
    """
    if model is None:
        # load_model() 실패 시 서버 시작이 안되지만, 만약을 대비
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다. 서버 로그를 확인하세요.")

    # 업로드된 파일을 임시 파일로 저장
    # tempfile.NamedTemporaryFile 사용 시 with 블록을 나가면 자동 삭제됨
    # delete=False로 하면 수동 삭제 필요
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file) # 비동기 파일 객체에서 동기 파일 객체로 복사
            tmp_file_path = tmp_file.name
        
        print(f"업로드된 파일이 임시 저장되었습니다: {tmp_file_path}")

        # 비디오 처리 함수 호출
        posedown_events = process_video_for_posedown(tmp_file_path)
        
        # 결과 JSON 반환
        return {
            "timeStamps": posedown_events
        }
    except ValueError as ve: # 비디오 파일 열기 실패 등
        print(f"입력 값 오류: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: # 모델 또는 처리 중 일반 오류
        print(f"런타임 오류: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="영상 처리 중 알 수 없는 오류가 발생했습니다.")
    finally:
        # 사용 후 임시 파일 삭제
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"임시 파일 삭제됨: {tmp_file_path}")
        await file.close() # 업로드 파일 객체 닫기

# FastAPI 서버를 실행하려면 터미널에서 다음 명령어를 사용하세요:
# uvicorn main:app --reload
# 예: (프로젝트 루트 디렉토리에서)
# python -m uvicorn main:app --reload --host localhost --port 3001
