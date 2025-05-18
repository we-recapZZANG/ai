import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import pygame
import os
from datetime import datetime

def initialize_alert_system():
    """알림 시스템 초기화 함수"""
    pygame.mixer.init()
    # 알림음 파일 경로 설정 (기본 경로, 실제 경로로 변경 필요)
    alert_sound_file = "alert.mp3"
    
    # 알림음 파일이 없으면 기본 비프음 생성
    if not os.path.exists(alert_sound_file):
        # 간단한 비프음 생성
        duration = 1  # 비프음 지속 시간(초)
        frequency = 440  # 주파수(Hz)
        sample_rate = 44100  # 샘플링 레이트
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.5
        
        sound = pygame.mixer.Sound(np.asarray(tone * 32767, dtype=np.int16))
        return pygame, sound
    else:
        pygame.mixer.music.load(alert_sound_file)
        return pygame, None

def play_alert(pygame_instance, sound=None, duration=2):
    """알림음 재생 함수"""
    if sound:
        sound.play()
    else:
        pygame_instance.mixer.music.play()
    time.sleep(duration)
    if sound:
        sound.stop()
    else:
        pygame_instance.mixer.music.stop()

def save_alert_frame(frame, output_dir="alerts"):
    """알림 시 프레임 저장 함수"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"alert_{timestamp}.jpg")
    cv2.imwrite(file_path, frame)
    print(f"알림 이미지 저장됨: {file_path}")
    
    return file_path

def setup_video_writer(cap, output_dir="video_results"):
    """비디오 작성자 설정 함수"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # 기본값 설정
        
    # 타임스탬프로 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"brightmoveandturnleft_{timestamp}.mp4")
    
    # 비디오 작성자 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 포맷
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"녹화 시작: {output_path}")
    return video_writer, output_path

def monitor_baby_position(
    model_path,
    video_source=0,
    conf_threshold=0.4,
    alert_cooldown=10,
    save_alerts=True,
    show_video=True,
    save_video=True
):
    """
    아기 자세 모니터링 및 알림 메인 함수
    
    Parameters:
        model_path (str): YOLOv8 모델 파일 경로
        video_source (int or str): 카메라 인덱스 또는 동영상 파일 경로
        conf_threshold (float): 감지 신뢰도 임계값
        alert_cooldown (int): 알림 사이의 최소 시간(초)
        save_alerts (bool): 알림 시 이미지 저장 여부
        show_video (bool): 비디오 화면 표시 여부
        save_video (bool): 감지 결과 영상 저장 여부
    """
    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = YOLO(model_path)
    
    # 클래스 이름 확인
    class_names = model.names
    print(f"감지 클래스: {class_names}")
    
    # 비디오 소스 열기
    print(f"비디오 소스 연결 중: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"오류: 비디오 소스 {video_source}를 열 수 없습니다.")
        return
    
    # 알림 시스템 초기화
    pygame_instance, sound = initialize_alert_system()
    
    # 화면 크기 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 비디오 저장 설정
    video_writer = None
    output_video_path = None
    if save_video:
        video_writer, output_video_path = setup_video_writer(cap)
    
    # 알림 관련 변수 초기화
    last_alert_time = 0
    alert_active = False
    pose_down_duration = 0
    start_pose_down_time = None
    frame_count = 0
    
    print("모니터링 시작...")
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("비디오 스트림 종료")
                break
            
            frame_count += 1
            
            # 현재 시간
            current_time = time.time()
            
            # YOLOv8 모델로 감지 실행
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # 결과 시각화를 위한 프레임 복사
            annotated_frame = frame.copy()
            
            # PoseDOWN 감지 여부
            pose_down_detected = False
            
            # 감지 결과 출력 (콘솔)
            detected_objects = []
            
            # 결과에서 검출된 객체 처리
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # 신뢰도와 클래스 ID 가져오기
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    
                    # 감지 목록에 추가
                    detected_objects.append(f"{class_name}: {confidence:.2f}")
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    
                    # PoseDOWN이 감지되면 알림 준비
                    if class_name == "PoseDOWN" and confidence >= conf_threshold:
                        pose_down_detected = True
                        
                        # 박스 색상 설정 (PoseDOWN은 빨간색, PoseUP은 녹색)
                        color = (0, 0, 255)  # 빨간색
                    else:
                        color = (0, 255, 0)  # 녹색
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 클래스명과 신뢰도 표시
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 감지 결과를 콘솔에 출력 (10프레임마다 출력)
            if frame_count % 10 == 0 and detected_objects:
                print(f"프레임 {frame_count} 감지 결과: {', '.join(detected_objects)}")
            
            # PoseDOWN 감지 시간 추적
            if pose_down_detected:
                if start_pose_down_time is None:
                    start_pose_down_time = current_time
                pose_down_duration = current_time - start_pose_down_time
            else:
                start_pose_down_time = None
                pose_down_duration = 0
            
            # 지속 시간 표시
            cv2.putText(
                annotated_frame, 
                f"PoseDOWN Duration: {pose_down_duration:.1f}s", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255) if pose_down_duration > 3 else (255, 255, 255), 
                2
            )
            
            # 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated_frame,
                timestamp,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # PoseDOWN이 3초 이상 지속되고 마지막 알림으로부터 쿨다운 시간이 지났을 때 알림
            if pose_down_detected and pose_down_duration > 3 and (current_time - last_alert_time) > alert_cooldown:
                if not alert_active:
                    print("경고: 아기가 엎드려 있습니다!")
                    
                    # 알림음 재생
                    play_alert(pygame_instance, sound)
                    
                    # 알림 이미지 저장
                    if save_alerts:
                        save_alert_frame(annotated_frame)
                    
                    last_alert_time = current_time
                    alert_active = True
            
            # PoseUP이 감지되면 알림 상태 해제
            elif not pose_down_detected:
                alert_active = False
            
            # 알림 상태 표시
            if alert_active:
                cv2.putText(
                    annotated_frame, 
                    "경고: 아기가 엎드려 있습니다!", 
                    (width // 2 - 150, height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # 비디오 저장
            if save_video and video_writer is not None:
                video_writer.write(annotated_frame)
            
            # 비디오 화면 표시
            if show_video:
                cv2.imshow("아기 자세 모니터링", annotated_frame)
                
                # q 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # 자원 해제
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"영상 저장 완료: {output_video_path}")
        if show_video:
            cv2.destroyAllWindows()
        print("모니터링 종료")

# 여기에서 직접 변수를 정의하고 함수를 호출합니다 (Jupyter Notebook용)
# 모델 경로 - 실제 모델 파일 경로로 업데이트하세요
model_path = "/mnt/hdd1/users/20011959_son/proneDetect/runs/detect/train17/weights/best.pt"

# 모델 파일이 존재하는지 확인하고 대체 모델 경로 시도
if not os.path.exists(model_path):
    print(f"경고: 지정된 모델 파일이 존재하지 않습니다: {model_path}")
    print("다른 모델 경로를 시도합니다...")
    
    # 대체 경로 시도
    alternative_paths = [
        "/mnt/hdd1/users/20011959_son/proneDetect/runs/detect/train15/weights/best.pt",
        "/mnt/hdd1/users/20011959_son/proneDetect/runs/detect/train16/weights/best.pth",
        "/mnt/hdd1/users/20011959_son/proneDetect/runs/detect/train15/weights/best.pth",
        # 기본 YOLOv8 모델도 대체 옵션으로 추가
        "yolov8n.pt"
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            print(f"대체 모델 파일을 찾았습니다: {alt_path}")
            model_path = alt_path
            break

# 비디오 소스 - 웹캠(0) 또는 동영상 파일 경로
video_source = "/mnt/hdd1/users/20011959_son/proneDetect/videos/brightmoveandturnleft.MP4"  # 웹캠 사용, 파일을 사용하려면 "파일경로.mp4"와 같이 지정

# 이 셀을 실행하면 모니터링이 시작됩니다
monitor_baby_position(
    model_path=model_path,
    video_source=video_source,
    conf_threshold=0.4,  # 감지 신뢰도 임계값
    alert_cooldown=10,   # 알림 사이의 최소 시간(초)
    save_alerts=True,    # 알림 이미지 저장 여부
    show_video=True,     # 비디오 화면 표시 여부
    save_video=True      # 감지 결과 영상 저장 여부
)