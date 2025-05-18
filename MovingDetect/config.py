# config.py

import os

# 프로젝트 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "videos/move1.MP4")  # 입력 비디오 경로
OUTPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "output/move1.MP4")  # 출력 비디오 경로

# 움직임 감지 설정
MOVEMENT_THRESHOLD = 50  # 큰 움직임으로 판단하는 임계값 (픽셀)
RELATIVE_MOVEMENT_THRESHOLD = 0.15  # 상대적 움직임 임계값 (신체 크기 대비 비율)
SMOOTHING_WINDOW = 5  # 움직임 평활화를 위한 프레임 수

# YOLO 설정
YOLO_MODEL = 'yolov8n-pose.pt'  # YOLOv8 포즈 모델
CONFIDENCE_THRESHOLD = 0.5  # 객체 감지 신뢰도 임계값

# 비디오 설정
FRAME_SKIP = 1  # 처리할 프레임 간격 (1 = 모든 프레임 처리)

# 시각화 설정
TEXT_COLOR = (0, 255, 0)  # 텍스트 색상 (BGR)
TEXT_POSITION = (50, 50)  # 텍스트 위치
FONT_SCALE = 1.0  # 폰트 크기
FONT_THICKNESS = 2  # 폰트 두께