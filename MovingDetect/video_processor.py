# video_processor.py

import cv2
import config
from motion_detector import MotionDetector

class VideoProcessor:
    def __init__(self):
        self.motion_detector = MotionDetector()
        
    def process_video(self, input_path, output_path):
        """비디오를 처리하고 움직임을 감지하여 결과를 저장합니다."""
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {input_path}")
        
        # 비디오 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 비디오 작성기 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"비디오 처리 시작 - 총 {total_frames} 프레임")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 프레임 건너뛰기 설정
            if frame_count % config.FRAME_SKIP != 0:
                continue
            
            # 움직임 감지
            is_large_movement, movement_distance, camera_motion, keypoints = self.motion_detector.detect_movement(frame)
            
            # 프레임에 정보 표시
            frame_with_info = self.draw_info_on_frame(frame.copy(), is_large_movement, movement_distance, camera_motion, keypoints)
            
            # 프레임 저장
            out.write(frame_with_info)
            
            # 진행상황 출력
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"진행 중: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # 리소스 해제
        cap.release()
        out.release()
        
        print("비디오 처리 완료!")
        
    def draw_info_on_frame(self, frame, is_large_movement, movement_distance, camera_motion, keypoints):
        """프레임에 움직임 정보와 스켈레톤을 표시합니다."""
        # 상대적 움직임 정보 텍스트 생성
        movement_text = f"Relative Movement: {movement_distance:.3f}"
        
        # 카메라 움직임 정보
        camera_text = f"Camera Motion: ({camera_motion[0]:.1f}, {camera_motion[1]:.1f})"
        
        # 큰 움직임 감지 여부에 따라 추가 텍스트 표시
        if is_large_movement:
            status_text = "MOVEMENT DETECTED!"
            color = (0, 0, 255)  # 빨간색
        else:
            status_text = ""
            color = config.TEXT_COLOR
        
        # 텍스트 그리기
        cv2.putText(frame, movement_text, 
                   config.TEXT_POSITION, 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   config.FONT_SCALE, 
                   config.TEXT_COLOR, 
                   config.FONT_THICKNESS)
        
        cv2.putText(frame, camera_text, 
                   (config.TEXT_POSITION[0], config.TEXT_POSITION[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   config.FONT_SCALE * 0.8, 
                   (255, 255, 0),  # 노란색 
                   config.FONT_THICKNESS)
        
        cv2.putText(frame, status_text, 
                   (config.TEXT_POSITION[0], config.TEXT_POSITION[1] + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   config.FONT_SCALE, 
                   color, 
                   config.FONT_THICKNESS)
        
        # 스켈레톤 그리기
        if keypoints is not None and len(keypoints) > 0:
            # COCO 17 키포인트 연결 순서
            # 각 숫자는 키포인트 인덱스를 나타냅니다.
            # (예: 0: 코, 1: 왼쪽 눈, 2: 오른쪽 눈, ..., 16: 오른쪽 발목)
            skeleton_connections = [
                # 머리
                (0, 1), (0, 2), (1, 3), (2, 4),
                # 몸통 (어깨-엉덩이)
                (5, 6), (5, 11), (6, 12), (11, 12),
                # 왼팔
                (5, 7), (7, 9),
                # 오른팔
                (6, 8), (8, 10),
                # 왼다리
                (11, 13), (13, 15),
                # 오른다리
                (12, 14), (14, 16)
            ]
            
            keypoint_color = (0, 255, 255) # 노란색 점
            line_color = (0, 255, 0)     # 초록색 선
            
            for i, point in enumerate(keypoints):
                x, y = int(point[0]), int(point[1])
                # keypoints 배열에는 confidence 값도 포함될 수 있으나, 여기서는 x, y만 사용
                # confidence 값은 motion_detector에서 이미 사용됨
                if x > 0 and y > 0: # 좌표가 유효한 경우에만 그림
                    cv2.circle(frame, (x, y), 5, keypoint_color, -1) # 점 그리기
            
            for (start_idx, end_idx) in skeleton_connections:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_point = tuple(map(int, keypoints[start_idx][:2]))
                    end_point = tuple(map(int, keypoints[end_idx][:2]))
                    
                    # 좌표가 유효한 경우에만 그림 (화면 밖 좌표 제외)
                    if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                        cv2.line(frame, start_point, end_point, line_color, 2) # 선 그리기
                        
        return frame