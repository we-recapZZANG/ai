# utils.py - 유틸리티 함수들

import numpy as np
from scipy.spatial.distance import euclidean
import cv2

def calculate_keypoints_distance(prev_keypoints, curr_keypoints):
    """
    이전 프레임과 현재 프레임의 키포인트 간 유클리드 거리 계산
    
    Args:
        prev_keypoints: 이전 프레임의 키포인트 [(x1, y1, conf1), ...]
        curr_keypoints: 현재 프레임의 키포인트 [(x1, y1, conf1), ...]
    
    Returns:
        float: 전체 키포인트의 총 이동 거리
    """
    if prev_keypoints is None or curr_keypoints is None:
        return 0.0
    
    total_distance = 0.0
    num_keypoints = min(len(prev_keypoints), len(curr_keypoints))
    
    for i in range(num_keypoints):
        prev_x, prev_y, prev_conf = prev_keypoints[i]
        curr_x, curr_y, curr_conf = curr_keypoints[i]
        
        # 키포인트가 감지되지 않은 경우 (신뢰도가 낮은 경우)
        if prev_conf < 0.5 or curr_conf < 0.5:
            # 이전 위치와 동일하다고 가정 (거리 = 0)
            distance = 0.0
        else:
            # 유클리드 거리 계산
            distance = euclidean([prev_x, prev_y], [curr_x, curr_y])
        
        total_distance += distance
    
    return total_distance

def draw_movement_status(frame, is_moving, movement_distance):
    """
    프레임에 움직임 상태를 그림
    
    Args:
        frame: 현재 프레임
        is_moving: 움직임 감지 여부
        movement_distance: 현재 프레임의 움직임 거리
    
    Returns:
        numpy.ndarray: 텍스트가 추가된 프레임
    """
    text_color = (0, 255, 0) if not is_moving else (0, 0, 255)
    status_text = "MOVING" if is_moving else "STILL"
    
    # 상태 표시
    cv2.putText(frame, f"Status: {status_text}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    
    # 움직임 거리 표시
    cv2.putText(frame, f"Distance: {movement_distance:.2f}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def save_video(frames, output_path, fps=30):
    """
    프레임들을 비디오로 저장
    
    Args:
        frames: 프레임 리스트
        output_path: 출력 경로
        fps: 프레임 레이트
    """
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()