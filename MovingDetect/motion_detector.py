# motion_detector.py

import numpy as np
from collections import deque
from ultralytics import YOLO
import cv2
import config

class MotionDetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL)
        self.previous_keypoints = None
        self.movement_history = deque(maxlen=config.SMOOTHING_WINDOW)
        self.previous_frame = None
        self.camera_motion_history = deque(maxlen=config.SMOOTHING_WINDOW)
        
    def calculate_euclidean_distance(self, points1, points2):
        """두 점 집합 간의 유클리드 거리를 계산합니다."""
        if points1 is None or points2 is None:
            return 0.0
        
        # 같은 shape로 맞추기
        if points1.shape != points2.shape:
            return 0.0
            
        distances = np.sqrt(np.sum((points1 - points2) ** 2, axis=1))
        return np.sum(distances)
    
    def estimate_camera_motion(self, current_frame):
        """카메라 움직임을 추정합니다."""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return 0.0, 0.0
        
        # 특징점 검출
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # ORB 특징점 검출기 사용
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray_previous, None)
        kp2, des2 = orb.detectAndCompute(gray_current, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            self.previous_frame = current_frame.copy()
            return 0.0, 0.0
        
        # 특징점 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 좋은 매칭점들의 움직임 계산
        if len(matches) < 10:
            self.previous_frame = current_frame.copy()
            return 0.0, 0.0
        
        # 상위 매칭점들의 움직임 벡터 계산
        good_matches = matches[:min(50, len(matches))]
        motion_vectors = []
        
        for match in good_matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            motion_vectors.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))
        
        # 평균 카메라 움직임 계산
        motion_vectors = np.array(motion_vectors)
        median_motion_x = np.median(motion_vectors[:, 0])
        median_motion_y = np.median(motion_vectors[:, 1])
        
        self.previous_frame = current_frame.copy()
        return median_motion_x, median_motion_y
    
    def compensate_camera_motion(self, keypoints, camera_motion):
        """카메라 움직임을 보정합니다."""
        if keypoints is None or camera_motion is None:
            return keypoints
        
        compensated_keypoints = keypoints.copy()
        compensated_keypoints[:, 0] -= camera_motion[0]  # x 좌표 보정
        compensated_keypoints[:, 1] -= camera_motion[1]  # y 좌표 보정
        
        return compensated_keypoints
    
    def detect_movement(self, frame):
        """프레임에서 움직임을 감지합니다 (카메라 움직임 보정 포함)."""
        # 카메라 움직임 추정
        camera_motion = self.estimate_camera_motion(frame)
        
        # YOLO로 포즈 추정
        results = self.model(frame, verbose=False)
        
        if len(results) == 0 or len(results[0].keypoints) == 0:
            return False, 0.0, camera_motion, None
        
        # 첫 번째 사람의 키포인트만 사용
        current_keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        # keypoints가 없으면 False 반환
        if len(current_keypoints) == 0:
            return False, 0.0, camera_motion, None
        
        # x, y 좌표만 추출
        current_points = current_keypoints[:, :2]
        
        # confidence가 낮은 관절은 이전 위치 유지
        confidences = current_keypoints[:, 2]
        if self.previous_keypoints is not None:
            for i, conf in enumerate(confidences):
                if conf < config.CONFIDENCE_THRESHOLD:
                    current_points[i] = self.previous_keypoints[i]
        
        # 카메라 움직임 보정
        compensated_points = self.compensate_camera_motion(current_points, camera_motion)
        
        # 이전 프레임이 없으면 현재 프레임을 저장하고 False 반환
        if self.previous_keypoints is None:
            self.previous_keypoints = compensated_points.copy()
            return False, 0.0, camera_motion, current_points
        
        # 이전 키포인트도 현재 카메라 움직임만큼 보정
        compensated_previous = self.compensate_camera_motion(self.previous_keypoints, camera_motion)
        
        # 움직임 계산 (보정된 좌표 사용)
        movement_distance = self.calculate_euclidean_distance(compensated_points, compensated_previous)
        
        # 상대적 움직임 계산 (신체 크기 대비)
        body_size = self.estimate_body_size(current_points)
        relative_movement = movement_distance / max(body_size, 1.0)
        
        # 움직임 이력에 추가
        self.movement_history.append(relative_movement)
        
        # 평균 움직임 계산
        average_movement = np.mean(self.movement_history)
        
        # 현재 키포인트를 이전 키포인트로 업데이트
        self.previous_keypoints = current_points.copy()
        
        # 큰 움직임 감지 (상대적 움직임 기준)
        is_large_movement = average_movement > config.RELATIVE_MOVEMENT_THRESHOLD
        
        return is_large_movement, average_movement, camera_motion, current_points
    
    def estimate_body_size(self, keypoints):
        """신체 크기를 추정합니다."""
        if keypoints is None or len(keypoints) < 5:
            return 100.0  # 기본값
        
        # 주요 관절 쌍의 거리를 이용해 신체 크기 추정
        # 예: 어깨 너비, 몸통 길이 등
        distances = []
        
        # 어깨 너비 (좌우 어깨)
        if len(keypoints) > 6:
            shoulder_width = np.linalg.norm(keypoints[5] - keypoints[6])
            distances.append(shoulder_width)
        
        # 몸통 길이 (목에서 엉덩이)
        if len(keypoints) > 11:
            torso_length = np.linalg.norm(keypoints[0] - keypoints[11])
            distances.append(torso_length)
        
        if len(distances) > 0:
            return np.mean(distances)
        else:
            return 100.0  # 기본값