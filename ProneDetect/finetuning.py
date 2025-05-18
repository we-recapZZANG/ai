import os
from ultralytics import YOLO

current_dir = os.getcwd()
yaml_path = os.path.join(current_dir,'babyData','PoseData.yaml')

yaml_path =os.path.abspath(yaml_path)

model = YOLO('yolov8n.pt')
model.train(data=yaml_path, epochs=300, patience=30, batch=32)


print(type(model.names), len(model.names))

print(model.names)