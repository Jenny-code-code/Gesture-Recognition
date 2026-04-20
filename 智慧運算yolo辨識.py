import cv2
import sys
import torch

camera_index = 0
device = torch.device("cuda")
window_name = 'YOLO Real-time Detection'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)
cap = cv2.VideoCapture(camera_index)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()


    result = model(frame)
    result.print()  
    result.save(save_dir='./runs', exist_ok=True)

    cv2.imshow(window_name, result.ims[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
