import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

for i in range(60):
    ret, frame = cap.read()
    cv2.imwrite(f'images/image{i}.png', frame)
    time.sleep(1)
    print(f"{i} is done")

print("Done All")
