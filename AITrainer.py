import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("assets/curls.mp4")
img = cv2.imread("assets/test2.jpeg")

detector = pm.poseDetector()


while True:
    # success, img = cap.read()
    # img = cv2.resize(img, (1280, 720))
    
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)

    if len(lmList) != 0:
        detector.findAngle(img, 12, 14, 16)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
