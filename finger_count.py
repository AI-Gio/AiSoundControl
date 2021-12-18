import cv2
import time
import os
import module_hand_detection as mhd

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderpath = "finger_images"
myList = os.listdir(folderpath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f"{folderpath}/{imPath}")
    overlayList.append(image)

p_time = 0

detector = mhd.HandDetector(detection_con=0.7)

tipIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        h, w, c = overlayList[totalFingers].shape
        cix = 10
        ciy = 10
        img[ciy:h + ciy, cix:w + cix] = overlayList[totalFingers]

    c_time = time.time()
    fps = 1 / (c_time-p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
