import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # can change params of Hands()
mpDraw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): # id stands for hand node
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS calculate
    c_time = time.time()
    fps = 1 / (c_time-p_time)
    p_time = c_time

    # FPS show
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

