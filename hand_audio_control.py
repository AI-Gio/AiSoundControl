import cv2
import time
import numpy as np
import module_hand_detection as mhd
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

detector = mhd.HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
print(vol_range)
min_vol = vol_range[0]
max_vol = vol_range[1]


while True:
    succes, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 8, (0,255,0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        # hand range 30 - 200, (-96.0, 0.0, 0.125)
        vol = np.interp(length, [10, 250], [min_vol+20, max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time-p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1) # 1ms delay

