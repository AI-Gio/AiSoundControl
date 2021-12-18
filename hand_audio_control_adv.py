import cv2
import time
import numpy as np
import module_hand_detection as mhd
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
# print(vol_range)
min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
volBar = 400
volPer = 0
area = 0
color_vol = (0, 255, 0)

while True:
    success, img = cap.read()

    # Find Hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:

        # Filter based on size
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 250 < area < 1000:

            # Find Distance between index and thumb
            length, img, line_info = detector.findDistance(4, 8, img)

            # Convert Volume
            # hand range 30 - 200, volRange = (-96.0, 0.0, 0.125)
            volBar = np.interp(length, [20, 200], [400, 150])
            volPer = np.interp(length, [20, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer/smoothness)

            # Check fingers up
            fingers = detector.fingersUp()

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)  # to have true linear distribution of volume
                cv2.circle(img, (line_info[4], line_info[5]), 8, (0, 0, 255), cv2.FILLED)
                color_vol = (0, 0, 255)
            else:
                color_vol = (0, 255, 0)

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f"Vol: {int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)
    curr_vol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, f"Vol set: {int(curr_vol)} %", (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, color_vol, 3)

    # Frame rate
    c_time = time.time()
    fps = 1 / (c_time-p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)  # 1ms delay
