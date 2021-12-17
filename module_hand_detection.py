import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=1, model_comp=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.modelComp = model_comp
        self.detectionCon = detection_con
        self.trackCon = track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp,
                                        self.detectionCon, self.trackCon)  #can change params of Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Find hands from the image of camera.

        :param img: the image from VideoCapture
        :param draw: bool, whether to draw the landmarks and connections of the hand(s)
        :return: img
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):
        """
        Finds the position of the hand and is able to draw on the img.

        :param img: the image from VideoCapture
        :param hand_no: which hand should be chosen to draw on, 0 refers to the hand that is last detected
        :param draw: bool, whether to draw circle on the hands
        :return: a list of all of the landmarks
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark): # id stands for hand node
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    """
    Run hand detection/tracking with fps on screen.
    """
    p_time = 0
    c_time = 0

    cap = cv2.VideoCapture(0)  # webcam
    detector = HandDetector()

    while True:
        succes, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)  # remove False to display circles on landmarks
        if len(lm_list) != 0:  # if there is no landmarks (so no hands in webcam)
            print(lm_list[4])  # show the

        # FPS calculate
        c_time = time.time()
        fps = 1 / (c_time - p_time)

        p_time = c_time
        # FPS show
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()