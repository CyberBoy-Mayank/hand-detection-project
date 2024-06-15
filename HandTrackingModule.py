import cv2
import mediapipe as mp


class handDetector:
    def __init__(self, mode=False, maxHands=4, complexity=0, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5,
                                                                       circle_radius=0),
                                               self.mpDraw.DrawingSpec(color=(57, 255, 20), thickness=2,
                                                                       circle_radius=2))
        return img

    def findPosition(self, img, handNum=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    # cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)
        return lmList
