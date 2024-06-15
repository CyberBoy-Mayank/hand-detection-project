import HandTrackingModule as htm
import cv2
from time import time


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        success, img = cap.read()
        img = detector.findHands(img=img, draw=False)
        detector.findPosition(img)
        cTime = time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
