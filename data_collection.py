import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 15
imgsize = 300
folder = "data/Comment"
counter = 0

while True:
    ret, frame = camera.read()
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        h_img, w_img, _ = frame.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(w_img, x + w + offset)
        y2 = min(h_img, y + h + offset)

        imgCrop = frame[y1:y2, x1:x2]
        hCrop, wCrop, _ = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            wGap = math.ceil((imgsize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImgaeCrop", imgCrop)
        cv2.imshow("ImgaeWhite", imgWhite)

    cv2.imshow("IMAGE", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
