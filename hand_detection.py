import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

model = tf.keras.models.load_model("hand_gesture.h5")
labels = ["A", "B", "BAD", "C", "COMMENT", "GOOD", "ILY", "NAMASTE", "NO", "YES"]


displayTextMap = {
    "A": "A",
    "B": "B",
    "BAD": "BAD",
    "C": "C",
    "COMMENT": "Please comment your reviews",
    "GOOD": "GOOD",
    "ILY": "I Love You",
    "NAMASTE": "Namaste Ji",
    "NO": "NO",
    "YES": "YES"
}


camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 15
imgsize = 300

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

        if imgCrop.size != 0:
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

            imgInput = imgWhite / 255.0
            imgInput = np.expand_dims(imgInput, axis=0)

            prediction = model.predict(imgInput)
            index = np.argmax(prediction)
            label = labels[index]
            displayText = displayTextMap[label]

            cv2.putText(frame, displayText, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()



