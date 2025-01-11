import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 224

folder = "Data/A"  # <-- change this to the folder/class you want to collect
counter = 0

while True:
    success, img = capture.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]  # the one hand we have
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Crop around the bounding box with offset
        # Make sure x-offset and y-offset don't go below 0
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            # If the crop is empty (hand partially out of frame?), skip
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wCal = math.floor(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            wGap = math.ceil((imgsize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgsize / w
            hCal = math.floor(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # For visualization
        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Only save when 's' is pressed and we have a valid white image
        if 'imgWhite' in locals():
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Saved image #{counter}")
        else:
            print("No image to save")
    elif key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
