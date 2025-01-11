import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# 1) Load your trained model and labels
model = load_model("Model/keras_model.h5")

# Load labels from the text file
labels = []
with open("Model/labels.txt", "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        if line.strip():
            labels.append(line.strip())

# Helper function for prediction
def getPrediction(img):
    """
    Resize 'img' to (224, 224), normalize it, and use the model to predict.
    Returns the raw prediction and the index with the highest probability.
    """
    # Model expects 224x224 if trained that way. Adjust if you used a different size.
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    predictions = model.predict(img, verbose=0)  # or remove verbose=0 if you want logs
    index = np.argmax(predictions)
    return predictions[0], index

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 224

# If you only have two classes:
# labels = ["A", "B"]

# If you have more classes, simply update the 'labels' list or use the ones loaded above.

while True:
    success, img = capture.read()
    if not success:
        break

    # Copy original to display final results
    imgOutput = img.copy()

    # Detect hands (draws bounding box and landmarks on 'img')
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image of size imgsize x imgsize
        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Crop around the bounding box with offset
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            # If hand is taller than wide
            if aspectRatio > 1:
                scale = imgsize / h
                wCal = math.ceil(scale * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                wGap = math.ceil((imgsize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize

            # If hand is wider than tall
            else:
                scale = imgsize / w
                hCal = math.ceil(scale * h)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                hGap = math.ceil((imgsize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Run the classifier (using our custom getPrediction)
            prediction, index = getPrediction(imgWhite)
            print(prediction, index)

            # Display predicted label
            cv2.putText(imgOutput, labels[index], (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

            # For visualization, draw bounding box on the main output
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

            # Debug windows
            cv2.imshow("imageCrop", imgCrop)
            cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):  # press 'q' to quit
        break

capture.release()
cv2.destroyAllWindows()
