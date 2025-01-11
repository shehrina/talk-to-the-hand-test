import cv2
import math
import numpy as np
import random
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# 1) Load your model with pure TensorFlow / Keras
model = load_model("Model/keras_model.h5")

# 2) Load labels from text file
labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            labels.append(line)

# Random target letter
target_letter = random.choice(labels)

# Hand Detector
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 224

# Initialize webcam
cap = cv2.VideoCapture(0)

def predict_label(img):
    """
    Receives a BGR image (imgWhite), resizes to 224x224, normalizes,
    and does a model prediction, returning (prediction_probabilities, max_index).
    """
    # Make sure dimensions match what you trained on (224, 224 here).
    # If you trained on 512x512, adjust accordingly.
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    # Expand dims to make shape = (1, 224, 224, 3)
    img_resized = np.expand_dims(img_resized, axis=0)

    predictions = model.predict(img_resized, verbose=0)  # shape = (1, num_classes)
    max_index = np.argmax(predictions[0])
    return predictions[0], max_index

def gen_frames():
    global target_letter
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        recognized_letter = None

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop region
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                # Create white background
                imgWhite = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 255
                aspectRatio = h / w

                # Resize/copy into imgWhite
                if aspectRatio > 1:
                    scale = imgsize / h
                    wCal = math.ceil(scale * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                    wGap = math.ceil((imgsize - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    scale = imgsize / w
                    hCal = math.ceil(scale * h)
                    imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                    hGap = math.ceil((imgsize - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Predict using our pure TF function
                _, index = predict_label(imgWhite)
                recognized_letter = labels[index]

                # Draw bounding box
                cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)
                # Show recognized letter
                cv2.putText(imgOutput, recognized_letter, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)

        # Show "Sign the letter"
        cv2.putText(imgOutput, f"Sign the letter: {target_letter}",
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255), 2)

        # Correct or not?
        if recognized_letter == target_letter:
            status_text = "Correct!"
        else:
            status_text = "Try again"

        cv2.putText(imgOutput, status_text, (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
