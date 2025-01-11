import cv2
import math
import numpy as np
import random

from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# 1) Create a map from numeric to letter, if your labels.txt uses digits
label_map = {
    "0": "A",
    "1": "B",
    # "2": "C", etc. if you have more classes
}

# 2) Load the model
model = load_model("Model/keras_model.h5")

# 3) Load the labels from text file, but convert them using label_map
labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            label_converted = label_map.get(line, line)  # fallback if not in map
            labels.append(label_converted)

# Random target letter
target_letter = random.choice(labels)

recognized_letter = None
status_text = "Loading..."

# Hand Detector
detector = HandDetector(maxHands=1)

offset = 20
imgsize = 224

# Initialize webcam
cap = cv2.VideoCapture(0)

def predict_label(img, threshold=0.8):
    """
    Resizes 'img' to 224x224, normalizes, and returns (predictions, predicted_label).
    - If the highest probability is below 'threshold', we return 'Unrecognized'.
    """
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # (1, 224, 224, 3)

    predictions = model.predict(img_resized, verbose=0)  # shape = (1, num_classes)
    pred_probs = predictions[0]
    max_index = np.argmax(pred_probs)
    max_prob = pred_probs[max_index]

    if max_prob < threshold:
        return predictions[0], "Unrecognized"
    else:
        return predictions[0], labels[max_index]

def gen_frames():
    global recognized_letter, status_text, target_letter

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = img.copy()
        # 'flipType=False' so it doesn't flip the image
        hands, _ = detector.findHands(imgRGB, flipType=False)

        recognized_letter = None
        if hands:
            # We only process the first hand (maxHands=1)
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Draw bounding box on the color feed (magenta rectangle)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

            # Crop region for classification
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = imgRGB[y1:y2, x1:x2]

            if imgCrop.size != 0:
                # Create a white background
                imgWhite = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 255
                aspectRatio = h / w

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

                # Predict
                _, pred_label = predict_label(imgWhite, threshold=0.8)
                recognized_letter = pred_label

        # Decide status
        if recognized_letter == "Unrecognized":
            status_text = "Unrecognized sign"
        elif recognized_letter == target_letter:
            status_text = "Correct!"
        else:
            status_text = "Try again"

        # Convert the frame with bounding box to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def main_menu():
    return render_template('main_menu.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/test')
def test_page():
    # Make sure you have a test.html that shows camera & letter side-by-side
    return render_template('test.html', target_letter=target_letter)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def recognition_status():
    return jsonify({'status_text': status_text})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/mark_correct')
def mark_correct():
    # Show green screen or increment a counter, then redirect
    return render_template('correct_screen.html')

@app.route('/mark_wrong')
def mark_wrong():
    # Show red screen or do something else
    return render_template('wrong_screen.html')
