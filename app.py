import cv2
import math
import numpy as np
import random

from flask import Flask, render_template, Response, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# 1) Map numeric labels to actual letters
label_map = {
    "0": "A",
    "1": "B",
}

# 2) Load your model
model = load_model("Model/keras_model.h5")

# 3) Load labels from text file and convert them
labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            label_converted = label_map.get(line, line)
            labels.append(label_converted)

# -------------------------------
# GAME/TEST CONFIG
# -------------------------------
TOTAL_ROUNDS = 10
attempt_count = 0
correct_count = 0

target_letter = None
recognized_letter = None
status_text = "Loading..."

# Hand Detector
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 224

# Webcam
cap = cv2.VideoCapture(0)

def pick_new_letter():
    """Pick a random letter from 'labels' for the next round."""
    return random.choice(labels)

def predict_label(img, threshold=0.8):
    """
    Resizes 'img' to 224x224, normalizes, and returns (predictions, predicted_label).
    If the highest probability is below 'threshold', we label it 'Unrecognized'.
    """
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)  # shape (1, 224, 224, 3)

    predictions = model.predict(img_resized, verbose=0)
    pred_probs = predictions[0]
    max_index = np.argmax(pred_probs)
    max_prob = pred_probs[max_index]

    if max_prob < threshold:
        return predictions[0], "Unrecognized"
    else:
        return predictions[0], labels[max_index]

def gen_frames():
    """
    Feeds frames from the webcam, draws a bounding box around the hand (if present),
    and updates recognized_letter and status_text accordingly.
    """
    global recognized_letter, status_text, target_letter
    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = img.copy()
        # 'flipType=False' so the feed isn't flipped
        hands, _ = detector.findHands(imgRGB, flipType=False)

        recognized_letter = None
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Draw bounding box (magenta)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

            # Crop
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = imgRGB[y1:y2, x1:x2]

            if imgCrop.size != 0:
                # White background
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

        # Update status_text
        if recognized_letter == "Unrecognized":
            status_text = "Unrecognized sign"
        elif recognized_letter == target_letter:
            status_text = "Correct!"
        else:
            status_text = "Try again"

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def main_menu():
    return render_template('main_menu.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/test')
def test_page():
    """ Show the test page with the camera feed, target letter, etc. """
    global target_letter
    # Pick a new letter each time user visits /test (if you only have 2 letters)
    target_letter = pick_new_letter()
    return render_template('test.html', target_letter=target_letter)

@app.route('/video_feed')
def video_feed():
    """ Streams frames for the <img> in test.html """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def recognition_status():
    """ Polled by JS for real-time status_text """
    return jsonify({'status_text': status_text})

# -------------- Correct/Wrong --------------
@app.route('/mark_correct')
def mark_correct():
    """
    Show green screen (correct_screen.html) for 1s, then redirect to /test
    """
    return render_template('correct_screen.html')

@app.route('/mark_wrong')
def mark_wrong():
    """
    Show red screen (wrong_screen.html) for 1s, then redirect to /test
    """
    return render_template('wrong_screen.html')

# -------------- Final Score --------------
@app.route('/final_score')
def final_score():
    """
    Show final_score.html, if you want a scoreboard you can pass it via template.
    But here we'll just show "0/0" for demonstration, or adapt as needed.
    """
    # If you want a real scoreboard, you'll need to track attempt_count/correct_count
    # but here's a placeholder
    return render_template('final_screen.html', score=0, total=2)

if __name__ == '__main__':
    app.run(debug=True)
