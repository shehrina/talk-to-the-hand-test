import cv2
import math
import numpy as np
import random

from flask import Flask, render_template, Response, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# -------------------------------------------------------
# 1) Model and Label Setup
# -------------------------------------------------------
label_map = {
    "0": "A",
    "1": "B",
    "2": "C",
    "3": "D",
    "4": "E",
    "5": "F",
    "6": "G",
    "7": "H",
    "8": "I",
    "9": "J",
    "10": "1",
    "11": "2",
    "12": "3",
    "13": "4",
    "14": "5"
}

model = load_model("Model/keras_model.h5")

labels = []
with open("Model/labels.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            label_converted = label_map.get(line, line)
            labels.append(label_converted)

# -------------------------------------------------------
# 2) Game Variables
# -------------------------------------------------------
TOTAL_ROUNDS = 10
attempt_count = 0
correct_count = 0

target_letter = None
recognized_letter = None
status_text = "Loading..."

# -------------------------------------------------------
# 3) Hand Detector and Webcam
# -------------------------------------------------------
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 224

cap = cv2.VideoCapture(0)
# Lower resolution to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Attempt to set FPS (some webcams ignore this)
cap.set(cv2.CAP_PROP_FPS, 30)

# We’ll skip frames to reduce load
FRAME_SKIP = 2
frame_counter = 0

# -------------------------------------------------------
# 4) Utility Functions
# -------------------------------------------------------
def reset_game():
    global attempt_count, correct_count
    attempt_count = 0
    correct_count = 0

def pick_new_letter():
    return random.choice(labels)

def predict_label(img, threshold=0.8):
    """
    Run inference on the image, return 'Unrecognized' if below threshold.
    """
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    predictions = model.predict(img_resized, verbose=0)
    pred_probs = predictions[0]
    max_index = np.argmax(pred_probs)
    max_prob = pred_probs[max_index]

    if max_prob < threshold:
        return predictions[0], "Unrecognized"
    else:
        return predictions[0], labels[max_index]

# -------------------------------------------------------
# 5) Frame Generator for MJPEG
# -------------------------------------------------------
def gen_frames():
    global recognized_letter, status_text, target_letter
    global frame_counter

    while True:
        success, img = cap.read()
        if not success:
            break

        # If skipping frames, only do detection/inference every FRAME_SKIP frames
        frame_counter += 1
        do_inference = (frame_counter % FRAME_SKIP == 0)

        if do_inference:
            # Make a copy for detection (hand tracking can flip or process differently)
            imgRGB = img.copy()
            hands, _ = detector.findHands(imgRGB, flipType=False)

            recognized_letter = None
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

                # Crop and prepare for classification
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                imgCrop = imgRGB[y1:y2, x1:x2]

                if imgCrop.size != 0:
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

            # Update status
            if recognized_letter == "Unrecognized":
                status_text = "Unrecognized sign"
            elif recognized_letter == target_letter:
                status_text = "Correct!"
            else:
                status_text = "Try again"

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------------------------------
# 6) Flask Routes
# -------------------------------------------------------
@app.route('/')
def main_menu():
    reset_game()
    return render_template('main_menu.html')

@app.route('/test')
def test_page():
    global attempt_count, correct_count, target_letter

    if attempt_count >= TOTAL_ROUNDS:
        return redirect(url_for('final_score'))

    target_letter = pick_new_letter()
    return render_template('test.html', target_letter=target_letter)

@app.route('/mark_correct')
def mark_correct():
    global attempt_count, correct_count
    attempt_count += 1
    correct_count += 1

    if attempt_count >= TOTAL_ROUNDS:
        return redirect(url_for('final_score'))
    return render_template('correct_screen.html')

@app.route('/mark_wrong')
def mark_wrong():
    global attempt_count
    attempt_count += 1

    if attempt_count >= TOTAL_ROUNDS:
        return redirect(url_for('final_score'))
    return render_template('wrong_screen.html')

@app.route('/final_score')
def final_score():
    return render_template('final_screen.html', score=correct_count, total=TOTAL_ROUNDS)

@app.route('/play_again')
def play_again():
    reset_game()
    return redirect(url_for('test_page'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def recognition_status():
    return jsonify({'status_text': status_text})

if __name__ == '__main__':
    app.run(debug=True)
