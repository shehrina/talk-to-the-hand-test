import cv2
import math
import numpy as np
import random
import os

from flask import Flask, render_template, Response, jsonify, redirect, url_for, send_from_directory, request
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
# 2) Game Variables (Test Mode)
# -------------------------------------------------------
TOTAL_ROUNDS = 10
attempt_count = 0
correct_count = 0

target_letter = None   # Used in Test mode
recognized_letter = None
status_text = "Loading..."

# -------------------------------------------------------
# 3) Train Mode Variables
# -------------------------------------------------------
train_letter = None     # Current letter in Train mode
train_status = "Loading..."
# We'll color the bounding box red if wrong, green if correct
train_border_color = (0, 0, 255)  # default red

# -------------------------------------------------------
# 4) Hand Detector and Webcam
# -------------------------------------------------------
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 224

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

FRAME_SKIP = 2
frame_counter = 0

# -------------------------------------------------------
# 5) Utility Functions
# -------------------------------------------------------
def reset_game():
    global attempt_count, correct_count
    attempt_count = 0
    correct_count = 0

def pick_new_letter():
    return random.choice(labels)

def predict_label(img, threshold=0.8):
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
# 6) Frame Generator for Test Mode (Unchanged)
# -------------------------------------------------------
def gen_frames():
    global recognized_letter, status_text, target_letter
    global frame_counter

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_counter += 1
        do_inference = (frame_counter % FRAME_SKIP == 0)

        if do_inference:
            imgRGB = img.copy()
            hands, _ = detector.findHands(imgRGB, flipType=False)

            recognized_letter = None
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

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
                        hCal = math.ceil(scale * w)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    _, pred_label = predict_label(imgWhite, threshold=0.8)
                    recognized_letter = pred_label

            if recognized_letter == "Unrecognized":
                status_text = "Unrecognized sign"
            elif recognized_letter == target_letter:
                status_text = "Correct!"
            else:
                status_text = "Try again"

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------------------------------
# 7) Frame Generator for Train Mode (No Timer)
# -------------------------------------------------------
def gen_train_frames():
    """
    This mode:
      - No timer
      - Shows the letter as an image (via train.html)
      - Red bounding box if incorrect, Green if correct
      - If correct, pick a new letter immediately
    """
    global train_letter, train_border_color
    global frame_counter

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_counter += 1
        do_inference = (frame_counter % FRAME_SKIP == 0)

        # Default color is red
        border_color = (0, 0, 255)

        if do_inference:
            imgRGB = img.copy()
            hands, _ = detector.findHands(imgRGB, flipType=False)

            recognized_sign = None
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

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
                        hCal = math.ceil(scale * w)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        hGap = math.ceil((imgsize - hCal) / 2)
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    _, pred_label = predict_label(imgWhite, threshold=0.8)
                    recognized_sign = pred_label

                # Decide bounding box color
                if recognized_sign == train_letter:
                    border_color = (0, 255, 0)  # green
                    # Immediately pick new letter
                    train_letter = pick_new_letter()
                else:
                    border_color = (0, 0, 255)  # red

            # Draw bounding box if we found a hand
            if hands:
                x, y, w, h = hands[0]['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 4)

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------------------------------
# 8) Serve Images from "Images/" folder
# -------------------------------------------------------
@app.route('/Images/<path:filename>')
def custom_images(filename):
    """
    Serves images from the 'Images' folder.
    Example: /Images/A.png => Images/A.png
    """
    return send_from_directory('Images', filename)


# -------------------------------------------------------
# 9) FLASK ROUTES (TEST MODE) - Unchanged
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


# -------------------------------------------------------
# 10) TRAIN MODE ROUTES
# -------------------------------------------------------
@app.route('/train')
def train_page():
    """
    Train page:
      - Sets a random letter
      - Renders train.html
    """
    global train_letter
    train_letter = pick_new_letter()  # pick an initial letter
    return render_template('train.html')

@app.route('/train_letter')
def train_letter_api():
    """
    Returns the current train_letter in JSON form,
    so the front-end can update the displayed letter image every second.
    """
    global train_letter
    return jsonify({'letter': train_letter})

@app.route('/train_video_feed')
def train_video_feed():
    """
    MJPEG feed specifically for Train mode, uses gen_train_frames().
    """
    return Response(gen_train_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------------------------------
# 11) "Guess The Sign" Feature
# -------------------------------------------------------
# NEW: "Guess The Sign" Variables
guess_sign_letter = None

@app.route('/guess_sign')
def guess_sign():
    """
    Renders the Guess The Sign page and picks a new random letter.
    """
    global guess_sign_letter
    guess_sign_letter = pick_new_letter()  # reuse your pick_new_letter() function
    return render_template('guess_sign.html', guess_sign_letter=guess_sign_letter)

@app.route('/guess_sign_check', methods=['POST'])
def guess_sign_check():
    """
    Handles form submission from the guess_sign.html page.
    Compares user guess with the current letter (in uppercase).
    """
    global guess_sign_letter
    user_guess = request.form.get('guess')
    if not user_guess:
        return redirect(url_for('guess_sign'))  # no guess provided, just reload

    # Convert guess to uppercase
    user_guess = user_guess.strip().upper()

    if user_guess == guess_sign_letter:
        # Redirect to "Guess The Sign" Correct screen
        return redirect(url_for('guess_sign_correct'))
    else:
        # Redirect to "Guess The Sign" Wrong screen
        return redirect(url_for('guess_sign_wrong'))

@app.route('/guess_sign_correct')
def guess_sign_correct():
    """
    Shows the "Correct!" screen for Guess The Sign,
    then after 1 second, goes back to /guess_sign.
    """
    return render_template('guess_sign_correct.html')

@app.route('/guess_sign_wrong')
def guess_sign_wrong():
    """
    Shows the "Wrong!" screen for Guess The Sign,
    then after 1 second, goes back to /guess_sign.
    """
    return render_template('guess_sign_wrong.html')


# -------------------------------------------------------
# 12) MAIN
# -------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
