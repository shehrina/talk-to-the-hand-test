import cv2
import math
import numpy as np
import random
import os
import logging

from flask import Flask, render_template, Response, jsonify, redirect, url_for, send_from_directory, request
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# -------------------------------------------------------
# 1) Logging Configuration
# -------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


try:
    model = load_model("Model/keras_model.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise e

import logging

labels = []
try:
    with open("Model/labels.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    logging.info(f"Labels loaded: {labels}")
except Exception as e:
    logging.error(f"Error loading labels: {e}")
    raise e


# -------------------------------------------------------
# 3) Game Variables (Test Mode)
# -------------------------------------------------------
TOTAL_ROUNDS = 10
attempt_count = 0
correct_count = 0

target_letter = None  # Used in Test mode
recognized_letter = None
status_text = "Loading..."

# -------------------------------------------------------
# 4) Train Mode Variables
# -------------------------------------------------------
train_letter = None  # Current letter in Train mode
train_status = "Loading..."
# We'll color the bounding box red if wrong, green if correct
train_border_color = (0, 0, 255)  # default red

# -------------------------------------------------------
# 5) "Guess The Sign" Feature Variables
# -------------------------------------------------------
guess_sign_letter = None

# -------------------------------------------------------
# 6) Hand Detector and Webcam
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
# 7) Utility Functions
# -------------------------------------------------------
def reset_game():
    global attempt_count, correct_count
    attempt_count = 0
    correct_count = 0
    logging.debug("Game reset.")


def pick_new_letter():
    new_letter = random.choice(labels)
    logging.debug(f"New letter picked: {new_letter}")
    return new_letter


def predict_label(img, threshold=0.8):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_resized = img_resized.astype('float32') / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        predictions = model.predict(img_resized, verbose=0)
        pred_probs = predictions[0]
        max_index = np.argmax(pred_probs)
        max_prob = pred_probs[max_index]

        if max_prob < threshold:
            recognized = "Unrecognized"
        else:
            recognized = labels[max_index]

        logging.debug(f"Predicted Label: {recognized} with probability {max_prob:.2f}")
        return predictions[0], recognized
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, "Unrecognized"


# -------------------------------------------------------
# 8) Frame Generator for Test Mode
# -------------------------------------------------------
def gen_frames():
    global recognized_letter, status_text, target_letter
    global frame_counter

    while True:
        success, img = cap.read()
        if not success:
            logging.warning("Failed to read frame from webcam.")
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
                # Automatically mark as correct and pick new letter
                attempt_count += 1
                correct_count += 1
                logging.info(f"Correct! Total correct: {correct_count}/{TOTAL_ROUNDS}")
                if attempt_count >= TOTAL_ROUNDS:
                    # End the game
                    status_text = "Game Over"
                else:
                    target_letter = pick_new_letter()
            else:
                status_text = "Try again"

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            logging.warning("Failed to encode frame.")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------------------------------
# 9) Frame Generator for Train Mode
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
            logging.warning("Failed to read frame from webcam.")
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
                    logging.info(f"Train Mode: Correct sign recognized: {recognized_sign}")
                    # Immediately pick new letter
                    train_letter = pick_new_letter()
                else:
                    border_color = (0, 0, 255)  # red
                    if recognized_sign != "Unrecognized":
                        logging.info(f"Train Mode: Incorrect sign recognized: {recognized_sign}")

            # Draw bounding box if we found a hand
            if hands:
                x, y, w, h = hands[0]['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 4)

        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            logging.warning("Failed to encode frame.")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------------------------------
# 10) Serve Images from "Images/" folder
# -------------------------------------------------------
@app.route('/Images/<path:filename>')
def custom_images(filename):
    """
    Serves images from the 'Images' folder.
    Example: /Images/A.png => Images/A.png
    """
    try:
        return send_from_directory('Images', filename)
    except Exception as e:
        logging.error(f"Error serving image {filename}: {e}")
        return "Image not found", 404


# -------------------------------------------------------
# 11) FLASK ROUTES (TEST MODE)
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

    if target_letter is None:
        target_letter = pick_new_letter()

    logging.debug(f"Test Mode: Target letter is {target_letter}")
    return render_template('test.html', target_letter=target_letter)


@app.route('/mark_correct')
def mark_correct():
    global attempt_count, correct_count, target_letter
    attempt_count += 1
    correct_count += 1
    logging.info(f"Marked Correct: {correct_count}/{TOTAL_ROUNDS}")

    if attempt_count >= TOTAL_ROUNDS:
        return redirect(url_for('final_score'))

    target_letter = pick_new_letter()
    return render_template('correct_screen.html', next_letter=target_letter)


@app.route('/mark_wrong')
def mark_wrong():
    global attempt_count, target_letter
    attempt_count += 1
    logging.info(f"Marked Wrong: Attempts {attempt_count}/{TOTAL_ROUNDS}")

    if attempt_count >= TOTAL_ROUNDS:
        return redirect(url_for('final_score'))

    target_letter = pick_new_letter()
    return render_template('wrong_screen.html', next_letter=target_letter)


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
# 12) TRAIN MODE ROUTES
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
    logging.debug(f"Train Mode: Initial train letter is {train_letter}")
    return render_template('train.html', train_letter=train_letter)


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
# 13) "Guess The Sign" Feature Routes
# -------------------------------------------------------
@app.route('/guess_sign')
def guess_sign():
    """
    Renders the Guess The Sign page and picks a new random letter.
    """
    global guess_sign_letter
    guess_sign_letter = pick_new_letter()  # reuse your pick_new_letter() function
    logging.debug(f"Guess The Sign: Current letter is {guess_sign_letter}")
    return render_template('guess_sign.html', guess_sign_letter=guess_sign_letter)


@app.route('/guess_sign_check', methods=['POST'])
def guess_sign_check():
    """
    Handles form submission from the guess_sign.html page.
    Compares user guess with the current letter (handles both letters and numbers).
    """
    global guess_sign_letter
    user_guess = request.form.get('guess')
    if not user_guess:
        logging.warning("Guess The Sign: No guess provided.")
        return redirect(url_for('guess_sign'))  # no guess provided, just reload

    user_guess = user_guess.strip()

    # Determine if the guess is a letter or number and normalize accordingly
    if user_guess.isalpha():
        user_guess = user_guess.upper()
    else:
        user_guess = user_guess  # Keep numbers as they are

    logging.debug(f"Guess The Sign: User guessed '{user_guess}' vs actual '{guess_sign_letter}'")

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
# 14) Error Handlers
# -------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    logging.error(f"404 Error: {e}")
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    logging.error(f"500 Error: {e}")
    return render_template('500.html'), 500


# -------------------------------------------------------
# 15) MAIN
# -------------------------------------------------------
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
