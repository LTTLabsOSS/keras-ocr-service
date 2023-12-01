"""Keras OCR service main script"""
from datetime import datetime
import os
import re
import time
import cv2
import numpy as np
import keras_ocr
from werkzeug.utils import secure_filename
from flask import Flask, request
from waitress import serve

pipeline = keras_ocr.pipeline.Pipeline()

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploaded")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
BOX_RGB = (0, 255, 0)
TEXT_RGB = (255, 0, 0)
FOUND_RGB = (0, 0, 255)


def text_origin(box) -> tuple[any]:
    """Gets a point for positioning text. See:
    https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    """
    return (box[0][0].astype("int32"), box[0][1].astype("int32"))


def mass_center(moments: any) -> tuple[int]:
    """Calculates the mass center from the moments of a contour.
    See: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html#details
    """
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return (center_x, center_y)


def load_image(path: str) -> np.ndarray:
    """Loads an image from file and returns it"""
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image) -> None:
    """Saves an image to disk"""
    out_path = os.path.join(OUTPUT_DIR, "attempt")
    cv2.imwrite(out_path + '_out.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def label_text(image: np.ndarray, polygon: any, box: any, text: str) -> None:
    org = text_origin(box)
    cv2.polylines(image, polygon, True, BOX_RGB, 2, )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_RGB, 1, cv2.LINE_AA)  


def find_word(word: str, image_path: str):
    """Given a target word and an image path. Searches for the word in an image.
    Will draw boxes around found text and label them with the detected word for the
    output image. Returns a dictionary with a status and the x, y coordinates of the
    center of the found word's bounding box.
    """
    image = load_image(image_path)

    # keras-ocr will automatically download pretrained weights for the detector and recognizer.
    # Each list of predictions in prediction_groups is a list of (word, box) tuples.
    pred = pipeline.recognize([image])[0]
    for text, box in pred:
        polygon = box[np.newaxis].astype("int32")
        label_text(image, polygon, box, text)

        if re.match(word, text):
            cv2.polylines(image, polygon, True, FOUND_RGB, 2, )
            for c in polygon:
                center_x, center_y = mass_center(cv2.moments(c))
                return {
                    "result": "found",
                    "x": center_x,
                    "y": center_y
                }

    # for debug save the latest image that didn't find the target word
    save_image(image)

    return {
        "result": "not found"
    }


app = Flask(__name__, instance_relative_config=True)
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")
logFile = f"{LOG_DIR}/{formatted_time}_ocr.csv"
count = 0

@app.route("/process", methods=['POST'])
def process():
    global count
    global logFile
    """Main API endpoint"""
    if request.method == 'POST':
        count += 1
        file = request.files['file']
        word = request.form['word']
        file_name = secure_filename(file.filename)
        current_time_millis = int(time.time() * 1000)
        newFileName = str(current_time_millis) + "-" + file_name
        file_path = os.path.join(UPLOAD_DIR, newFileName)
        file.save(file_path)
        with open(logFile, 'a') as log:
            result = find_word(word, file_path)
            t2 = int(time.time() * 1000)
            duration = str(round(t2 - current_time_millis))
            print(f"ocr duration: {duration} ms")
            log.write(f"{current_time_millis},{str(count)},{duration}\n")
        os.remove(file_path)
        return result

if __name__ == '__main__':    
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(logFile, 'a') as log:
            log.write(f"timestamp, count, duration (ms)\n")
    except OSError as err:
        print(err)

    # Uncomment below for debug server
    #app.run()
    serve(app, host='0.0.0.0', port=5001)
