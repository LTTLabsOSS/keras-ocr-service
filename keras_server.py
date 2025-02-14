"""Keras OCR service main script"""
from datetime import datetime
import io
import os
import re
import sys
import time
import cv2
import numpy as np
import keras_ocr
from werkzeug.utils import secure_filename
from flask import Flask, make_response, request
from pathlib import Path
from waitress import serve
import logging

pipeline = keras_ocr.pipeline.Pipeline()

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = None
UPLOAD_DIR = None

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


def save_image(image) -> None:
    """Saves an image to disk"""
    out_path = os.path.join(OUTPUT_DIR, "attempt")
    cv2.imwrite(out_path + '_out.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def label_text(image: np.ndarray, polygon: any, box: any, text: str) -> None:
    org = text_origin(box)
    cv2.polylines(image, polygon, True, BOX_RGB, 2, )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                0.6, TEXT_RGB, 1, cv2.LINE_AA)


def find_word(word: str, return_image: bool, image_to_process):
    """Given a target word and an image path. Searches for the word in an image.
    Will draw boxes around found text and label them with the detected word for the
    output image. Returns a dictionary with a status and the x, y coordinates of the
    center of the found word's bounding box.
    """

    # keras-ocr will automatically download pretrained weights for the detector and recognizer.
    # Each list of predictions in prediction_groups is a list of (word, box) tuples.
    pred = pipeline.recognize([image_to_process])[0]
    for text, box in pred:
        polygon = box[np.newaxis].astype("int32")
        label_text(image_to_process, polygon, box, text)

        if re.match(word, text):
            cv2.polylines(image_to_process, polygon, True, FOUND_RGB, 2, )
            for c in polygon:
                center_x, center_y = mass_center(cv2.moments(c))
                return {
                    "result": "found",
                    "x": center_x,
                    "y": center_y
                }

    # for debug save the latest image that didn't find the target word
    save_image(image_to_process)

    return {
        "result": "not found"
    }


FLASK_APP = Flask(__name__, instance_relative_config=True)


def setup_logging(log_dir: str) -> None:
    """setup logging configuration"""
    logging_format = '%(asctime)s %(levelname)-s %(message)s'
    file_name = log_dir / "keras_service.log"
    logging.basicConfig(filename=file_name,
                        format=logging_format,
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    formatter = logging.Formatter(logging_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


@FLASK_APP.route("/process", methods=['POST'])
def process():
    global count
    global logFile
    """Main API endpoint"""
    if request.method == 'POST':
        file = request.files['file'].read()
        file_bytes = np.fromstring(file, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        process_me = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        word = request.form['word']
        current_time_millis = int(time.time() * 1000)
        result = find_word(word, False, process_me)
        t2 = int(time.time() * 1000)
        duration = round(t2 - current_time_millis)
        logging.info("ocr duration: %d ms", duration)
        return result


@FLASK_APP.route("/test_image", methods=['POST'])
def test_image():
    global count
    global logFile
    """return an image with the words found on it"""
    if request.method == 'POST':
        file = request.files['file'].read()
        file_bytes = np.fromstring(file, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        image_to_process = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        word = request.form['word']
        pred = pipeline.recognize([image_to_process])[0]
        for text, box in pred:
            polygon = box[np.newaxis].astype("int32")
            label_text(image_to_process, polygon, box, text)

        if re.match(word, text):
            cv2.polylines(image_to_process, polygon, True, FOUND_RGB, 2, )

        response_image = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode('.jpg', response_image)
        response = make_response(encoded_image.tobytes())
        response.headers['Content-Type'] = "image/jpeg"
        response.headers['Content-Length'] = len(encoded_image)
        return response


def main():
    """entry point"""
    global UPLOAD_DIR
    global OUTPUT_DIR

    try:
        work_dir = os.environ.get("WORK_DIR", None)
        if work_dir is None:
            work_dir = ROOT_DIR / "work"
        else:
            work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        UPLOAD_DIR = work_dir / "uploaded"
        OUTPUT_DIR = work_dir / "output"
        log_dir = ROOT_DIR / "logs"

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        Path(FLASK_APP.instance_path).mkdir(exist_ok=True)

        setup_logging(log_dir)

        # Uncomment below for debug server
        # FLASK_APP.run()
        serve(FLASK_APP, host='0.0.0.0', port=5001)
    except OSError as err:
        logging.exception(err)
        sys.exit(1)


if __name__ == '__main__':
    main()
