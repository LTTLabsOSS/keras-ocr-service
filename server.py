from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
from pathlib import Path
import keras_ocr
import os
import cv2
import numpy as np
import re
from waitress import serve

pipeline = keras_ocr.pipeline.Pipeline()

upload_dir = "uploaded"
output_dir = "output"

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.


def find_word(word, screenshot):
    image = cv2.imread(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred = pipeline.recognize([image])[0]
    for text, box in pred:
        polygon = box[np.newaxis].astype("int32")
        cv2.polylines(image, polygon, True, (0, 255, 0), 2, )
        org = (box[0][0].astype("int32"), box[0][1].astype("int32"))
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 1, cv2.LINE_AA)

        if re.match(word, text):
            cv2.polylines(image, polygon, True, (0, 0, 255), 2, )
            for c in polygon:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return {
                    'result': 'found',
                    "x": cX,
                    "y": cY
                }

    # for debug save the latest image that didn't find the target word
    out_path = os.path.join(output_dir, "attempt")
    cv2.imwrite(out_path + '_out.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {
        "result": 'not found'
    }


app = Flask(__name__, instance_relative_config=True)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
    os.makedirs(upload_dir)
    os.makedirs(output_dir)
except OSError:
    pass


@app.route("/process", methods=['POST'])
def process():
    if request.method == 'POST':
        f = request.files['file']
        word = request.form['word']
        file_name = secure_filename(f.filename)
        file_path = os.path.join(upload_dir, file_name)
        f.save(file_path)
        result = find_word(word, file_path)
        # os.remove(file_path)
        return result


if __name__ == '__main__':
    # Uncomment below for debug server
    #app.run()
    serve(app, host='0.0.0.0', port=5000)
