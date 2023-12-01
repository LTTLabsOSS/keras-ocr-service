"""Test find word functionality"""
from pathlib import Path
import re
import cv2
import keras_ocr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_server import mass_center, load_image, label_text

INPUT_DIR = "images"
OUTPUT_DIR = "test_output_keras"
TEST_WORD = "options"

def main() -> None:
    """This is a test to process all the images inside of the images folder and output 
    bounding boxes of found text to test_output_keras.
    """
    output_dir = Path(OUTPUT_DIR)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print("Output directory is created.")

    pipeline = keras_ocr.pipeline.Pipeline()

    for image_name in Path(INPUT_DIR).iterdir():
        image = load_image(str(image_name))
        out_path = str(output_dir / f"{image_name.stem}_out.png")

        pred = pipeline.recognize([image])[0]
        num_tar_word = 0
        for text, box in pred:
            polygon = box[np.newaxis].astype("int32")
            label_text(image, polygon, box, text)

            if re.match(TEST_WORD, text):
                cv2.polylines(image, polygon, True, (0, 0, 255), 2, )
                for c in polygon:
                    center_x, center_y = mass_center(cv2.moments(c))
                    print("Center coordinate : " +  str(center_x) + " , " + str(center_y))
                    num_tar_word += 1
        if num_tar_word == 0:
            print("Target word not found.")

        plt.imshow(image)
        plt.show()
        cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


print(tf.__version__)
print(tf.config.list_physical_devices())

if __name__ == '__main__':
    main()
