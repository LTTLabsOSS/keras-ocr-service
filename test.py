import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import keras_ocr
from pathlib import Path
import numpy as np
import os
import re
import argparse

##
# This is a test to process all the images inside of the images folder and output 
# bounding boxes of found text to test_output_keras.
##
def main() -> None: 
    # set image path and export folder directory
    input_dir = Path(r".\images")
    output_dir = Path(r".\test_output_keras")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', default='options', type=str, help="Enter a target word")
    args = parser.parse_args()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory is created.")

    pipeline = keras_ocr.pipeline.Pipeline()

    word = 'options'

    for image_name in os.listdir(input_dir):
        image = cv2.imread(os.path.join(input_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_without_suff = os.path.splitext(image_name)[0]
        out_path = os.path.join(output_dir, img_without_suff)

        pred = pipeline.recognize([image])[0]
        num_tar_word = 0
        for text, box in pred:
            polygon = box[np.newaxis].astype("int32")
            cv2.polylines(image, polygon, True, (0, 255, 0), 2, )
            org = (box[0][0].astype("int32"), box[0][1].astype("int32"))
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
            
            if re.match(word, text):
                cv2.polylines(image, polygon, True, (0, 0, 255), 2, )
                for c in polygon:
                    # compute the center of the contour
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
                    print("Center coordinate : " +  str(cX) + " , " + str(cY))
                    num_tar_word += 1
        if num_tar_word == 0:
            print("Target word not found.")

        plt.imshow(image)
        plt.show()
        cv2.imwrite(out_path + '_out.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        break



print(tf.__version__)
print(tf.config.list_physical_devices())

if __name__ == '__main__':
    main()