import os
import time
import cv2

def get_latest_image(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    if image_files:
        return os.path.join(directory, image_files[0])
    else:
        return None        

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    upload_dir = os.path.join(script_dir, "uploaded")
    nothing_to_display = os.path.join(script_dir, "images", "nothing_to_display.png")
    nothing_to_display_image = cv2.resize(cv2.imread(nothing_to_display), (1280, 720))
    previous_latest_image = nothing_to_display_image
    while True:
        latest_image_path = get_latest_image(upload_dir)
        if latest_image_path:
            image = cv2.imread(latest_image_path)
            resized_image = cv2.resize(image, (1280, 720))
            previous_latest_image = resized_image
            cv2.imshow("latest upload", resized_image)
        elif previous_latest_image is not None:
            cv2.imshow("latest upload", previous_latest_image)
        else: 
            cv2.imshow("latest upload", nothing_to_display_image)
        time.sleep(0.05)
        cv2.waitKey(1)

    
    