import os
import cv2

def open_path(path, start, end):
    total_img = os.listdir(path)[start - 1: end - 1]
    for img in total_img:
        img_np = cv2.imread(os.path.join(path, img))
        print(img)
        yield img_np

