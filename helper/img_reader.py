import os
import cv2


def open_path(path, start, end):
    total_img = os.listdir(path)
    total_img.sort()
    for img in total_img[start - 1: end - 1]:
        img_np = cv2.imread(os.path.join(path, img))
        yield img_np

