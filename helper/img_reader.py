import os
import cv2

def open_path(path, start, end):
    total_img = os.listdir(path)[start - 1: end - 1]
    for img in total_img:
        img_np = cv2.imread(os.path.join(path, img))
        yield img_np


def save_video(path, image_np_list):
    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (800, 600))
    for image_np in image_np_list:
        out.write(cv2.resize(image_np, (800, 600)))
    cv2.destroyAllWindows()
