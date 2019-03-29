import cv2
import os

def open_video(path, max_frame):
    cap = cv2.VideoCapture(path)
    count = 0
    ret, image_np = cap.read()
    while (max_frame == -1 or count <= max_frame) and ret:
        yield image_np
        count += 1
        ret, image_np = cap.read()
    cv2.destroyAllWindows()


def save_img(path, image_np_list):
    index = 0
    for image_np in image_np_list:
        name = str(index) + '.jpg'
        index += 1
        cv2.imwrite(os.path.join(path, name), image_np)


def save_video(path, image_np_list):
    name = path + '.avi'
    out = cv2.VideoWriter(
            name,
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (800, 600))
    for image_np in image_np_list:
        out.write(cv2.resize(image_np, (800, 600)))
    cv2.destroyAllWindows()
