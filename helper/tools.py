import numpy as np
import cv2


def calc_distance_between_2_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.sqrt(np.sum(np.square(v1 - v2)))
    return dist


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def get_all_detected_boxes(
        original_boxes,
        scores,
        max_boxes_to_draw=20,
        min_score_thresh=0.6):
    boxes = []
    original_boxes = np.squeeze(original_boxes)
    scores = np.squeeze(scores)
    for i in range(min(max_boxes_to_draw, original_boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = original_boxes[i].tolist()
            boxes.append(box)
    return boxes


def get_point(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def get_width(box):
    return abs(box[3] - box[1])


def get_player_img(box, image_np, small=False):
    im_width, im_height, _ = image_np.shape
    ymin, xmin, ymax, xmax = box
    player_img = image_np[int(im_width * ymin): int(im_width * ymax) + 1,
                          int(im_height * xmin): int(im_height * xmax) + 1]
    if small:
        return cv2.resize(player_img, (28, 28))
    return cv2.resize(player_img, (128, 128))
