# coding: utf-8
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from object_detection.utils import visualization_utils as vis_util
from siamese import siamese_network
from helper import tools, detection, tracklet, img_reader, bbox_tools
import construct_similarity_matrix
import detector
import argparse
import cv2

PATH_TO_MODEL = os.path.join('save_models', 'faster_rcnn', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('player_label.txt')
NUM_CLASSES = 1


# Thresholds
DISAPPEAR_THRESHOLD = 5
QUALITY_THRESHOLD = 0.95
NEAR_THRESHOLD = 1.0


width = 624
height = 352

def visualize_boxes_and_labels(image_np,
                               boxes,
                               classes,
                               scores,
                               category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)


def visualize_tracklets(image_np, tracklets):
    tools.visualize_tracklets_on_image_array(
        image_np,
        tracklets,
        use_normalized_coordinates=True,
        line_thickness=8)


def save_tracklets(tracklets: [tracklet.Tracklet]):
    p = [t.points for t in tracklets]
    np.savetxt("points.csv", p, delimiter=",", fmt='%s')


def find_nearest_detection(origin, detections, threshold=NEAR_THRESHOLD):
    nearest_dis = None
    nearest_detection = None
    for d in detections:
        point = d.location
        dis = tools.calc_distance_between_2_vectors(origin, point)
        if nearest_dis is None:
            nearest_dis = dis
            nearest_detection = d
            continue
        if dis < nearest_dis:
            nearest_dis = dis
            nearest_detection = d
    if nearest_detection is not None:
        if nearest_dis > threshold * nearest_detection.width:
            nearest_detection = None
    return nearest_detection


def get_new_detections(boxes, scores, image_np, siamese_model):
    detections = []
    detected_boxes = tools.get_all_detected_boxes(boxes, scores)
    for box in detected_boxes:
        location = tools.get_point(box)
        width = tools.get_width(box)
        player_img = tools.get_player_img(box, image_np)
        feat_cnn = [1, 1, 1, 1]
        feat_sim = np.squeeze(siamese_model.run(player_img))
        detections.append(detection.Detection(location, feat_cnn, feat_sim, width, box))
    return detections


def save_player_img(video_path, tracklet_id, img, img_id):
    video_name = str(str(video_path).split('/')[-1]).split('.')[0]
    if not os.path.exists(video_name):
        os.mkdir(video_name)
    if not os.path.exists(video_name + '/' + tracklet_id):
        os.mkdir(video_name + '/' + tracklet_id)
    cv2.imwrite(video_name + '/' + tracklet_id + '/' + img_id + '.jpg', img)


def get_target_detection(obj, detections):
    gt_bbox = [obj[1], obj[0], obj[0] + obj[3], obj[0] + obj[2]]
    det_bbox = [[d.box[1] * width,
                 d.box[0] * height,
                 d.box[3] * width,
                 d.box[2] * height] for d in detections]
    IoUs = bbox_tools.bbox_iou(np.array([gt_bbox]),np.array(det_bbox))
    index = np.argmax(IoUs[0])
    print(np.max(IoUs[0]))
    print(detections[index].box)
    print(IoUs[0, index])



def tracking(args):
    img_set = img_reader.open_path(args.input, 40, 376)
    obj = (132, 256, 18, 42)
    progress = 0
    # the tracklet set at time T-1
    tracklets = []
    # the detector
    player_detector = detector.Detector(PATH_TO_MODEL, PATH_TO_LABELS, NUM_CLASSES)
    # the siamese network model for extracting feat_sim
    siamese_model = siamese_network.Siamese()
    result_img = []
    for image_np in img_set:
        progress += 1
        _, boxes, scores, classes, _ = player_detector.detecting_from_img(image_np)
        # the detection set at time T
        detections = get_new_detections(boxes, scores, image_np, siamese_model)
        target = get_target_detection(obj, detections)
        return
        # construct similarity matrix S
        S = np.array([])
        try:
            S = construct_similarity_matrix.get_similarity_matrix(tracklets, detections)
        except construct_similarity_matrix.DetectionsEmpty:
            continue
        except construct_similarity_matrix.TrackletsEmpty:
            for d in detections:
                tracklets.append(tracklet.Tracklet(d, len(tracklets) + 1))
            continue

        # the Hungarian algorithm
        trk_index, det_index = linear_sum_assignment(1. - S)
        low_quality_trk_index = []
        low_quality_det_index = []
        for i, j in zip(trk_index, det_index):
            if S[i, j] < QUALITY_THRESHOLD:
                low_quality_trk_index.append(i)
                low_quality_det_index.append(j)
                continue
            tracklets[i].add_detection(detections[j])
            if args.save_player_img:
                save_player_img(
                    args.video,
                    str(tracklets[i].id),
                    tools.get_player_img(detections[j].box, image_np),
                    str(len(tracklets[i].detections)))
        tracklets_left_index = [x for x in range(0, len(tracklets))
                                if x not in trk_index
                                or x in low_quality_trk_index]
        detections_left_index = [x for x in range(0, len(detections))
                                 if x not in det_index
                                 or x in low_quality_det_index]

        if tracklets_left_index:
            disappear_tracklets = []
            for t in tracklets_left_index:
                origin = tracklets[t].predict()
                foreground_detection = find_nearest_detection(origin, detections)
                if foreground_detection is not None and detections.index(
                        foreground_detection) not in detections_left_index:
                    tracklets[t].add_foreground_detection(foreground_detection)
                if tracklets[t].vanish() > DISAPPEAR_THRESHOLD:
                    disappear_tracklets.append(tracklets[t])
            for t in disappear_tracklets:
                tracklets.remove(t)
        if detections_left_index:
            for d in detections_left_index:
                tracklets.append(tracklet.Tracklet(detections[d], len(tracklets) + 1))

        visualize_boxes_and_labels(image_np, boxes, classes, scores, player_detector.category_index)
        visualize_tracklets(image_np, tracklets)
        result_img.append(image_np)
    player_detector.sess_end()
    video_util.save_video(args.output, result_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        dest='input'
    )
    parser.add_argument(
        '--output',
        dest='output',
        default='./out'
    )
    parser.add_argument(
        '--save_player_img',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    tracking(args)


if __name__ == "__main__":
    main()
