# coding: utf-8
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from object_detection.utils import visualization_utils as vis_util
from siamese import siamese_network
from helper import tools, detection, tracklet, img_reader, bbox_tools, video_util
import construct_similarity_matrix
import detector
import argparse
import cv2
import circumstance

PATH_TO_MODEL = os.path.join('save_models', 'faster_rcnn', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('player_label.txt')
NUM_CLASSES = 1


# Thresholds
DISAPPEAR_THRESHOLD = 5
QUALITY_THRESHOLD = 0.8
NEAR_THRESHOLD = 1.0


width = 624
height = 352


def visualize_boxes_and_labels(image_np,
                               boxes,
                               classes,
                               scores,
                               category_index,):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        np.squeeze(classes).astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        skip_labels=True,
        skip_scores=True)


def visualize_tracklets(image_np, tracklets):
    tools.visualize_tracklets_on_image_array(
        image_np,
        tracklets,
        use_normalized_coordinates=True,
        line_thickness=4)


def save_tracklets(tracklets: [tracklet.Tracklet]):
    det_bbox = [[d.box[1] * width,
                 d.box[0] * height,
                 (d.box[3] - d.box[1]) * width,
                 (d.box[2] - d.box[0]) * height] for d in tracklets[0].detections]
    # for b in det_bbox:
    #     print(*b)
    return det_bbox


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


def get_target_detection(obj, detections):
    gt_bbox = [obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3]]
    det_bbox = [[d.box[1] * width,
                 d.box[0] * height,
                 d.box[3] * width,
                 d.box[2] * height] for d in detections]
    IoUs = bbox_tools.bbox_iou(np.array([gt_bbox]),np.array(det_bbox))
    index = np.argmax(IoUs[0])
    # print(IoUs[0, index])
    return detections[index]


def calc_AUC(gt_bbox, det_bbox):
    auc = 0.0
    gt_bbox = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_bbox]
    det_bbox = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in det_bbox]
    IoUs = bbox_tools.bbox_iou(np.array(gt_bbox), np.array(det_bbox))
    IoUs = np.diag(IoUs[ :len(det_bbox), :])
    sp = [np.count_nonzero(IoUs >= x / 100) / len(det_bbox) for x in range(0, 100)]
    auc = np.sum(np.multiply(sp, 0.01))
    return auc, sp

def resize_det_bbox(obj, det_bbox):
    det_bbox = [[
        b[0] + 0.5 * (b[2] - obj[2]),
        b[1] + 0.5 * (b[3] - obj[3]),
        obj[2],
        obj[3]
    ] for b in det_bbox]
    return det_bbox


def tracking(args):
    # the detector
    player_detector = detector.Detector(PATH_TO_MODEL, PATH_TO_LABELS, NUM_CLASSES)
    # the siamese network model for extracting feat_sim
    siamese_model = siamese_network.Siamese()
    total_frames = 0
    success_rate = np.zeros(100)
    for shot, gt in circumstance.circu_1:
        print('%s, %s' % (shot, gt))
        img_path = os.path.join(args.input, shot)
        gt_file = os.path.join(img_path, gt)
        with open(gt_file, 'r') as f:
            begin, end = [int(x) for x in next(f).split()]
            gt_bbox = [[int(x) for x in line.split()] for line in f]
            img_set = img_reader.open_path(img_path, begin, end)
            obj = gt_bbox[0]
            progress = begin
            # the tracklet set at time T-1
            tracklets = []
            result_img = []
            for image_np in img_set:
                _, boxes, scores, classes, _ = player_detector.detecting_from_img(image_np)
                # the detection set at time T
                detections = get_new_detections(boxes, scores, image_np, siamese_model)
                if progress == begin:
                    target = get_target_detection(obj, detections)
                    tracklets.append(tracklet.Tracklet(target, 1))
                progress += 1
                # construct similarity matrix S
                S = np.array([])
                try:
                    S = construct_similarity_matrix.get_similarity_matrix(tracklets, detections)
                except construct_similarity_matrix.DetectionsEmpty:
                    continue
                except construct_similarity_matrix.TrackletsEmpty:
                    break

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
                if not tracklets:
                    break
                visualize_boxes_and_labels(image_np,
                                           np.array([tracklets[0].detections[-1].box]),
                                           classes,
                                           np.array([0.9999]),
                                           player_detector.category_index)
                visualize_tracklets(image_np, tracklets)
                result_img.append(image_np)
            print('Total frames %s, tracked %s, end at #%s. ' % (end - begin + 1, progress - begin + 1, progress), end='')
            total_frames += end - begin + 1
            det_bbox = save_tracklets(tracklets)
            if args.resize:
                det_bbox = resize_det_bbox(obj, det_bbox)
            auc, sp = calc_AUC(gt_bbox, det_bbox)
            success_rate = np.array(success_rate) + np.multiply(sp, (end - begin + 1))
            print('AUC is %s' % auc)
    success_rate = success_rate / total_frames
    for i in success_rate:
        print(i)
    player_detector.sess_end()



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
        default='out.avi'
    )

    parser.add_argument(
        '--resize',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    tracking(args)


if __name__ == "__main__":
    main()
