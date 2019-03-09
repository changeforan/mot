# coding: utf-8
import os
import sys
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from siamese import siamese_network
from helper import tools, video_util, detection, tracklet
import construct_similarity_matrix
from helper import sample

PATH_TO_MODEL = os.path.join('save_models', 'faster_rcnn', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('player_label.txt')
VIDEO_PATH = '/home/cs/Desktop/dataset/ISSIA/filmrole/filmrole4.avi'
NUM_CLASSES = 1
GLOBAL_SEARCH = False


def load_tf_model(path_to_model):
    """Load a (frozen) Tensorflow model into memory.
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_label_map(path_to_labels, num_classes):
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, num_classes) 
    category_index = label_map_util.create_category_index(categories)
    return category_index


def find_surrounding_boxes(path, new_boxes, global_search):
    """find the boxes around the last box of a path.

    Arguments:
        path {[type]} -- [description]
        new_boxes {[type]} -- [description]
        global_search {boolean} -- return all new boxes or not
    Returns:
        [type] -- [description]
    """
    box = path.last_box
    close_boxes = []
    if global_search:
        return new_boxes
    else:
        max_distance = (box[3] - box[1]) * 5
        for b in new_boxes:
            distance = tools.calc_distance_between_2_vectors(box, b)
            if distance < max_distance:
                close_boxes.append(b)
    return close_boxes


def find_box(path, close_boxes, image_np, sampler):
    min_distance = sys.float_info.max
    box_to_add = None
    box_feat = None
    for box in close_boxes:
        feat = sampler.sample(box, image_np)
        distance = tools.calc_distance_between_2_vectors(path.last_feat, feat)
        if distance < min_distance:
            min_distance = distance
            box_to_add = box
            box_feat = feat
    return box_to_add, box_feat


def add_boxes_to_paths(new_boxes,
                       feat_conv,
                       paths,
                       image_np,
                       sampler):
    for path in paths:
        close_boxes = find_surrounding_boxes(path, new_boxes, GLOBAL_SEARCH)
        box_to_add, box_feat = find_box(path, close_boxes, image_np, sampler)
        if box_to_add is not None:
            path.add_point(tools.get_point(box_to_add))
            path.last_box = box_to_add
            path.last_feat = box_feat
            new_boxes.remove(box_to_add)
    for box in new_boxes:
        path = tracklet.Tracklet(
            tools.get_point(box),
            box,
            sampler.sample(box, image_np),
            len(paths) + 1)
        paths.append(path)


def get_sess(detection_graph):
    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        detection_feat_cnn = detection_graph.get_tensor_by_name(
            'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/Relu:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        return sess, image_tensor, detection_boxes, detection_scores, detection_classes, detection_feat_cnn, num_detections


def detecting(image_tensor,
              detection_boxes,
              detection_scores,
              detection_classes,
              detection_feat_cnn,
              num_detections,
              image_np,
              sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return sess.run(
        [detection_feat_cnn,
         detection_boxes,
         detection_scores,
         detection_classes,
         num_detections],
        feed_dict={image_tensor: image_np_expanded})


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


def visualize_paths(image_np, paths):
    vis_util.visualize_paths_on_image_array(
        image_np,
        paths,
        use_normalized_coordinates=True,
        line_thickness=8)


def main():
    detection_graph = load_tf_model(PATH_TO_MODEL)
    category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)
    video = video_util.open_video(VIDEO_PATH, 1)

    # the tracklet set at time T-1
    tracklets = []
    image_np_list = []
    sess, image_tensor, detection_boxes, \
    detection_scores, detection_classes, \
    detection_feat_cnn, num_detections = get_sess(detection_graph)

    # the siamese network model for extracting feat_sim
    siamese_model = siamese_network.Siamese()

    for image_np in video:
        feat_cnn, boxes, scores, classes, _ = detecting(
            image_tensor,
            detection_boxes,
            detection_scores,
            detection_classes,
            detection_feat_cnn,
            num_detections,
            image_np,
            sess)

        # the detection set at time T
        detections = []
        detected_boxes = tools.get_all_detected_boxes(boxes, scores)
        for box in detected_boxes:
            location = tools.get_point(box)
            player_img = tools.get_player_img(box, image_np)
            feat_cnn = [1,1,1,1]
            feat_sim = np.squeeze(siamese_model.run(player_img))
            detections.append(detection.Detection(location, feat_cnn, feat_sim))

        S = np.array([])
        try:
            S = construct_similarity_matrix.get_similarity_matrix(tracklets, detections)
        except construct_similarity_matrix.DetectionsEmpty:
            continue
        except construct_similarity_matrix.TrackletsEmpty:
            for d in detections:
                tracklets.append(tracklet.Tracklet(d.location,
                                                   d.feat_cnn,
                                                   d.feat_sim,
                                                   len(tracklets) + 1
                                                   ))

        print(S)
        print(S.shape)
        row_index, col_index = linear_sum_assignment(S)
        print(row_index + 1)
        print(col_index + 1)
        print(S[row_index, col_index])
        # add_boxes_to_paths(new_boxes, feat_conv, paths, image_np, sampler)
        # visualize_boxes_and_labels(image_np, boxes, classes, scores, category_index)
        # visualize_paths(image_np, paths)
        # image_np_list.append(image_np)
    video_util.save_video('out.avi', image_np_list)
    sess.close()


if __name__ == "__main__":
    main()
