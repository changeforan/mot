# coding: utf-8

import numpy as np
import os
import sys
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

from helper import tools
from helper import _video
from helper import _path
from helper import sample

MODEL_NAME = 'save_models/faster_rcnn'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('/home/cs/work/training/config', 'object_detection.pbtxt')
VIDEO_PATH = '/home/cs/Videos/cap/fifa4.mp4'
NUM_CLASSES = 1
GLOBAL_SEARCH = False

def load_tf_model():
# ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # ## Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    return detection_graph, category_index



def find_close_boxes(path, new_boxes):
    box = path.last_box
    close_boxes = []
    if GLOBAL_SEARCH:
        return new_boxes
    else:
        max_distance = (box[3] - box[1]) * 5
        for b in new_boxes:
            distance = tools.calc_distance_between_2_vectors(box, b)
            if distance < max_distance:
                close_boxes.append(b)
    return close_boxes


def find_box(path, close_boxes, sampler):
    min_distance = sys.float_info.max
    box_to_add = None
    box_feat = None
    for box in close_boxes:
        feat = sampler.sample(box)
        distance = tools.calc_distance_between_2_vectors(path.last_feat, feat)
        if distance < min_distance:
            min_distance = distance
            box_to_add = box
            box_feat = feat
    return box_to_add, box_feat


def add_boxes_to_paths(new_boxes,
                       feat_conv,
                       paths,
                       image_np):
    sampler = sample.SiameseSampler(feat_conv, image_np)
    for path in paths:
        close_boxes = find_close_boxes(path, new_boxes)
        box_to_add, box_feat = find_box(path, close_boxes, sampler)
        if box_to_add is not None:
            path.add_point(tools.get_point(box_to_add))
            path.last_box = box_to_add
            path.last_feat = box_feat
            new_boxes.remove(box_to_add)
    for box in new_boxes:
        path = _path.Path(
            tools.get_point(box),
            box,
            sampler.sample(box),
            len(paths) + 1
        )
        paths.append(path)


def get_sess(detection_graph):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            detection_feat_conv = detection_graph.get_tensor_by_name(
                'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_1/bottleneck_v1/Relu:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            return sess, image_tensor, detection_boxes, detection_scores, detection_classes, detection_feat_conv, num_detections

def detecting(image_tensor,
              detection_boxes,
              detection_scores,
              detection_classes,
              detection_feat_conv,
              num_detections,
              image_np,
              sess):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return sess.run(
        [detection_feat_conv,
         detection_boxes,
         detection_scores,
         detection_classes,
         num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )


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
        line_thickness=8
    )


def main():
    detection_graph, category_index = load_tf_model()
    video = _video.Video(VIDEO_PATH, 180)
    paths = []
    sess, image_tensor, detection_boxes, detection_scores, detection_classes, detection_feat_conv, num_detections = get_sess(detection_graph)
    for image_np in video:
        feat_conv, boxes, scores, classes, _ = detecting(
            image_tensor,
            detection_boxes,
            detection_scores,
            detection_classes,
            detection_feat_conv,
            num_detections,
            image_np,
            sess)
        new_boxes = tools.get_all_detected_boxes(boxes, scores)
        add_boxes_to_paths(new_boxes, feat_conv, paths, image_np)
        visualize_boxes_and_labels(image_np, boxes, classes, scores, category_index)
        visualize_paths(image_np, paths)
        video.write(image_np)


if __name__ == "__main__":
    main()


