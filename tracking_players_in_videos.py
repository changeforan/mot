# coding: utf-8
import os
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from siamese import siamese_network
from helper import tools, video_util, detection, tracklet
import construct_similarity_matrix

PATH_TO_MODEL = os.path.join('save_models', 'faster_rcnn', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('player_label.txt')
VIDEO_PATH = '/home/cs/Desktop/dataset/ISSIA/filmrole/filmrole4.avi'
NUM_CLASSES = 1
GLOBAL_SEARCH = False
DISAPPEAR_THRESHOLD = 5
QUALITY_THRESHOLD = 0.8

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


def visualize_paths(image_np, tracklets):
    vis_util.visualize_tracklets_on_image_array(
        image_np,
        tracklets,
        use_normalized_coordinates=True,
        line_thickness=8)


def main():
    detection_graph = load_tf_model(PATH_TO_MODEL)
    category_index = load_label_map(PATH_TO_LABELS, NUM_CLASSES)
    video = video_util.open_video(VIDEO_PATH, 400)
    progress = 0

    # the tracklet set at time T-1
    tracklets = []
    image_np_list = []
    sess, image_tensor, detection_boxes, \
    detection_scores, detection_classes, \
    detection_feat_cnn, num_detections = get_sess(detection_graph)

    # the siamese network model for extracting feat_sim
    siamese_model = siamese_network.Siamese()

    for image_np in video:
        progress += 1
        print(progress)

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
            continue


        row_index, col_index = linear_sum_assignment(1.- S)
        for i,j in zip(row_index, col_index):
            if S[i,j] < tracklets[i].quality * QUALITY_THRESHOLD:
                row_index.remove(i)
                col_index.remove(j)
            tracklets[i].add_detection(detections[j], S[i,j])

        tracklets_left_index = [x for x in range(0, len(tracklets)) if not x in row_index]
        detections_left_index = [x for x in range(0, len(detections)) if not x in col_index]

        if tracklets_left_index:
            disappear_index = []
            for t in tracklets_left_index:
                if tracklets[t].vanish() > DISAPPEAR_THRESHOLD:
                    disappear_index.append(t)
            for t in disappear_index:
                tracklets.remove(tracklets[t])

        if detections_left_index:
            for d in detections_left_index:
                tracklets.append(tracklet.Tracklet(detections[d].location,
                                                   detections[d].feat_cnn,
                                                   detections[d].feat_sim,
                                                   len(tracklets) + 1
                                                   ))

        visualize_boxes_and_labels(image_np, boxes, classes, scores, category_index)
        visualize_paths(image_np, tracklets)
        image_np_list.append(image_np)
    video_util.save_video('out.avi', image_np_list)
    sess.close()


if __name__ == "__main__":
    main()
