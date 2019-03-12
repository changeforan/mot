from object_detection.utils import label_map_util
import numpy as np
import tensorflow as tf


class Detector:

    def __init__(self, path_to_model, path_to_labels, num_classes):
        self.detection_graph = self.load_tf_model(path_to_model)
        self.category_index = self.load_label_map(path_to_labels, num_classes)
        self.sess, self.image_tensor, self.detection_boxes, \
        self.detection_scores, self.detection_classes, \
        self.detection_feat_cnn, self.num_detections = self.get_sess(self.detection_graph)

    @staticmethod
    def load_label_map(path_to_labels, num_classes):
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, num_classes)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    @staticmethod
    def load_tf_model(path_to_model):
        """Load a (frozen) tensorflow model into memory.
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    @staticmethod
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

    @staticmethod
    def detecting(image_tensor,
                  detection_boxes,
                  detection_scores,
                  detection_classes,
                  detection_feat_cnn,
                  num_detections,
                  image_np,
                  sess):
        image_np_expanded = np.expand_dims(image_np, axis=0)
        return sess.run([detection_feat_cnn,
                        detection_boxes,
                        detection_scores,
                        detection_classes,
                        num_detections],
                        feed_dict={image_tensor: image_np_expanded})


    def sess_end(self):
        self.sess.close()

    def detecting_from_img(self, image_np):

        return  self.detecting(self.image_tensor,
                               self.detection_boxes,
                               self.detection_scores,
                               self.detection_classes,
                               self.detection_feat_cnn,
                               self.num_detections,
                               image_np,
                               self.sess
                               )


