from helper import detection, tracklet
import numpy as np
import math

w_1 = .5
w_2 = 1.
w_3 = 1.


class TrackletsEmpty(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message


class DetectionsEmpty(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message


def calc_cosine_similarity(A, B):
    num = np.dot(A, B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom  # cosine
    return 0.5 + 0.5 * cos


def get_mot_matrix(tracklets:[tracklet.Tracklet], detections:[detection.Detection]):
    mot = np.empty((len(tracklets), len(detections)))
    for i in range(0, len(tracklets)):
        last_point = tracklets[i].points[-1]
        for j in range(0, len(detections)):
            location = detections[j].location
            width = detections[j].width
            dis = math.e ** (-w_1 * (((last_point[0] - location[0]) / width) ** 2 + ((last_point[1] - location[1]) / width)** 2))
            mot[i][j] = dis
    return mot


def get_cnn_matrix(tracklets: [tracklet.Tracklet], detections: [detection.Detection]):
    cnn = np.empty((len(tracklets), len(detections)))
    for i in range(0, len(tracklets)):
        last_feat_cnn = tracklets[i].last_feat_cnn
        for j in range(0, len(detections)):
            feat_cnn = detections[j].feat_cnn
            cnn[i][j] = w_2 * calc_cosine_similarity(last_feat_cnn, feat_cnn)
    return cnn


def get_app_matrix(tracklets: [tracklet.Tracklet], detections: [detection.Detection]):
    app = np.empty((len(tracklets), len(detections)))
    for i in range(0, len(tracklets)):
        last_feat_sim = tracklets[i].last_feat_sim
        for j in range(0, len(detections)):
            feat_sim = detections[j].feat_sim
            app[i][j] = w_3 * calc_cosine_similarity(last_feat_sim, feat_sim)
    return app


def get_similarity_matrix(tracklets:[tracklet.Tracklet], detections:[detection.Detection])->np.array:
    if not tracklets:
        raise TrackletsEmpty('tracklets is empty')
    if not detections:
        raise DetectionsEmpty('detections is empty')

    M = get_mot_matrix(tracklets, detections)
    C = get_cnn_matrix(tracklets, detections)
    A = get_app_matrix(tracklets, detections)
    return np.multiply(M, C, A)

