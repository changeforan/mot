import numpy as np

class Tracklet:
    def __init__(self, point, feat_cnn, feat_sim, id):
        self.color = np.random.randint(30)
        self.points = [point]
        self.last_feat_cnn = feat_cnn
        self.last_feat_sim = feat_sim
        self.id = id

    def add_detection(self, detection):
        self.points.append(detection.point)
        self.last_feat_cnn = detection.feat_cnn
        self.last_feat_sim = detection.feat_sim


