import numpy as np
from . import kalman_filter


class Tracklet:
    def __init__(self, point, feat_cnn, feat_sim, id, disappear=0, quality=1):
        self.color = np.random.randint(30)
        self.points = [point]
        self.last_feat_cnn = feat_cnn
        self.last_feat_sim = feat_sim
        self.id = id
        self.disappear = disappear
        self.quality = quality
        self.ss = kalman_filter.KalmanFilter()
        self.ss.correct([point[0] * 100, point[1] * 100], 1)

    def add_detection(self, detection, score):
        self.points.append(detection.location)
        self.last_feat_cnn = detection.feat_cnn
        self.last_feat_sim = detection.feat_sim
        self.disappear = 0
        self.quality = 0.5 * (self.quality + score)
        self.ss.correct(detection.location, 1)


    def add_foreground_detection(self, detection):
        self.points.append(detection.location)
        self.disappear = 0
        self.ss.correct(detection.location, 1)


    def vanish(self):
        self.disappear += 1
        return self.disappear

    def predict(self):
        return self.points[-1]
        # return self.ss.predict() / 100.



