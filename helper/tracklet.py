import numpy as np
from . import kalman_filter
from . import detection

class Tracklet:
    def __init__(self, det:detection.Detection, id:int, disappear=0):
        self.color = np.random.randint(30)
        self.detections = [det]
        self.id = id
        self.disappear = disappear
        self.ss = kalman_filter.KalmanFilter()
        #self.ss.correct([point[0] * 100, point[1] * 100], 1)

    def add_detection(self, det):
        self.detections.append(det)
        self.disappear = 0
        #self.ss.correct(detection.location, 1)


    def add_foreground_detection(self, det):
        self.detections.append(det)
        self.disappear = 0
        #self.ss.correct(detection.location, 1)


    def vanish(self):
        self.disappear += 1
        return self.disappear

    def predict(self):
        return self.detections[-1].location
        # return self.ss.predict() / 100.

    def get_points(self):
        return [x.location for x in self.detections]

    def get_feat_cnn(self):
        return np.mean([x.feat_cnn for x in self.detections], axis=1)

    def get_feat_sim(self):
        return np.mean([x.feat_sim for x in self.detections], axis=1)