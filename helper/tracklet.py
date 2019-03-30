import numpy as np
import copy
from . import detection
from . import kalman_filter


class Tracklet:
    def __init__(self, det: detection.Detection, id: int, disappear=0):
        self.color = np.random.randint(30)
        self.detections = [det]
        self.id = id
        self.disappear = disappear
        self.filter = kalman_filter.KalmanFilter()
        for i in range(20):
            self.current_prediction = self.filter.predict()
            self.filter.correct(det.location)
        self.t = 5

    def add_detection(self, det):
        self.filter.correct(det.location)
        if len(self.detections) > self.t:
            det.change_location(self.current_prediction)
        self.current_prediction = self.filter.predict()
        self.detections.append(det)
        self.disappear = 0

    def add_foreground_detection(self, foreground_det):
        self.filter.correct(foreground_det.location)
        det = copy.copy(self.detections[-1])
        if len(self.detections) > self.t:
            det.change_location(self.current_prediction, True)
        else:
            det.change_location(foreground_det.location, True)
        self.current_prediction = self.filter.predict()
        self.detections.append(det)
        self.disappear = 0

    def vanish(self):
        # det = copy.copy(self.detections[-1])
        # det.location = self.predict()
        # self.detections.append(det)
        # self.move(*det.location)
        self.disappear += 1
        return self.disappear

    def predict(self):
        if len(self.detections) < self.t:
            return self.detections[-1].location
        else:
            return self.current_prediction

    def get_points(self):
        return [x.location for x in self.detections]

    def get_feat_cnn(self):
        if len(self.detections) == 1:
            return self.detections[0].feat_cnn
        return np.mean([x.feat_cnn for x in self.detections], axis=0)

    def get_feat_sim(self):
        if len(self.detections) == 1:
            return self.detections[0].feat_sim
        return np.mean([x.feat_sim for x in self.detections], axis=0)
