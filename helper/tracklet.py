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
        self.current_prediction = self.filter.predict()
        self.filter.correct(det.location)

    def add_detection(self, det):
        self.filter.correct(det.location)
        if len(self.detections) > 5:
            det.location = self.current_prediction
        self.current_prediction = self.filter.predict()
        self.detections.append(det)
        self.disappear = 0

    def add_foreground_detection(self, foreground_det):
        self.filter.correct(foreground_det.location)
        det = copy.copy(self.detections[-1])
        if len(self.detections) > 5:
            det.location = self.current_prediction
        else:
            det.location = foreground_det.location
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
        if len(self.detections) < 5:
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
