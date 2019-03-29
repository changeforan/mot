import numpy as np
from . import detection
import cv2


class Tracklet:
    def __init__(self, det:detection.Detection, id:int, disappear=0):
        self.color = np.random.randint(30)
        self.detections = [det]
        self.id = id
        self.disappear = disappear
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 设置转移矩阵
        self.filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 设置过程噪声协方差矩阵
        self.filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.move(*det.location)
        self.current_prediction = np.zeros((2, 1), np.float32)

    def move(self, x, y):
        x *= 100
        y *= 100
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        # 用来修正卡尔曼滤波的预测结果
        self.filter.correct(current_measurement)
        self.current_prediction= self.filter.predict()[:2] / 100

    def add_detection(self, det):
        if len(self.detections) > 20:
            pre_det = detection.Detection(det.location, det.feat_cnn, det.feat_sim, det.width, det.box)
            pre_det.location = self.current_prediction
            self.detections.append(pre_det)
        else:
            self.detections.append(det)
        self.disappear = 0
        self.move(*det.location)

    def add_foreground_detection(self, det):
        self.detections.append(det)
        self.disappear = 0
        self.move(*det.location)

    def vanish(self):
        self.disappear += 1
        return self.disappear

    def predict(self):
        if len(self.detections) < 20:
            return self.detections[-1].location
        else:
            p = self.current_prediction
            return p

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