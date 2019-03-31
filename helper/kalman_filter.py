import cv2
import numpy as np


class KalmanFilter(object):
    def __init__(self):
        self.filter = cv2.KalmanFilter(4, 2)
        # 设置测量矩阵
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 设置转移矩阵
        self.dt = 0.01
        self.filter.transitionMatrix = np.array([[1, 0, self.dt, 0],
                                                 [0, 1, 0, self.dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        # 设置过程噪声协方差矩阵
        self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.1

    def predict(self):
        return np.squeeze(self.filter.predict()[:2])

    def correct(self, z):
        current_measurement = np.array([[np.float32(z[0])], [np.float32(z[1])]])
        return np.squeeze(self.filter.correct(current_measurement)[:2])









