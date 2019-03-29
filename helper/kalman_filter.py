import numpy as np
import cv2


def move(filter, x, y):
    x *= 100
    y *= 100
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    # 用来修正卡尔曼滤波的预测结果
    filter.correct(current_measurement)  # 用当前测量来校正卡尔曼滤波器


def get_filter():
    kalman = cv2.KalmanFilter(4, 2)
    # 设置测量矩阵
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # 设置转移矩阵
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 设置过程噪声协方差矩阵
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    return kalman


def predict(filter):
    current_prediction = filter.predict()
    return current_prediction[0] / 100, current_prediction[1] / 100



if __name__ == '__main__':

    z = [(50, 50), (20, 20), (30, 30), (40, 20), (50, 10)]
    # fil = get_filter()
    fil = cv2.KalmanFilter(4, 2)
    fil.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # 设置转移矩阵
    fil.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 设置过程噪声协方差矩阵
    fil.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    for x in z:
        current_measurement = np.array([[np.float32(x[0])], [np.float32(x[1])]])
        fil.correct(current_measurement)
        fil.predict()[:2]




