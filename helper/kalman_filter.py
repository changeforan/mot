import numpy as np


class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of
    the system and the variance or uncertainty of the estimate.
    Predict and Correct methods implement the functionality
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    Attributes: None
    """

    def __init__(self):
        """Initialize variable used by Kalman Filter class
        Args:
            None
        Return:
            None
        """
        self.dt = .05   # delta time

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])# matrix in observation equations, measurement function A ==> for H
        self.x = np.zeros((4, 1))# previous state vector, state mean know as x, u ==> for x

        # (x,y) tracking object center
        self.z = np.array([[1], [0]])# vector of observations, measurement matrix , b ==> for z

        self.P = np.diag((3.0, 3.0, 3.0, 3.0)) * 1000# covariance matrix

        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # state transition mat,     dinamics, transition matrix

        self.Q = np.eye(self.x.shape[0])
        # process noise matrixnp.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) * 0.03
        self.R = np.eye(self.z.shape[0])
        # np.eye(self.z.shape[0])  # observation noise matrix np.array([[1,0,],[0,1]]) * 0.05
        self.lastResult = np.array([[0], [255]])

        self.B = np.array([[0.5 * self.dt ** 2],
                           [0.5 * self.dt ** 2],
                           [self.dt],
                           [self.dt]])
        self.u = 50

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector, u ==> for x
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.x = np.round(np.dot(self.F, self.x)) + np.round(np.dot(self.B, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.x  # same last predicted result
        # print ("pred", self.x)
        return self.x[0]

    def correct(self, z, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u ==> for x
        A: matrix in observation equations, measurement function, A ==> for H
        b: vector of observations, measurement matrix, b ==> for z
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        C: system uncertainty , S ==> for C
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.z = np.array([[self.lastResult[0][0]], [self.lastResult[1][0]]])
            LastR = self.z


        else:  # update using detection
            self.z = z

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        self.x = np.round(self.x + np.dot(K, (self.z - np.dot(self.H, self.x))))
        self.P = self.P - np.dot(K, np.dot(S, K.T))

        if not flag:
            self.x = self.lastResult

        return self.x


if __name__ == '__main__':
    SS = KalmanFilter()
    z = [(1,1), (2,2), (3,3), (4,2), (5,1)]
    for i in range(len(z)) :
        print('ordem',i)
        Correct = SS.correct([z[i][0], z[i][1]], 1)
        Predict = SS.predict()
        print ("Predict :", Predict)
        print("")