'''
    File name         : kalman_filter.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Kuntima Kiala MIGuEL and Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 18/05/2017
    Python Version    : 2.7
'''

# Import python libraries
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
        self.dt = 0.007  # 0.005  # delta time

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        # np.array([[1, 0], [0, 1]])  # matrix in observation equations, measurement function A ==> for H
        self.x = np.zeros((4, 1))

        # np.zeros((2, 1))  # previous state vector, state mean know as x, u ==> for x

        # (x,y) tracking object center
        self.z = np.array([[1], [0]])
        # np.array([[0], [255]])  # vector of observations, measurement matrix , b ==> for z

        self.P = np.diag((3.0, 3.0, 3.0, 3.0)) * 1000
        # np.diag((3.0, 3.0))  # covariance matrix

        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat,     dinamics, transition matrix

        self.Q = np.eye(self.x.shape[0])
        # np.eye(self.x.shape[0])  # process noise matrixnp.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) * 0.03
        self.R = np.eye(self.z.shape[0])
        # np.eye(self.z.shape[0])  # observation noise matrix np.array([[1,0,],[0,1]]) * 0.05
        self.lastResult = np.array([[0], [255]])

        self.B = np.array([[0.5 * self.dt ** 2],
                           [0.5 * self.dt ** 2],
                           [self.dt],
                           [self.dt]])
        self.u = 5

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
        return self.x / 1000

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

        z = np.array(z) * 1000
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

        return self.x / 1000

if __name__ == '__main__':

    z = [(	0.508647561	,	0.283452913	)	,
        (	0.509904027	,	0.285304263	)	,
        (	0.513042927	,	0.288052469	)	,
        (	0.513438284	,	0.290927649	)	,
        (	0.514692008	,	0.293992415	)	,
        (	0.514156103	,	0.296018019	)	,
        (	0.508971214	,	0.298457772	)	,
        (	0.507026792	,	0.301905885	)	,
        (	0.506543756	,	0.305701688	)	,
        (	0.50458324	,	0.308180287	)	,
        (	0.505075932	,	0.312892064	)	,
        (	0.504645169	,	0.315631524	)	,
        (	0.500742018	,	0.319953889	)	,
        (	0.499737054	,	0.323665842	)	,
        (	0.499726385	,	0.3268262	)	,
        (	0.502308786	,	0.329423353	)	,
        (	0.502923787	,	0.33283326	)	,
        (	0.503394306	,	0.337223083	)	,
        (	0.49891603	,	0.341020003	)	,
        (	0.499298513	,	0.343585685	)	,
        (	0.499552816	,	0.346844837	)	,
        (	0.498672992	,	0.350572899	)	,
        (	0.490882277	,	0.355071917	)	,
        (	0.493919641	,	0.35736835	)	,
        (	0.499034792	,	0.361362278	)	,
        (	0.50037837	,	0.36426051	)	,
        (	0.495776981	,	0.366248414	)	,
        (	0.496742427	,	0.369064495	)	,
        (	0.495818913	,	0.374552965	)	,
        (	0.495754808	,	0.378392711	)	]

    SS = KalmanFilter()

    for i in range(len(z)):

        Predict = SS.predict()
        Correct = SS.correct([z[i][0], z[i][1]], 1)

    for i in range(10):
        predict = SS.predict()[0]
        print(*predict)
        SS.correct(*predict)











