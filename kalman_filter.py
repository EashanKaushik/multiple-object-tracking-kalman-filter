import numpy as np


class KalmanFilter:
    def __init__(self,
                 x: int,
                 u: int,
                 std_acc: float,
                 std_meas: float) -> None:

        # initialize the model

        # control vector velocity
        self.u = u

        # acceleration standard deviation
        self.std_acc = std_acc

        # observation model
        self.H = np.matrix([[1, 0]])

        # covariance of the observation noise
        self.R = std_meas**2

        # Predicted (a priori) estimate covariance
        self.P = np.eye(2)

        # Predicted (a priori) state estimate
        self.x = np.matrix([[x], [u]])

    def predict(self, dt: float) -> None:
        """predict the position of x

        Args:
            dt (float): time between two frames
        """

        # the control-input model
        B = np.matrix([[(dt**2)/2], [dt]])

        # state-transition model
        F = np.matrix([[1, dt],
                       [0, 1]])

        # covariance of the process noise
        Q = np.matrix([[(dt**4)/4, (dt**3)/2],
                       [(dt**3)/2, dt**2]]) * self.std_acc**2

        # PREDICT

        self.x = np.dot(F, self.x) + np.dot(B, self.u)

        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, z: int) -> None:
        """_summary_

        Args:
            z (int): true measurement (Ground Truth)
        """

        # Innovation or measurement pre-fit residual
        y = self.H * z + np.random.normal(0, 3)
        y = y.item(0)

        # Innovation (or pre-fit residual) covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # UPDATE

        # Updated(a posteriori) state estimate
        self.x = np.round(
            self.x + np.dot(K, (y - np.dot(self.H, self.x))))

        # Updated (a posteriori) estimate covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P

    @ property
    def get_x(self) -> int:
        return int(self.x[0])
