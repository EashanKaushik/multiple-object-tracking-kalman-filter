import numpy as np


class KalmanFilter:
    def __init__(self, x, u, std_acc, std_meas):
        self.u = u
        self.std_acc = std_acc
        self.H = np.matrix([[1, 0]])
        self.R = std_meas**2
        self.P = np.eye(2)
        self.x = np.matrix([[x], [u]])

    def predict(self, dt):
        # Ref :Eq.(9) and Eq.(10)
        # Update time state

        self.B = np.matrix([[(dt**2)/2], [dt]])
        self.A = np.matrix([[1, dt],
                            [0, 1]])
        self.Q = np.matrix([[(dt**4)/4, (dt**3)/2],
                            [(dt**3)/2, dt**2]]) * self.std_acc**2

        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # Ref :Eq.(11) , Eq.(11) and Eq.(13)
        # S = H*P*H'+R
        z = self.H * z + np.random.normal(0, 3)
        z = z.item(0)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)

        self.x = np.round(
            self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)

    @ property
    def get_x(self) -> int:
        print(self.x)
        return int(self.x[0])

    @ property
    def x_float(self) -> float:
        return self.x[0]

    # @property
    # def id(self) -> int:
    #     return int(self._id)
