import numpy as np


class KalmanFilter:
    def __init__(self, initial_pos, initial_velocity, acceleration):
        self.X_t = np.array([initial_pos, initial_velocity])
        self.P_t = np.eye(2) ** 2
        self.H_t = np.zeros((1, 2))
        self.H_t[0, 0] = 1

        self.acceleration = acceleration

    def predict(self, dt):
        # x = F x
        # P = F P Ft + G Gt a
        F = np.array([[1, dt], [0, 1]])
        G = np.array([0.5 * dt**2, dt]).reshape((2, 1))

        self.X_t = F.dot(self.X_t)
        self.P_t = F.dot(self.P_t).dot(F.T) + G.dot(G.T) * self.acceleration

    def update(self, meas_value, meas_variance):

        Z_t = np.array([meas_value])
        R_t = np.array([meas_variance])

        y_t = Z_t - self.H_t.dot(self.X_t)
        S_t = self.H_t.dot(self.P_t).dot(self.H_t.T) + R_t

        K = self.P_t.dot(self.H_t.T).dot(np.linalg.inv(S_t))

        self.X_t = self.X_t + K.dot(y_t)
        self.P_t = (np.eye(2) - K.dot(self.H_t)).dot(self.P_t)

    @property
    def x(self) -> int:
        return int(self.X_t[0])

    @property
    def x_float(self) -> float:
        return self.X_t[0]
