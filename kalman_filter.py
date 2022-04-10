import numpy as np

ACCELERATION = 0


class KalmanFilter:
    def __init__(self, x, y):
        # TODO: Initial x, xv, y, yv

        self.P_t = np.identity(4)
        self.X_hat = np.array([[x], [y], [0], [0]])
        self.U_t = ACCELERATION
        self.B_t = np.array([[0], [0], [0], [0]])
        self.Q_t = np.identity(4)
        self.H_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R_t = (np.identity(2) * 0.1) ** 5

    def predict(self, delta_t=1/15):

        F_t = np.array([[1, 0, delta_t, 0], [0, 1, 0, delta_t],
                        [0, 0, 1, 0], [0, 0, 0, 1]])

        self.X_hat = F_t.dot(self.X_hat) + \
            (self.B_t.dot(self.U_t).reshape(self.B_t.shape[0], -1))

        self.P_t = np.diag(
            np.diag(F_t.dot(self.P_t).dot(F_t.transpose()))) + self.Q_t

    def update(self, Z_t):
        # Z_t shape (2, 1)

        K_prime = self.P_t.dot(self.H_t.transpose()).dot(
            np.linalg.inv(self.H_t.dot(self.P_t).dot(self.H_t.transpose()) + self.R_t))

        self.X_hat = self.X_hat + K_prime.dot(Z_t - self.H_t.dot(self.X_hat))
        self.P_t = self.P_t - K_prime.dot(self.H_t).dot(self.P_t)

    @ property
    def cov(self) -> np.array:
        return self.P_t

    @ property
    def mean(self) -> np.array:
        return self.X_hat

    @ property
    def x(self) -> float:
        return int(self.X_hat[0][0])

    @ property
    def y(self) -> float:
        return int(self.X_hat[1][0])
