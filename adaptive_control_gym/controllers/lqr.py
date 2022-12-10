import control as ct

class LRQ:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = ct.lqr(A, B, Q, R)[0]

    def __call__(self, state):
        return (-self.K @ state)[0]