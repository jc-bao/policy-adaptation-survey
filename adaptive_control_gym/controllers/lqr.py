class LRQ:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def select_action(self, state):
        return -self.gain @ (x - self.x_0) + self.u_0
        