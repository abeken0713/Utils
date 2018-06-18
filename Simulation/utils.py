import numpy as np
from scipy.integrate import RK45

class RegularRK45(RK45):
    def __init__(self, fun, t0, y0, t_bound, t_step, rtol=0.001, atol=1e-06, vectorized=False, **extraneous):
        super(RegularRK45, self).__init__(fun, t0, y0, t_bound, t_step, rtol=0.001, atol=1e-06, vectorized=False, **extraneous)
        self.t_now = t0
        self.y_now = y0
        self.t_step = t_step

    def step(self):
        super(RegularRK45, self).step()
        t = self.t_now + self.t_step
        if self.t_old <= t < self.t:
            self.y_now = self.dense_output()(t)
            self.t_now = t