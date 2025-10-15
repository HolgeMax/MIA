
class NonLinearRegistration:
    def __init__(self, fixed, moving, M = 2, order = 3):
        import numpy as np
        self.fixed = fixed
        self.moving = moving
        self.M = M
        self.order = order
        self.h = (self.fixed.shape[0] - 1) / (self.M - 1)
        self.x = np.linspace(0, self.fixed.shape[0]-1, self.fixed.shape[0])
        # intialize weights
        self.w = np.zeros((2, self.M**2))
        
        # compute basis functions
        self.phi = self.basis_functions()

        self.F_flat = self.fixed.flatten()

        self.energy = []

        def eval_BSpline(self, order=0):
            import numpy as np
            if order == 0: # Zero order B-Spline
                out = np.zeros(self.x.shape)
                out[(self.x >= -0.5) & (self.x < 0.5)] = 1
                out[np.abs(self.x) == 0.5] = 0.5
                out[(self.x < -0.5) | (self.x >= 0.5)] = 0

            elif order == 1: # First order B-Spline
                out = np.zeros(self.x.shape)
                mask = np.abs(self.x) < 1
                out[mask] = 1 - np.abs(self.x[mask])

            elif order == 3: # Third order B-Spline
                out = np.zeros(self.x.shape)
                mask1 = np.abs(self.x) < 1
                out[mask1] = 2/3-np.abs(self.x[mask1])**2 + 0.5*np.abs(self.x[mask1])**3
                mask2 = (1 <= np.abs(self.x)) & (np.abs(self.x) < 2)
                out[mask2] = (1/6)*(2-np.abs(self.x[mask2]))**3
            return out

        def basis_functions(self):
            self.centers = self.h * np.arange(self.M)
            for i in range(self.fixed.shape[1]):
            phi1 = self.eval_BSpline((self.x[:, None] - self.centers[None, :]) / self.h, order=self.order)
            phi2 = self.eval_BSpline((self.x[:, None] - self.centers[None, :]) / self.h, order=self.order)
            return phi1

        