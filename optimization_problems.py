import numpy as np


class OptimizationProblem:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def evaluate(self, position):
        raise NotImplementedError("Subclasses must override evaluate() method")

    def get_dimensions(self):
        return self.dimensions


class AckleyFunction(OptimizationProblem):
    def __init__(self, dimensions, lower_bound=-30, upper_bound=30, a=20, b=0.2, c=2*np.pi):
        super().__init__(dimensions)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.a = a
        self.b = b
        self.c = c

    def is_within_bounds(self, position):
        return all(self.lower_bound <= x <= self.upper_bound for x in position)

    def evaluate(self, position):
        if not self.is_within_bounds(position):
            return 1e6
        d = self.dimensions
        sum1 = np.sum(np.square(position))
        sum2 = np.sum(np.cos(self.c * np.array(position)))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)

        return term1 + term2 + self.a + np.exp(1)
