import numpy as np

__all__ = ["ZeroMeanFunction", "MeanFunctionSum"]


class MeanFunction:
    def calc_mean(self, x):
        raise NotImplementedError("This has to be implemented in a child class")


class ZeroMeanFunction(MeanFunction):
    def calc_mean(self, x):
        return np.zeros(len(x))


class MeanFunctionSum(MeanFunction):
    def __init__(self, *means):
        for mean in means:
            if not isinstance(mean, MeanFunction):
                raise TypeError
        self.means = means

    def calc_mean(self, x):
        meansum = np.zeros(len(x))
        for mean in self.means:
            meansum += mean.calc_mean(x)
        return meansum
