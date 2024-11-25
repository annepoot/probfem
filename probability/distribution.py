import numpy as np


class Distribution:
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def calc_sample(self, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_samples(self, n, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_pdf(self, x):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_logpdf(self, x):
        raise NotImplementedError("This has to be implemented in a child class")


class UnivariateDistribution(Distribution):
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __len__(self):
        return 1

    def calc_mean(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_std(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_var(self):
        return self.calc_std() ** 2


class MultivariateDistribution(Distribution):
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __len__(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_mean(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_cov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_sqrtcov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_std(self):
        return np.sqrt(np.diagonal(self.calc_cov()))
