class Distribution:
    """
    This class defines all core functionality for Gaussian distributions.
    All classes that implement some sort of Gaussian are derived from this class
    """

    def __len__(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def update_mean(self, mean):
        raise NotImplementedError("This has to be implemented in a child class")

    def update_cov(self, cov):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_mean(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_cov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_sqrtcov(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_std(self):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_sample(self, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_samples(self, n, seed):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_pdf(self, x):
        raise NotImplementedError("This has to be implemented in a child class")

    def calc_logpdf(self, x):
        raise NotImplementedError("This has to be implemented in a child class")
