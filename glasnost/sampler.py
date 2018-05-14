import numpy as np

class RejectionSampler(object):
    """

    Simple accept/reject 1D sampler with adaptive cieling value.

    """

    def __init__(self, function, min, max, ceiling = 1.0):

        self.function = function

        self.min = min
        self.max = max

        self.ceiling = ceiling

    def sample(self, nSamples):
        samples = []

        while len(samples) < nSamples:

            xVal = np.random.uniform(self.min, self.max)
            fVal = self.function(xVal)
            yVal = np.random.uniform(0, self.ceiling)

            if yVal <= fVal:
                samples.append(xVal)
            if self.ceiling < fVal:
                self.ceiling = fVal

        return samples
