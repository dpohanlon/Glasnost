import numpy as np

class RejectionSampler(object):
    """

    Simple accept/reject 1D sampler with adaptive cieling value.

    """

    def __init__(self, function, min, max):

        self.function = function

        self.min = min
        self.max = max

    def sample(self, nSamples):
        samples = []

        ceiling = 1.0

        while len(samples) < nSamples:

            xVal = np.random.uniform(self.min, self.max)
            fVal = self.function(xVal)
            yVal = np.random.uniform(0, ceiling)

            if yVal <= fVal:
                samples.append(xVal)
            if ceiling < fVal:
                ceiling = fVal

        return samples
