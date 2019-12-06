import numpy as np

class RejectionSampler(object):
    """

    Simple accept/reject 1D sampler with adaptive cieling value.

    """

    def __init__(self, function, minVal, maxVal, ceiling = 1.0):

        self.function = function

        self.min = minVal
        self.max = maxVal

        self.ceiling = ceiling

    def sample(self, nSamples):

        nSamples = int(nSamples)
        samples = []

        blockSize = 10000
        # Maybe instead sample all in one go? Cannot adjust ceiling on the fly

        while len(samples) < nSamples:

            xVals = np.random.uniform(self.min, self.max, size = blockSize)
            fVals = self.function(xVals)
            yVals = np.random.uniform(0, self.ceiling, size = blockSize)

            if len(fVals[fVals > self.ceiling]) > 0:
                self.ceiling = np.max(fVals)
            else:
                samples.extend(list(xVals[yVals <= fVals]))

        return samples[:nSamples]
