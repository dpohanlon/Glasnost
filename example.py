import numpy as np

import glasnost as gl

data = np.linspace(0, 10, 100)

with gl.name_scope("massFit"):
    with gl.name_scope("coreGaussian"):
        m = gl.Parameter(name = 'mean', initialValue = 1.0)
        s = gl.Parameter(name = 'sigma', initialValue = 1.0)

        g = gl.Gaussian()
