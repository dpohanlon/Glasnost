import numpy as np

from pprint import pprint

import glasnost as gl

data = np.linspace(0, 10, 100)

with gl.name_scope("massFit"):
    with gl.name_scope("coreGaussian"):
        m = gl.Parameter(1.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        g = gl.Gaussian({'mean' : m, 'sigma' : s})

        print(m)
        print(g)

        # pprint(m + 3)

        pprint(g(data))

        pprint(np.sqrt(m + 2) - 3)

        print(np.sqrt((m + s)))
