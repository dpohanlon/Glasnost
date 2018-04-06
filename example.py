import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])

from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

from matplotlib import cm

rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams.update({'figure.autolayout': True})

import numpy as np

from pprint import pprint

import glasnost as gl

np.random.seed(42)

# data = np.concatenate((np.random.normal(-1., 4., 1000), np.random.normal(0., 1., 1000), np.random.normal(3., 1., 1000)))

data = np.concatenate((np.random.normal(0., 1., 1000), np.random.normal(0., 3., 1000)))


with gl.name_scope("massFit"):

    with gl.name_scope("coreGaussian"):

        m1 = gl.Parameter(0.1, name = 'mean')
        s1 = gl.Parameter(1.0, name = 'sigma')

        y1 = gl.Parameter(len(data) // 2, name = 'yield')

        g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

    with gl.name_scope("secondGaussian"):

        s2 = gl.Parameter(4.0, name = 'sigma')

        y2 = gl.Parameter(len(data) // 2, name = 'yield')

        g2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

    # with gl.name_scope("thirdGaussian"):
    #
    #     m3 = gl.Parameter(-1.0, name = 'mean')
    #     s3 = gl.Parameter(5.0, name = 'sigma')
    #
    #     y3 = gl.Parameter(len(data) // 3, name = 'yield')
    #
    #     g3 = gl.Gaussian({'mean' : m3, 'sigma' : s3})

    # initialFitYields = {g1.name : y1, g2.name : y2, g3.name : y3}
    # initialFitComponents = {g1.name : g1, g2.name : g2, g3.name : g3}
    #
    # model = gl.Model(name = 'model', initialFitYields = initialFitYields,
    #                                  initialFitComponents = initialFitComponents)

    initialFitYields = {g1.name : y1, g2.name : y2}
    initialFitComponents = {g1.name : g1, g2.name : g2}

    model = gl.Model(name = 'model', initialFitYields = initialFitYields,
                                     initialFitComponents = initialFitComponents)

fitter = gl.Fitter(model, backend = 'minuit')

res = fitter.fit(data, verbose = True)

plotter = gl.Plotter(model, data)

fig, ax = plotter.plotDataModel()
ax.set_xlabel('The whats')
ax.set_ylabel('How many things')

plt.savefig('testPlot.pdf')
