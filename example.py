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

data1 = np.concatenate((np.random.normal(0., 1., 1000), np.random.normal(0., 3., 1000)))
data2 = np.concatenate((np.random.normal(0., 1., 500), np.random.normal(0., 3., 500)))

with gl.name_scope('massFit'):

    with gl.name_scope("model1"):

        with gl.name_scope("firstGaussian"):

            m1 = gl.Parameter(0.2, name = 'mean')
            s1 = gl.Parameter(1.1, name = 'sigma')

            y1 = gl.Parameter(len(data1) // 2 + 100, name = 'yield')

            g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope("secondGaussian"):

            s2 = gl.Parameter(5.1, name = 'sigma')

            y2 = gl.Parameter(len(data1) // 2 + 100, name = 'yield')

            g2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

        initialFitYields1 = {g1.name : y1, g2.name : y2}
        initialFitComponents1 = {g1.name : g1, g2.name : g2}

        model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1)

    with gl.name_scope("model2"):

        with gl.name_scope("firstGaussian"):

            y3 = gl.Parameter(len(data2) // 2 + 100, name = 'yield')

        with gl.name_scope("secondGaussian"):

            y4 = gl.Parameter(len(data2) // 2 + 100, name = 'yield')

        initialFitYields2 = {g1.name : y3, g2.name : y4}

        model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents1)

    models = {model1.name : model1, model2.name : model2}

    model = gl.Model(initialFitComponents = models, simultaneous = True)

fitter = gl.Fitter(model, backend = 'minuit')

res = fitter.fit([data1, data2], verbose = True)

#

# f, axarr = plt.subplots(2, sharey = True)

f, axarr = plt.subplots(1, 2, sharey = True)

plotter = gl.Plotter(model1, data1)
plotter.plotDataModel(ax = axarr[0])

axarr[0].set_xlabel('The things')
axarr[0].set_ylabel('How many things')

plt.xlim(np.min(data1), np.max(data1))

# ax.set_yscale("log", nonposy='clip')
# ax.set_ylim(1)

plotter = gl.Plotter(model2, data2)
plotter.plotDataModel(ax = axarr[1])

axarr[1].set_xlabel('The things')
# axarr[1].set_ylabel('How many things')

plt.xlim(np.min(data2), np.max(data2))

# ax.set_yscale("log", nonposy='clip')
# ax.set_ylim(1)

plt.savefig('testPlot.pdf', bbox_inches = 'tight')
