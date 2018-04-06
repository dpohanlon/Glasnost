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

from iminuit import Minuit

from iminuit.util import describe

import glasnost as gl

x = np.linspace(-3, 6, 1000)
# data = np.concatenate((np.random.normal(0, 1, 100), np.random.normal(3, 1, 100)))
data = np.random.normal(0, 1, 100)

with gl.name_scope("massFit"):

    with gl.name_scope("coreGaussian"):

        m1 = gl.Parameter(0.1, name = 'mean')
        s1 = gl.Parameter(1.1, name = 'sigma')

        y1 = gl.Parameter(90, name = 'yield')

        g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

    # with gl.name_scope("secondGaussian"):
    #
    #     m2 = gl.Parameter(3.5, name = 'mean')
    #     s2 = gl.Parameter(1.0, name = 'sigma')
    #
    #     y2 = gl.Parameter(110, name = 'yield')
    #
    #     g2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})
    #
    # model = gl.Model(name = 'model', initialFitYields = {g1.name : y1, g2.name : y2},
    #                                  initialFitComponents = {g1.name : g1, g2.name : g2})

    model = gl.Model(name = 'model', initialFitYields = {g1.name : y1},
                                     initialFitComponents = {g1.name : g1})

plt.plot(data, g1.prob(data), lw = 1.0)
# plt.plot(data, g2.prob(data), lw = 1.0)
plt.savefig('test.pdf')
plt.clf()

# print(g1.mean)
# print(g2.sigma)

plt.plot(data, model.prob(data), lw = 1.0)
plt.savefig('testModel.pdf')
plt.clf()

# pprint(model.getFloatingParameterValues())

# pprint(gl.utils.scopesUsed)

model.setData(data)

# pprint(model.getFloatingParameterValues())

minuit = Minuit(model, errordef = 1.0, **model.getInitialParameterValuesAndStepSizes())
minuit.migrad()

# plt.hist(data, bins = 25, normed = True, histtype = 'stepfilled')
plt.plot(x, model.prob(x), lw = 1.0)
plt.savefig('fitModel.pdf')
plt.clf()
