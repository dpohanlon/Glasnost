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

data = np.linspace(-10, 10, 1000)

with gl.name_scope("massFit"):

    with gl.name_scope("coreGaussian"):

        m = gl.Parameter(1.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        y = gl.Parameter(100, name = 'yield')

        g = gl.Gaussian({'mean' : m, 'sigma' : s})

    # Fix these names
    model = gl.Model(name = 'model', initialFitYields = {g.name : y}, initialFitComponents = {g.name : g})

plt.plot(data, g.prob(data))
plt.savefig('test.pdf')
plt.clf()

plt.plot(data, model.prob(data))
plt.savefig('testModel.pdf')
plt.clf()

pprint(gl.utils.scopesUsed)
