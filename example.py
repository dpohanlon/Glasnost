import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])

from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.ticker as plticker

from matplotlib import cm

from collections import OrderedDict

rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams.update({'figure.autolayout': True})

import numpy as np

from pprint import pprint

import glasnost as gl

import corner

np.random.seed(42)

# data = np.concatenate((np.random.normal(-1., 4., 1000), np.random.normal(0., 1., 1000), np.random.normal(3., 1., 1000)))

data1 = np.concatenate((np.random.normal(0., 1., 10000), np.random.normal(0., 3., 10000)))
data2 = np.concatenate((np.random.normal(0., 1., 5000), np.random.normal(0., 3., 5000)))

with gl.name_scope('massFit'):

    m1 = gl.Parameter(0.2, name = 'mean')

    with gl.name_scope("model1"):

        with gl.name_scope("firstGaussian"):

            s1 = gl.Parameter(1.0, name = 'sigma')

            y1 = gl.Parameter(len(data1) // 2, name = 'yield', minVal = 0)

            g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope("secondGaussian"):

            s2 = gl.Parameter(3.0, name = 'sigma')

            y2 = gl.Parameter(len(data1) // 2, name = 'yield', minVal = 0)

            g2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

        initialFitYields1 = {g1.name : y1, g2.name : y2}
        initialFitComponents1 = {g1.name : g1, g2.name : g2}

        model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1)

    with gl.name_scope("model2"):

        with gl.name_scope("firstGaussian"):

            y3 = gl.Parameter(len(data2) // 2, name = 'yield', minVal = 0)

        with gl.name_scope("secondGaussian"):

            y4 = gl.Parameter(len(data2) // 2, name = 'yield', minVal = 0)

        initialFitYields2 = {g1.name : y3, g2.name : y4}

        model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents1)

    models = {model1.name : model1, model2.name : model2}

    model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

s1, s2 = model.generate(-10, 10)

plt.hist(s1, bins = 50)
plt.savefig('samples1.pdf')
plt.clf()
plt.hist(s2, bins = 50)
plt.savefig('samples2.pdf')

#
# fitter = gl.Fitter(model, backend = 'emcee')
#
# res = fitter.fit([data1, data2], verbose = True)
#
# print(res.acceptance_fraction)
#
# samples = res.chain[:, 200:, :].reshape((-1, 7))
#
# means = np.mean(samples, axis = 0)
# stds = np.std(samples, axis = 0)
#
# pprint(means)
#
# pprint(stds)
#
# for i in range(7):
#
#     plt.plot(samples[:,i], lw = 2.0)
#     plt.savefig('samples' + str(i) + '.png')
#     plt.clf()
#
#     plt.hist(samples[:,i], bins = 50)
#     plt.savefig('samplesHist' + str(i) + '.png')
#     plt.clf()
#
# c = corner.corner(samples, truths = [0.0, 1.0, 1000., 3.0, 1000., 500., 500.])
# c.savefig('corner.pdf')
#
# f, axarr = plt.subplots(1, 2, sharey = True)
#
# plotter = gl.Plotter(model1, data1)
# plotter.plotDataModel(ax = axarr[0])
#
# axarr[0].set_xlabel('The things')
# axarr[0].set_ylabel('How many things')
#
# plt.xlim(np.min(data1), np.max(data1))
#
# # axarr[0].set_yscale("log", nonposy='clip')
# # axarr[0].set_ylim(1)
#
# plotter = gl.Plotter(model2, data2)
# plotter.plotDataModel(ax = axarr[1])
#
# axarr[1].set_xlabel('The things')
#
# plt.xlim(np.min(data2), np.max(data2))
#
# # axarr[1].set_yscale("log", nonposy='clip')
# # axarr[1].set_ylim(1)
#
# plt.savefig('testPlot.pdf', bbox_inches = 'tight')
