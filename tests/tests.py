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

import corner

np.random.seed(42)

def simpleGaussianModel(mean, width, nEvents):

    with gl.name_scope('simpleGaussianTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 5700)
        s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

        gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        gaussYield = gl.Parameter(nEvents, name = 'gaussYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {gauss.name : gaussYield}
    fitComponents = {gauss.name : gauss}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def simpleGaussianTests():

    # Test generating and fitting back with the same model

    model = simpleGaussianModel(5279., 20., 1000000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussian1Test.pdf')
    plt.clf()

    # Test generating with NumPy and fitting back with a similar model

    model = simpleGaussianModel(5270., 23., 1050000.) # Similar enough to what is generated

    dataNumpy = np.random.normal(5279, 20, 1000000)
    dataNumpy = dataNumpy[(dataNumpy > 4200) & (dataNumpy < 5700)]

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataNumpy, verbose = True)

    plotter = gl.Plotter(model, dataNumpy)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussian2Test.pdf')
    plt.clf()

    print(model)

if __name__ == '__main__':

    simpleGaussianTests()
