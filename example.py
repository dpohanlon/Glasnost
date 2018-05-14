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

def testPhysicalSimBF():

    with gl.name_scope('massFit'):

        minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
        maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

        m = gl.Parameter(5.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        with gl.name_scope('model1'):

            with gl.name_scope("expBkg"):
                yBkg1 = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

                aExp1 = gl.Parameter(-0.2, name = 'aExp')
                expBkg1 = gl.Exponential({'a' : aExp1, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig1 = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

                g1 = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('model2'):

            with gl.name_scope("expBkg"):
                yBkg2 = gl.Parameter(1500, name = 'yieldExp', minVal = -1000)

                aExp2 = gl.Parameter(-0.1, name = 'aExp')
                expBkg2 = gl.Exponential({'a' : aExp2, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig2 = gl.Parameter(1000, name = 'yieldSig', minVal = -1000)

                g2 = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields1 = {g1.name : ySig1, expBkg1.name : yBkg1}
    initialFitComponents1 = {g1.name : g1, expBkg1.name : expBkg1}

    initialFitYields2= {g2.name : ySig2, expBkg2.name : yBkg2}
    initialFitComponents2 = {g2.name : g2, expBkg2.name : expBkg2}

    model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1)
    model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents2)

    data1 = model1.generate(0, 10)
    data2 = model2.generate(0, 10)

    model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit([data1, data2], verbose = True, nIterations = 2000)

    samples = res.chain[:, 100:, :].reshape((-1, model.getNFloatingParameters()))

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot1.pdf')
    plt.clf()

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot2.pdf')
    plt.clf()

    c = corner.corner(samples)#, truths = [-0.2, 500., 5.0, 1.0, 500.])
    c.savefig('corner.pdf')
    plt.clf()

def testPhysicalSim():

    with gl.name_scope('massFit'):

        minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
        maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

        m = gl.Parameter(5.0, name = 'mean')
        s = gl.Parameter(1.0, name = 'sigma')

        with gl.name_scope('model1'):

            with gl.name_scope("expBkg"):
                yBkg1 = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

                aExp1 = gl.Parameter(-0.2, name = 'aExp')
                expBkg1 = gl.Exponential({'a' : aExp1, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig1 = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

                g1 = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('model2'):

            with gl.name_scope("expBkg"):
                yBkg2 = gl.Parameter(1500, name = 'yieldExp', minVal = -1000)

                aExp2 = gl.Parameter(-0.1, name = 'aExp')
                expBkg2 = gl.Exponential({'a' : aExp2, 'min' : minParam, 'max' : maxParam})

            with gl.name_scope("gaussSignal"):
                ySig2 = gl.Parameter(1000, name = 'yieldSig', minVal = -1000)

                g2 = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields1 = {g1.name : ySig1, expBkg1.name : yBkg1}
    initialFitComponents1 = {g1.name : g1, expBkg1.name : expBkg1}

    initialFitYields2= {g2.name : ySig2, expBkg2.name : yBkg2}
    initialFitComponents2 = {g2.name : g2, expBkg2.name : expBkg2}

    model1 = gl.Model(initialFitYields = initialFitYields1, initialFitComponents = initialFitComponents1)
    model2 = gl.Model(initialFitYields = initialFitYields2, initialFitComponents = initialFitComponents2)

    data1 = model1.generate(0, 10)
    data2 = model2.generate(0, 10)

    model = gl.SimultaneousModel(initialFitComponents = [model1, model2])

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit([data1, data2], verbose = True, nIterations = 2000)

    samples = res.chain[:, 100:, :].reshape((-1, model.getNFloatingParameters()))

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot1.pdf')
    plt.clf()

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(nDataBins = 50)
    plt.savefig('testPlot2.pdf')
    plt.clf()

    c = corner.corner(samples)#, truths = [-0.2, 500., 5.0, 1.0, 500.])
    c.savefig('corner.pdf')
    plt.clf()

def testPhysical():

    with gl.name_scope('massFit'):
        with gl.name_scope("expBkg"):
            yBkg = gl.Parameter(500, name = 'yieldExp', minVal = -1000)

            minParam =  gl.Parameter(0.0, name = 'minVal', fixed = True)
            maxParam =  gl.Parameter(10.0, name = 'maxVal', fixed = True)

            aExp = gl.Parameter(-0.2, name = 'aExp')
            # expBkg = gl.Exponential({'a' : aExp, 'min' : gl.Parameter(0.0, fixed = True), 'max' : gl.Parameter(10.0, fixed = True)})
            expBkg = gl.Exponential({'a' : aExp, 'min' : minParam, 'max' : maxParam})

        with gl.name_scope("gaussSignal"):
            ySig = gl.Parameter(500, name = 'yieldSig', minVal = -1000)

            m = gl.Parameter(5.0, name = 'mean')
            s = gl.Parameter(1.0, name = 'sigma')

            g = gl.Gaussian({'mean' : m, 'sigma' : s})

    initialFitYields = {g.name : ySig, expBkg.name : yBkg}
    initialFitComponents = {g.name : g, expBkg.name : expBkg}

    model = gl.Model(initialFitYields = initialFitYields, initialFitComponents = initialFitComponents)

    data = model.generate(0, 10)

    # plt.hist(data, bins = 50, histtype = 'stepfilled')
    # plt.savefig('textExp.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit(data, verbose = True, nIterations = 5000)

    samples = res.chain[:, 500:, :].reshape((-1, model.getNFloatingParameters()))
    plotter = gl.Plotter(model, data)
    plotter.plotDataModel(nDataBins = 30)
    plt.savefig('testPlot.pdf')
    plt.clf()

    c = corner.corner(samples, truths = [-0.2, 500., 5.0, 1.0, 500.], lw = 1.0)
    c.savefig('corner.pdf')
    plt.clf()

def testToy():

    data = np.concatenate((np.random.normal(-1., 4., 1000), np.random.normal(0., 1., 1000), np.random.normal(3., 1., 1000)))

    data1 = np.concatenate((np.random.normal(0., 1., 1000), np.random.normal(0., 3., 1000)))
    data2 = np.concatenate((np.random.normal(0., 1., 500), np.random.normal(0., 3., 500)))

    with gl.name_scope('massFit'):

        m1 = gl.Parameter(0.2, name = 'mean')
        s1 = gl.Parameter(1.0, name = 'sigma1')
        s2 = gl.Parameter(3.0, name = 'sigma2')

        with gl.name_scope("model1"):

            with gl.name_scope("firstGaussian"):

                y1 = gl.Parameter(len(data1) // 2, name = 'yield', minVal = 0)

                g1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            with gl.name_scope("secondGaussian"):

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

    # s1, s2 = model.generate(-10, 10)
    #
    # plt.hist(s1, bins = 100, histtype = 'stepfilled')
    # plt.savefig('samples1.pdf')
    # plt.clf()
    # plt.hist(s2, bins = 100, histtype = 'stepfilled')
    # plt.savefig('samples2.pdf')

    fitter = gl.Fitter(model, backend = 'emcee')

    res = fitter.fit([data1, data2], verbose = True)

    samples = res.chain[:, 200:, :].reshape((-1, 7))

    means = np.mean(samples, axis = 0)
    stds = np.std(samples, axis = 0)

    for i in range(7):

        plt.plot(samples[:,i], lw = 2.0)
        plt.savefig('samples' + str(i) + '.png')
        plt.clf()

        plt.hist(samples[:,i], bins = 50)
        plt.savefig('samplesHist' + str(i) + '.png')
        plt.clf()

    c = corner.corner(samples, truths = [0.0, 1000., 1000., 500., 500., 1., 3.])
    c.savefig('corner.pdf')

    f, axarr = plt.subplots(1, 2, sharey = True)

    plotter = gl.Plotter(model1, data1)
    plotter.plotDataModel(ax = axarr[0])

    axarr[0].set_xlabel('The things')
    axarr[0].set_ylabel('How many things')

    plt.xlim(np.min(data1), np.max(data1))

    # axarr[0].set_yscale("log", nonposy='clip')
    # axarr[0].set_ylim(1)

    plotter = gl.Plotter(model2, data2)
    plotter.plotDataModel(ax = axarr[1])

    axarr[1].set_xlabel('The things')

    plt.xlim(np.min(data2), np.max(data2))

    # axarr[1].set_yscale("log", nonposy='clip')
    # axarr[1].set_ylim(1)

    plt.savefig('testPlot.pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    testPhysicalSimBF()
