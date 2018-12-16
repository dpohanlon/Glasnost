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

# from seaborn import apionly as sns # Deprecated?
import seaborn.apionly as sns

# colours = sns.light_palette((210, 90, 60), input="husl")
# colours = ['#ff3000', '#ff8f00', '#14b8b8']

import numpy as np

class Plotter(object):
    """
    
    Handle plotting data and model projections.

    """

    # Deal with multiple data/model projections for simultaneous fits
    # Probably will need to specify independent variable (like in RooFit)

    def __init__(self, model = None, data = None):

        self.model = model
        self.data = data

        self.errorbarConfig = {'capthick' : 1.0, 'capsize' : 0.0, 'barsabove' : True,
                               'elinewidth' : 1.0, 'color' : 'k', 'markersize' : 6,
                               'fmt' : '.', 'zorder' : 100}

        self.totalCurveConfig = {'lw' : 3.0, 'color' : sns.xkcd_rgb["steel blue"],
                                 'zorder' : 99}

        self.componentCurveConfig = {'lw' : 2.0, 'zorder' : 98, 'ls' : '--'}

        self.dataBinning = None

    def guessNBins(self, minVal, maxVal):
        # Using the Freedman-Diaconis rule

        iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        width = 2. * iqr / np.cbrt(len(self.data))

        nBins = int((maxVal - minVal) / width)

        return nBins

    def plotData(self, data, nDataBins = None, minVal = None, maxVal = None, fig = False, ax = None, log = False, **kwargs):

        minVal = minVal if minVal else np.min(data)
        maxVal = maxVal if maxVal else np.max(data)

        bins = np.linspace(minVal, maxVal, nDataBins if nDataBins else self.guessNBins(minVal, maxVal))

        self.dataBinning = bins

        binnedData, b = np.histogram(data, bins = bins)

        binEdges = (b + 0.5 * (b[1] - b[0]))[:-1]

        dataXErrs = 0.5 * (b[1] - b[0])
        dataYErrs = np.sqrt(binnedData)

        f = plt
        if ax:
            f = ax

        f.errorbar(binEdges, binnedData, xerr = dataXErrs, yerr = dataYErrs, **self.errorbarConfig)

        if log : f.yscale("log", nonposy='clip')
        else : plt.ylim(ymin = 0)

        plt.xlim(minVal, maxVal)

        return

    def plotModel(self, data, model, minVal = None, maxVal = None, fig = False, nSamples = 1000, ax = None, log = False, **kwargs):

        minVal = minVal if minVal else np.min(data)
        maxVal = maxVal if maxVal else np.max(data)

        binWidth = (self.dataBinning[1] - self.dataBinning[0]) if not self.dataBinning is None else None

        if not binWidth:

            # If this is not a plot with data, then plot unnormalised, as it doesn't matter
            # but warn that if this is, then the data should be plotted first

            print('WARNING: No binning scheme is defined.')
            print('If this is for a plot with data, run the data plot first.')

            binWidth = 1.0

        x = np.linspace(minVal, maxVal, nSamples)

        f = plt
        if ax:
            f = ax

        modelToPlot = model.prob(x)

        # Normalise to total fitted yield
        modelToPlot *= model.getTotalYield()

        # Normalised according to the data binning also plotted
        modelToPlot *= binWidth

        f.plot(x, modelToPlot, **self.totalCurveConfig) # Doesn't work with fracs + normalisation, fix me

        # Components
        # WOULD BE GOOD TO HAVE A POSTPROCESS STEP THAT FINALISES ALL INFORMATION INCLUDING FRACS FOR ALL COMPONENTS,
        # PROPAGATES CONSTRAINTS, ETC
        # In the mean time, calculate this on the fly

        yields = {}
        if model.fitYields:
            yields = model.fitYields
        else:
            totalYield = model.getTotalYield()
            if totalYield < 1E-3:
                print('WARNING: Total yield must be set if plotting with fractional components.')

            totFF = 0

            for n, frac in model.fitFracs.items():
                yields[n] = totalYield * frac.value
                totFF += frac.value

            for k in model.fitComponents.keys():
                if k not in yields:
                    yields[k] = totalYield * (1. - totFF)

        norm = model.getComponentIntegrals(minVal, maxVal)

        colours = sns.color_palette("YlOrRd", len(model.fitComponents.keys()))

        if len(model.fitComponents.keys()) > 1:

            for i, (n, c) in enumerate(model.fitComponents.items()):
                plt.plot(x, c.prob(x) * yields[n] * binWidth / norm[n], color = colours[i], **self.componentCurveConfig)

        if log : f.yscale("log", nonposy='clip')
        else : plt.ylim(ymin = 0)

        plt.xlim(minVal, maxVal)

        return fig, ax

        # yields = list(model.fitYields.values())[0:]
        # for i, c in enumerate(list(model.fitComponents.values())[0:]):
            # f.fill_between(x, 0, c.prob(x) * yields[i] * binWidth, color = colours[i])

        # plt.ylim(ymin = 0)
        # plt.xlim(minVal, maxVal)

        # return

    def plotDataModel(self, **kwargs):

        self.plotData(data = self.data, **kwargs)
        self.plotModel(data = self.data, model = self.model, **kwargs)

        return
