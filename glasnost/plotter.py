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

from seaborn import apionly as sns
colours = sns.light_palette((210, 90, 60), input="husl")

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
                               'elinewidth' : 1.0, 'color' : 'k', 'markersize' : 6, 'fmt' : '.'}

    def plotData(self, nDataBins = None, minVal = None, maxVal = None):

        minVal = minVal if minVal else np.min(self.data)
        maxVal = maxVal if maxVal else np.max(self.data)

        bins = np.linspace(minVal, maxVal, nDataBins if nDataBins else 100)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        binnedData, b = np.histogram(self.data, bins = bins)

        binEdges = (b + 0.5 * (b[1] - b[0]))[:-1]
        dataXErrs = 0.5 * (b[1] - b[0])
        dataYErrs = np.sqrt(binnedData)

        plt.errorbar(binEdges, binnedData, xerr = dataXErrs, yerr = dataYErrs, **self.errorbarConfig)

        plt.ylim(ymin = 0)
        plt.xlim(minVal, maxVal)

        return fig
