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

def parameterPullsEval(trueParams, fittedParams):

    pulls = []

    for param in trueParams.keys():
        pull = (fittedParams[param].value - trueParams[param]) / fittedParams[param].error
        pulls.append(np.abs(pull))

    return np.array(pulls)

def parameterPullsOkay(trueParams, fittedParams):

    pulls = parameterPullsEval(trueParams, fittedParams)

    if any(pulls[pulls > 3.0]):
        return 1
    else:
        return 0

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

def studentsTModel(mean, width, nu, nEvents):

    with gl.name_scope('studentsTTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 5000, maxVal = 5600)
        s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)
        n = gl.Parameter(nu, name = 'nu', minVal = 0, maxVal = nu * 3)

        studentsT = gl.StudentsT({'mean' : m, 'sigma' : s, 'nu' : n})

        studentsTYield = gl.Parameter(nEvents, name = 'studentsTYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {studentsT.name : studentsTYield}
    fitComponents = {studentsT.name : studentsT}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 5000, maxVal = 5600)

    return model

def simpleGaussianWithExpModel(mean, width, a, nEventsGauss, nEventsExp):

    with gl.name_scope('simpleGaussianWithExpTest'):

        with gl.name_scope('gauss'):

            m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 6000)
            s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

            gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('exp'):

            # Maybe have a global scope for these, if not otherwise specified
            min = gl.Parameter(4200., name = 'min', fixed = True)
            max = gl.Parameter(6000., name = 'max', fixed = True)

            aExp = gl.Parameter(a, name = 'a', minVal = -0.05, maxVal = -0.0001)

            exp = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

        gaussYield = gl.Parameter(nEventsGauss, name = 'gaussYield', minVal = 0.8 * nEventsGauss, maxVal = 1.2 * nEventsGauss)
        expYield = gl.Parameter(nEventsExp, name = 'expYield', minVal = 0.8 * nEventsExp, maxVal = 1.2 * nEventsExp)

    fitYields = {gauss.name : gaussYield, exp.name : expYield}
    fitComponents = {gauss.name : gauss, exp.name : exp}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 6000)

    return model

def simpleGaussianWithUniformModel(mean, width, nEventsGauss, nEventsUni):

    with gl.name_scope('simpleGaussianWithUniformTest'):

        with gl.name_scope('gauss'):

            m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 6000)
            s = gl.Parameter(width, name = 'sigma', minVal = 0, maxVal = width * 5)

            gauss = gl.Gaussian({'mean' : m, 'sigma' : s})

        with gl.name_scope('uni'):

            # Maybe have a global scope for these, if not otherwise specified
            min = gl.Parameter(4200., name = 'min', fixed = True)
            max = gl.Parameter(6000., name = 'max', fixed = True)

            uni = gl.Uniform({'min' : min, 'max' : max})

        gaussYield = gl.Parameter(nEventsGauss, name = 'gaussYield', minVal = 0.8 * nEventsGauss, maxVal = 1.2 * nEventsGauss)
        uniYield = gl.Parameter(nEventsUni, name = 'uniYield', minVal = 0.8 * nEventsUni, maxVal = 1.2 * nEventsUni)

    fitYields = {gauss.name : gaussYield, uni.name : uniYield}
    fitComponents = {gauss.name : gauss, uni.name : uni}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200., maxVal = 6000.)

    return model

def doubleGaussianYieldsModel(mean1, width1, nEvents1, mean2, width2, nEvents2):

    with gl.name_scope('doubleGaussianYieldsModel'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean1, name = 'mean', minVal = 4800, maxVal = 5700)
            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0, maxVal = width1 * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter(mean2, name = 'mean', minVal = 4800, maxVal = 5700)
            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0, maxVal = width2 * 5)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

            gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield', minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

    fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield}
    fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800, maxVal = 5700)

    return model

def simultaneousGaussiansModel(mean1, width1, nEvents1, width2, nEvents2):

    with gl.name_scope('simultaneousGaussiansModel'):

        m1 = gl.Parameter(mean1, name = 'mean', minVal = 5250., maxVal = 5400.)

        with gl.name_scope('gauss1'):

            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0.1, maxVal = width1 * 1.5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

            gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

        with gl.name_scope('gauss2'):

            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0.1, maxVal = width2 * 1.5)

            gauss2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

            gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield', minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

    fitYields1 = {gauss1.name : gauss1Yield}
    fitComponents1 = {gauss1.name : gauss1}

    model1 = gl.Model(name = 's1', initialFitYields = fitYields1, initialFitComponents = fitComponents1, minVal = 5000, maxVal = 5600)

    fitYields2 = {gauss2.name : gauss2Yield}
    fitComponents2 = {gauss2.name : gauss2}

    model2 = gl.Model(name = 's2', initialFitYields = fitYields2, initialFitComponents = fitComponents2, minVal = 5000, maxVal = 5600)

    model = gl.SimultaneousModel(name = 's', initialFitComponents = {model1.name : model1, model2.name : model2})

    return model, (model1.name, model1), (model2.name, model2)

def simultaneousModelLarge(mean1, width1, width2, a, nSignal, nBkg):

    nSignal1, nSignal2, nSignal3, nSignal4 = nSignal
    nBkg1, nBkg2, nBkg3, nBkg4 = nBkg

    with gl.name_scope('simultaneousModelLarge'):

        max = gl.Parameter(5600., name = 'max', fixed = True)
        min = gl.Parameter(5000., name = 'min', fixed = True)

        m1 = gl.Parameter(mean1, name = 'mean', minVal = 5250., maxVal = 5400.)
        s1 = gl.Parameter(width1, name = 'sigma1', minVal = 0.1, maxVal = width1 * 1.5)
        s2 = gl.Parameter(width2, name = 'sigma2', minVal = 0.1, maxVal = width2 * 1.5)

        aExp = gl.Parameter(a, name = 'a', minVal = -0.05, maxVal = -0.0001)

        with gl.name_scope('model1'):

            with gl.name_scope('sig1'):

                gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

                gauss1Yield = gl.Parameter(nSignal1, name = 'gauss1Yield', minVal = 0.8 * nSignal1, maxVal = 1.2 * nSignal1)

            with gl.name_scope('bkg1'):

                exp1 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp1Yield = gl.Parameter(nBkg1, name = 'exp1Yield', minVal = 0.8 * nBkg1, maxVal = 1.2 * nBkg1)

            model1 = gl.Model(name = 's1', initialFitYields = {gauss1.name : gauss1Yield, exp1.name : exp1Yield}, initialFitComponents = {gauss1.name : gauss1, exp1.name : exp1}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model2'):

            with gl.name_scope('sig2'):

                gauss2 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

                gauss2Yield = gl.Parameter(nSignal2, name = 'gauss2Yield', minVal = 0.8 * nSignal2, maxVal = 1.2 * nSignal2)

            with gl.name_scope('bkg2'):

                exp2 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp2Yield = gl.Parameter(nBkg2, name = 'exp2Yield', minVal = 0.8 * nBkg2, maxVal = 1.2 * nBkg2)

            model2 = gl.Model(name = 's2', initialFitYields = {gauss2.name : gauss2Yield, exp2.name : exp2Yield}, initialFitComponents = {gauss2.name : gauss2, exp2.name : exp2}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model3'):

            with gl.name_scope('sig3'):

                gauss3 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

                gauss3Yield = gl.Parameter(nSignal3, name = 'gauss3Yield', minVal = 0.8 * nSignal3, maxVal = 1.2 * nSignal3)

            with gl.name_scope('bkg3'):

                exp3 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp3Yield = gl.Parameter(nBkg3, name = 'exp3Yield', minVal = 0.8 * nBkg3, maxVal = 1.2 * nBkg3)

            model3 = gl.Model(name = 's3', initialFitYields = {gauss3.name : gauss3Yield, exp3.name : exp3Yield}, initialFitComponents = {gauss3.name : gauss3, exp3.name : exp3}, minVal = 5000, maxVal = 5600)

        with gl.name_scope('model4'):

            with gl.name_scope('sig4'):

                gauss4 = gl.Gaussian({'mean' : m1, 'sigma' : s2})

                gauss4Yield = gl.Parameter(nSignal4, name = 'gauss4Yield', minVal = 0.8 * nSignal4, maxVal = 1.2 * nSignal4)

            with gl.name_scope('bkg4'):

                exp4 = gl.Exponential({'a' : aExp, 'min' : min, 'max' : max})

                exp4Yield = gl.Parameter(nBkg4, name = 'exp4Yield', minVal = 0.8 * nBkg4, maxVal = 1.2 * nBkg4)

            model4 = gl.Model(name = 's4', initialFitYields = {gauss4.name : gauss4Yield, exp4.name : exp4Yield}, initialFitComponents = {gauss4.name : gauss4, exp4.name : exp4}, minVal = 5000, maxVal = 5600)

        model = gl.SimultaneousModel(name = 's', initialFitComponents = {model1.name : model1, model2.name : model2, model3.name : model3, model4.name : model4})

    # Return names so that these can be pulled separately from the sampled data (with scope prefixes)

    return model, ((model1.name, model1), (model2.name, model2), (model3.name, model3), (model4.name, model4))

def doubleGaussianFracModel(mean1, width1, frac, mean2, width2, nEvents):

    with gl.name_scope('doubleGaussianFracModel'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean1, name = 'mean', minVal = 4200., maxVal = 5700.)
            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0., maxVal = width1 * 5.)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter(mean2, name = 'mean', minVal = 4200., maxVal = 5700.)
            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0., maxVal = width2 * 5.)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        gaussFrac = gl.Parameter(frac, name = 'gaussFrac', minVal = 0.0, maxVal = 1.0)
        totalYield = gl.Parameter(nEvents, name = 'totalYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

        fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}
        doubleGaussian = gl.Model(initialFitFracs = {gauss1.name : gaussFrac}, initialFitComponents = fitComponents, minVal = 4200., maxVal = 5700.)

    model = gl.Model(initialFitYields = {doubleGaussian.name : totalYield}, initialFitComponents = {doubleGaussian.name : doubleGaussian}, minVal = 4200., maxVal = 5700.)

    return model

def simpleCBModel(mean, width, aVal, nVal, nEvents):

    with gl.name_scope('simpleCBTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = -1., maxVal = 1.)
        s = gl.Parameter(width, name = 'sigma', minVal = 0.1 * width, maxVal = width * 2.0)
        a = gl.Parameter(aVal, name = 'a', minVal = 0 * aVal, maxVal = 1.5 * aVal)
        n = gl.Parameter(nVal, name = 'n', fixed = True)

        cb = gl.CrystalBall({'mean' : m, 'sigma' : s, 'a' : a, 'n' : n})

        cbYield = gl.Parameter(nEvents, name = 'cbYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {cb.name : cbYield}
    fitComponents = {cb.name : cb}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = -10., maxVal = 10.)

    return model

def constrainedGaussiansModel(mean, width, nEvents1, nEventsOther):

    with gl.name_scope('constrainedGaussiansTest'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean, name = 'mean1', minVal = 4200, maxVal = 5700)
            s1 = gl.Parameter(width, name = 'sigma1', minVal = 0, maxVal = width * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter('m1 + 100.', name = 'mean2', m1 = m1)
            s2 = gl.Parameter('s1 / 2.', name = 'sigma2', s1 = s1)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        with gl.name_scope('gauss3'):

            m3 = gl.Parameter('m1 - 300.', name = 'mean3', m1 = m1)
            s3 = gl.Parameter('s1 * 2.', name = 'sigma3', s1 = s1)

            gauss3 = gl.Gaussian({'mean' : m3, 'sigma' : s3})

        gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)
        gauss2Yield = gl.Parameter(nEventsOther, name = 'gauss2Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)
        gauss3Yield = gl.Parameter(nEventsOther, name = 'gauss3Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)

        fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield, gauss3.name : gauss3Yield}
        fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2, gauss3.name : gauss3}

        model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def externConstrainedGaussiansModel(mean, width, nEvents1, nEventsOther):

    with gl.name_scope('externConstrainedGaussiansTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 5700)

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter('m + 0.', name = 'mean1', m = m)
            s1 = gl.Parameter(width, name = 'sigma1', minVal = 0, maxVal = width * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter('m1 + 100.', name = 'mean2', m1 = m1)
            s2 = gl.Parameter('s1 / 2.', name = 'sigma2', s1 = s1)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        with gl.name_scope('gauss3'):

            m3 = gl.Parameter('m1 - 300.', name = 'mean3', m1 = m1)
            s3 = gl.Parameter('s1 * 2.', name = 'sigma3', s1 = s1)

            gauss3 = gl.Gaussian({'mean' : m3, 'sigma' : s3})

        gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)
        gauss2Yield = gl.Parameter(nEventsOther, name = 'gauss2Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)
        gauss3Yield = gl.Parameter(nEventsOther, name = 'gauss3Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)

        fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield, gauss3.name : gauss3Yield}
        fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2, gauss3.name : gauss3}

        model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def priorGaussiansModel(mean, width, nEvents1, nEventsOther):

    with gl.name_scope('priorGaussiansTest'):

        m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 5700)
        meanConstraint = gl.Gaussian({'mean' : 5400., 'sigma' : 0.01})
        m.priorDistribution = meanConstraint

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter('m + 0.', name = 'mean1', m = m)
            s1 = gl.Parameter(width, name = 'sigma1', minVal = 0, maxVal = width * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter('m1 + 100.', name = 'mean2', m1 = m1)
            s2 = gl.Parameter('s1 / 2.', name = 'sigma2', s1 = s1)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        with gl.name_scope('gauss3'):

            m3 = gl.Parameter('m1 - 300.', name = 'mean3', m1 = m1)
            s3 = gl.Parameter('s1 * 2.', name = 'sigma3', s1 = s1)

            gauss3 = gl.Gaussian({'mean' : m3, 'sigma' : s3})

        gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)
        gauss2Yield = gl.Parameter(nEventsOther, name = 'gauss2Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)
        gauss3Yield = gl.Parameter(nEventsOther, name = 'gauss3Yield', minVal = 0.8 * nEventsOther, maxVal = 1.2 * nEventsOther)

        fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield, gauss3.name : gauss3Yield}
        fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2, gauss3.name : gauss3}

        model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4200, maxVal = 5700)

    return model

def hierarchicalGaussiansModel(mean1, delta1, delta2, delta3, width1, nEvents1, width2, nEvents2, width3, nEvents3):

    with gl.name_scope('hierarchicalGaussiansModel'):

        # constraintWidth = gl.Parameter(1.0, name = 'constraintWidth', minVal = 0.01, maxVal = 20)

        latentMean = gl.Parameter(mean1, name = 'latentMean', minVal = 5100., maxVal = 5500.)

        # deltaConstraint = gl.Gaussian({'mean' : 0.0, 'sigma' : 0.5})
        # deltaConstraint = gl.Gaussian({'mean' : 0.0, 'sigma' : constraintWidth})

        with gl.name_scope('gauss1'):

            deltaConstraint = gl.Gaussian({'mean' : 0.0, 'sigma' : 0.5})

            md1 = gl.Parameter(delta1, name = 'dMean', minVal = -200., maxVal = 200.)
            md1.priorDistribution = deltaConstraint

            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0.1, maxVal = width1 * 1.5, fixed = True)

            mean1 = gl.Parameter('latentMean + md1', name = 'mean', minVal = 5250., maxVal = 5400., latentMean = latentMean, md1 = md1)

            gauss1 = gl.Gaussian({'mean' : mean1, 'sigma' : s1})

            gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield', minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

        with gl.name_scope('gauss2'):

            deltaConstraint = gl.Gaussian({'mean' : 0.0, 'sigma' : 0.5})

            md2 = gl.Parameter(delta2, name = 'dMean', minVal = -200., maxVal = 200.)
            md2.priorDistribution = deltaConstraint

            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0.1, maxVal = width2 * 1.5, fixed = True)

            mean2 = gl.Parameter('latentMean + md2', name = 'mean', minVal = 5250., maxVal = 5400., latentMean = latentMean, md2 = md2)

            gauss2 = gl.Gaussian({'mean' : mean2, 'sigma' : s2})

            gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield', minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

        with gl.name_scope('gauss3'):

            deltaConstraint = gl.Gaussian({'mean' : 0.0, 'sigma' : 0.5})

            md3 = gl.Parameter(delta3, name = 'dMean', minVal = -200., maxVal = 200.)
            md3.priorDistribution = deltaConstraint

            s3 = gl.Parameter(width3, name = 'sigma', minVal = 0.1, maxVal = width3 * 1.5, fixed = True)

            mean3 = gl.Parameter('latentMean + md3', name = 'mean', minVal = 5250., maxVal = 5400., latentMean = latentMean, md3 = md3)

            gauss3 = gl.Gaussian({'mean' : mean3, 'sigma' : s3})

            gauss3Yield = gl.Parameter(nEvents3, name = 'gauss2Yield', minVal = 0.8 * nEvents3, maxVal = 1.2 * nEvents3)

    fitYields1 = {gauss1.name : gauss1Yield}
    fitComponents1 = {gauss1.name : gauss1}

    model1 = gl.Model(name = 's1', initialFitYields = fitYields1, initialFitComponents = fitComponents1, minVal = 5000, maxVal = 5600)

    fitYields2 = {gauss2.name : gauss2Yield}
    fitComponents2 = {gauss2.name : gauss2}

    model2 = gl.Model(name = 's2', initialFitYields = fitYields2, initialFitComponents = fitComponents2, minVal = 5000, maxVal = 5600)

    fitYields3 = {gauss3.name : gauss3Yield}
    fitComponents3 = {gauss3.name : gauss3}

    model3 = gl.Model(name = 's3', initialFitYields = fitYields3, initialFitComponents = fitComponents3, minVal = 5000, maxVal = 5600)

    model = gl.SimultaneousModel(name = 's', initialFitComponents = {model1.name : model1, model2.name : model2, model3.name : model3})

    return model, (model1.name, model1), (model2.name, model2), (model3.name, model3)

def simpleARGausModel(c, p, chi, sigma, nEvents):

    with gl.name_scope('simpleARGausTest'):

        cA = gl.Parameter(c, name = 'c', minVal = 4800., maxVal = 5700.)
        pA = gl.Parameter(p, name = 'p', minVal = 0., maxVal = 10.)
        chiA = gl.Parameter(chi, name = 'chi', minVal = 0., maxVal = 50.)
        sigmaA = gl.Parameter(sigma, name = 'sigma', minVal = 0., maxVal = 50.)

        argaus = gl.ARGaus({'c' : cA, 'chi' : chiA, 'p' :pA, 'sigma' : sigmaA}, minVal = 4800., maxVal = 5700., gridSize = 10000)

        argausYield = gl.Parameter(nEvents, name = 'argausYield', minVal = 0.8 * nEvents, maxVal = 1.2 * nEvents)

    fitYields = {argaus.name : argausYield}
    fitComponents = {argaus.name : argaus}

    model = gl.Model(initialFitYields = fitYields, initialFitComponents = fitComponents, minVal = 4800., maxVal = 5700.)

    return model

def testSimpleGaussian():

    print('testSimpleGaussian')

    # Test generating and fitting back with the same model

    model = simpleGaussianModel(4200., 20., 1E6)
    model.useMinuit()

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussianTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 4200.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 1E6}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testStudentsT():

    model = studentsTModel(5279., 20., 1.5, 1000000.)
    model.useMinuit()

    dataGen = model.sample(minVal = 5000., maxVal = 5600.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 400, log = True)
    plt.savefig('studentsTTest.pdf')
    plt.clf()

    generatedParams = {'studentsTTest/mean' : 5279.,
                       'studentsTTest/sigma' : 20,
                       'studentsTTest/nu' : 1.5,
                       'studentsTTest/studentsTYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianWithExp():

    print('testSimpleGaussianWithExp')

    # Test generating and fitting back with the same model

    model = simpleGaussianWithExpModel(5279., 150., -0.002, 1000000., 2000000.)
    model.useMinuit()

    dataGen = model.sample(minVal = 4200., maxVal = 6000.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    resPlot = plotter.plotDataModel(nDataBins = 300, chiSq = True)
    chiSq = resPlot[-1]

    plt.savefig('simpleGaussianWithExpTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianWithExpTest/gauss/mean' : 5279.,
                       'simpleGaussianWithExpTest/gauss/sigma' : 150,
                       'simpleGaussianWithExpTest/exp/a' : -0.002,
                       'simpleGaussianWithExpTest/expYield' : 2000000.,
                       'simpleGaussianWithExpTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters()) and chiSq > 0.5 and chiSq < 1.5

def testSimpleGaussianWithUniform():

    print('testSimpleGaussianWithUniform')

    model = simpleGaussianWithUniformModel(5279., 150., 1000000., 2000000.)
    model.useMinuit()

    dataGen = model.sample(minVal = 4200., maxVal = 6000.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 300)
    plt.savefig('simpleGaussianWithUniformTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianWithUniformTest/gauss/mean' : 5279.,
                       'simpleGaussianWithUniformTest/gauss/sigma' : 150,
                       'simpleGaussianWithUniformTest/uniYield' : 2000000.,
                       'simpleGaussianWithUniformTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianGen():

    print('testSimpleGaussianGen')

    # Test generating with NumPy and fitting back with a similar model

    model = simpleGaussianModel(5270., 23., 1050000.) # Similar enough to what is generated
    model.useMinuit()

    dataNumpy = np.random.normal(5279, 20, 1000000)
    dataNumpy = dataNumpy[(dataNumpy > 4200) & (dataNumpy < 5700)]

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataNumpy, verbose = True)

    plotter = gl.Plotter(model, dataNumpy)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleGaussianGenTest.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 5279.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 1000000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleGaussianMCMC():

    print('testSimpleGaussianMCMC')

    # Test generating and fitting back with the same model

    model = simpleGaussianModel(5279., 20., 10000.)

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'emcee')
    res = fitter.fit(dataGen, verbose = True, nIterations = 1000, nWalkers = 10)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 30)
    plt.savefig('simpleGaussianMCMCTest.pdf')
    plt.clf()

    fig = plt.figure(figsize = (16, 12))

    samples = res.chain[:, 200:, :].reshape((-1, model.getNFloatingParameters()))
    c = corner.corner(samples, lw = 1.0)
    c.savefig('simpleGaussianMCMCTestCorner.pdf')
    plt.clf()

    generatedParams = {'simpleGaussianTest/mean' : 5279.,
                       'simpleGaussianTest/sigma' : 20,
                       'simpleGaussianTest/gaussYield' : 10000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimpleCB():

    print('testSimpleCB')

    # Test generating and fitting back with the same model

    model = simpleCBModel(0., 1., 1.0, 1.1, 100000.)
    model.useMinuit()

    dataGen = model.sample(minVal = -10, maxVal = 10.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleCBTest.pdf')
    plt.clf()

    generatedParams = {'simpleCBTest/mean' : 0.,
                       'simpleCBTest/sigma' : 1.,
                       'simpleCBTest/a' : 1.0,
                       'simpleCBTest/cbYield' : 100000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testSimultaneousGaussians():

    print('testSimultaneousGaussians')

    simModel, model1, model2 = simultaneousGaussiansModel(5279., 15., 500000, 30., 300000)
    simModel.useMinuit()

    dataGen = simModel.sample(minVal = 5000., maxVal = 5600.)
    fitter = gl.Fitter(simModel, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter1 = gl.Plotter(model1[1], dataGen[model1[0]])
    plotter1.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousGaussiansTest1.pdf')
    plt.clf()

    plotter2 = gl.Plotter(model2[1], dataGen[model2[0]])
    plotter2.plotDataModel(nDataBins = 100)
    plt.xlim(5100, 5500)
    plt.savefig('simultaneousGaussiansTest2.pdf')
    plt.clf()

    generatedParams = {'simultaneousGaussiansModel/mean' : 5279.,
                       'simultaneousGaussiansModel/gauss1/sigma' : 15.,
                       'simultaneousGaussiansModel/gauss2/sigma' : 30.,
                       'simultaneousGaussiansModel/gauss1/gauss1Yield' : 500000.,
                       'simultaneousGaussiansModel/gauss2/gauss2Yield' : 300000.}

    return parameterPullsOkay(generatedParams, simModel.getFloatingParameters())

def testHierarchicalGaussians():

    print('testHierarchicalGaussians')

    # simModel, model1, model2, model3 = hierarchicalGaussiansModel(mean1 = 5279., delta1 = 10., delta2 = -1., delta3 = 2., width1 = 40., nEvents1 = 100., width2 = 20., nEvents2 = 1000., width3 = 10., nEvents3 = 2000.)
    simModel, model1, model2, model3 = hierarchicalGaussiansModel(mean1 = 5279., delta1 = 0., delta2 = 0., delta3 = 0., width1 = 40., nEvents1 = 100., width2 = 20., nEvents2 = 1000., width3 = 10., nEvents3 = 2000.)
    dataGen = simModel.sample(minVal = 4800., maxVal = 5600.)

    simModel.useMinuit()

    # simModel, model1, model2, model3 = hierarchicalGaussiansModel(mean1 = 5279., delta1 = 0., delta2 = 0., delta3 = 0., width1 = 40., nEvents1 = 100., width2 = 20., nEvents2 = 1000., width3 = 10., nEvents3 = 2000.)

    fitter = gl.Fitter(simModel, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    print(res[1])
    # fitter = gl.Fitter(simModel, backend = 'emcee')
    # res = fitter.fit(dataGen, verbose = True, nIterations = 1000, nWalkers = 20)

    print(list(simModel.getFloatingParameters().keys()))
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/latentMean'])
    # pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/constraintWidth'])
    print('')
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss1/dMean'])
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss2/dMean'])
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss3/dMean'])
    print('')
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/latentMean'] + simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss1/dMean'])
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/latentMean'] + simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss2/dMean'])
    pprint(simModel.getFloatingParameters()['hierarchicalGaussiansModel/latentMean'] + simModel.getFloatingParameters()['hierarchicalGaussiansModel/gauss3/dMean'])

    plotter1 = gl.Plotter(model1[1], dataGen[model1[0]])
    plotter1.plotDataModel(nDataBins = 50, minVal = 5100, maxVal = 5400)
    plt.xlim(5100, 5400)
    plt.savefig('hierarchicalGaussiansTest1.pdf')
    plt.clf()

    plotter2 = gl.Plotter(model2[1], dataGen[model2[0]])
    plotter2.plotDataModel(nDataBins = 50, minVal = 5100, maxVal = 5400)
    plt.xlim(5100, 5400)
    plt.savefig('hierarchicalGaussiansTest2.pdf')
    plt.clf()

    plotter3 = gl.Plotter(model3[1], dataGen[model3[0]])
    plotter3.plotDataModel(nDataBins = 50, minVal = 5100, maxVal = 5400)
    plt.xlim(5100, 5400)
    plt.savefig('hierarchicalGaussiansTest3.pdf')
    plt.clf()

    # fig = plt.figure(figsize = (16, 12))
    #
    # samples = res.chain[:, 200:, :].reshape((-1, simModel.getNFloatingParameters()))
    # c = corner.corner(samples, lw = 1.0)
    # c.savefig('hierarchicalGaussiansTestCorner.pdf')
    # plt.clf()

    # generatedParams = {'hierarchicalGaussiansModel/latentMean' : 5279.,
    #                    'hierarchicalGaussiansModel/gauss1/sigma' : 15.,
    #                    'hierarchicalGaussiansModel/gauss2/sigma' : 30.,
    #                    'hierarchicalGaussiansModel/gauss1/gauss1Yield' : 5000.,
    #                    'hierarchicalGaussiansModel/gauss2/gauss2Yield' : 3000.}

    # return parameterPullsOkay(generatedParams, simModel.getFloatingParameters())

    return 0

def testSimultaneousModelLarge():

    print('testSimultaneousModelLarge')

    nSignal = (10000., 10000., 10000., 10000.)
    nBkg = (10000., 10000., 10000., 10000.)

    simModel, (model1, model2, model3, model4) = simultaneousModelLarge(5279., 15., 35., -0.003, nSignal, nBkg)
    simModel.useMinuit()

    dataGen = simModel.sample(minVal = 5000., maxVal = 5600.)

    fitter = gl.Fitter(simModel, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter1 = gl.Plotter(model1[1], dataGen[model1[0]])
    plotter1.plotDataModel(nDataBins = 100)
    plt.xlim(5000, 5600)
    plt.savefig('simultaneousModelLargeTest1.pdf')
    plt.clf()

    plotter2 = gl.Plotter(model2[1], dataGen[model2[0]])
    plotter2.plotDataModel(nDataBins = 100)
    plt.xlim(5000, 5600)
    plt.savefig('simultaneousModelLargeTest2.pdf')
    plt.clf()

    plotter3 = gl.Plotter(model3[1], dataGen[model3[0]])
    plotter3.plotDataModel(nDataBins = 100)
    plt.xlim(5000, 5600)
    plt.savefig('simultaneousModelLargeTest3.pdf')
    plt.clf()

    plotter4 = gl.Plotter(model4[1], dataGen[model4[0]])
    plotter4.plotDataModel(nDataBins = 100)
    plt.xlim(5000, 5600)
    plt.savefig('simultaneousModelLargeTest4.pdf')
    plt.clf()

    v = gl.modelGraphViz(simModel, 'test')

    generatedParams = {'simultaneousModelLarge/a' : -0.003,
                       'simultaneousModelLarge/mean' : 5279.,
                       'simultaneousModelLarge/model1/bkg1/exp1Yield' : 10000,
                       'simultaneousModelLarge/model1/sig1/gauss1Yield' : 10000,
                       'simultaneousModelLarge/model2/bkg2/exp2Yield' : 10000,
                       'simultaneousModelLarge/model2/sig2/gauss2Yield' : 10000,
                       'simultaneousModelLarge/model3/bkg3/exp3Yield' : 10000,
                       'simultaneousModelLarge/model3/sig3/gauss3Yield' : 10000,
                       'simultaneousModelLarge/model4/bkg4/exp4Yield' : 10000,
                       'simultaneousModelLarge/model4/sig4/gauss4Yield' : 10000,
                       'simultaneousModelLarge/sigma1' : 15.,
                       'simultaneousModelLarge/sigma2' : 35. }

    return parameterPullsOkay(generatedParams, simModel.getFloatingParameters())

def testDoubleGaussianYields():

    print('testDoubleGaussianYields')

    model = doubleGaussianYieldsModel(5279., 20., 50000, 5379., 20., 30000)
    model.useMinuit()

    dataGen = model.sample(minVal = 4800., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('doubleGaussianYieldsTest.pdf')
    plt.clf()

    generatedParams = {'doubleGaussianYieldsModel/gauss1/mean' : 5279.,
                       'doubleGaussianYieldsModel/gauss2/mean' : 5379.,
                       'doubleGaussianYieldsModel/gauss1/sigma' : 20.,
                       'doubleGaussianYieldsModel/gauss2/sigma' : 20.,
                       'doubleGaussianYieldsModel/gauss1/gauss1Yield' : 50000.,
                       'doubleGaussianYieldsModel/gauss2/gauss2Yield' : 30000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testDoubleGaussianFrac():

    print('testDoubleGaussianFrac')

    model = doubleGaussianFracModel(5279., 15., 0.75, 5379., 20., 10000.)
    model.useMinuit()

    dataGen = model.sample(minVal = 4800., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('doubleGaussianFracTest.pdf')
    plt.clf()

    generatedParams = {'doubleGaussianFracModel/gauss1/mean' : 5279.,
                       'doubleGaussianFracModel/gauss2/mean' : 5379.,
                       'doubleGaussianFracModel/gauss1/sigma' : 15.,
                       'doubleGaussianFracModel/gauss2/sigma' : 20.,
                       'doubleGaussianFracModel/gaussFrac' : 0.75,
                       'doubleGaussianFracModel/totalYield' : 10000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testDoubleGaussianFracMCMC():

    print('testDoubleGaussianFracMCMC')

    model = doubleGaussianFracModel(5279., 15., 0.75, 5379., 20., 10000.)

    dataGen = model.sample(minVal = 4800., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'emcee')
    try:
        res = fitter.fit(dataGen, verbose = True, nIterations = 1000, nWalkers = 12)
    except:
        # print(model.parameters)
        exit(0)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('doubleGaussianFracMCMCTest.pdf')
    plt.clf()

    fig = plt.figure(figsize = (16, 12))

    samples = res.chain[:, 200:, :].reshape((-1, model.getNFloatingParameters()))
    c = corner.corner(samples, lw = 1.0)
    c.savefig('doubleGaussianFracMCMCTestCorner.pdf')
    plt.clf()

    generatedParams = {'doubleGaussianFracModel/gauss1/mean' : 5279.,
                       'doubleGaussianFracModel/gauss2/mean' : 5379.,
                       'doubleGaussianFracModel/gauss1/sigma' : 15.,
                       'doubleGaussianFracModel/gauss2/sigma' : 20.,
                       'doubleGaussianFracModel/gaussFrac' : 0.75,
                       'doubleGaussianFracModel/totalYield' : 10000.}

    return parameterPullsOkay(generatedParams, model.getFloatingParameters())

def testConstrainedGaussians():

    print('testConstrainedGaussians')

    model = constrainedGaussiansModel(5300., 20., 100., 100000.)
    model.useMinuit()

    dataGen = model.sample(minVal = 4200., maxVal = 5700.)
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('constrainedGaussiansTest.pdf')
    plt.clf()

    generatedParams = {'constrainedGaussiansTest/gauss1/mean1' : 5300.,
                       'constrainedGaussiansTest/gauss1/sigma1' : 20.,
                       'constrainedGaussiansTest/gauss1Yield' : 100.,
                       'constrainedGaussiansTest/gauss2Yield' : 100000.,
                       'constrainedGaussiansTest/gauss3Yield' : 100000.}

    # Important that the two high yield components contribute to the precision, not just fitting
    # the mean of the small one, so check this

    meanErrorOkay = model.getFloatingParameters()['constrainedGaussiansTest/gauss1/mean1'].error < 0.05
    sigmaErrorOkay = model.getFloatingParameters()['constrainedGaussiansTest/gauss1/sigma1'].error < 0.05

    return parameterPullsOkay(generatedParams, model.getFloatingParameters()) and meanErrorOkay and sigmaErrorOkay

def testExternConstrainedGaussians():

    print('testExternConstrainedGaussians')

    model = externConstrainedGaussiansModel(5300., 20., 100., 100000.)
    model.useMinuit()

    dataGen = np.concatenate( (np.random.normal(5400, 30, size = int(100.)),
                              np.random.normal(5500, 15, size = int(100000.)),
                              np.random.normal(5100, 60, size = int(100000.)) ))
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('externConstrainedGaussiansTest.pdf')
    plt.clf()

    generatedParams = {'externConstrainedGaussiansTest/mean' : 5400.,
                       'externConstrainedGaussiansTest/gauss1/sigma1' : 20.,
                       'externConstrainedGaussiansTest/gauss1Yield' : 100.,
                       'externConstrainedGaussiansTest/gauss2Yield' : 100000.,
                       'externConstrainedGaussiansTest/gauss3Yield' : 100000.}

    # Important that the two high yield components contribute to the precision, not just fitting
    # the mean of the small one, so check this

    meanErrorOkay = model.getFloatingParameters()['externConstrainedGaussiansTest/mean'].error < 0.05
    sigmaErrorOkay = model.getFloatingParameters()['externConstrainedGaussiansTest/gauss1/sigma1'].error < 0.05

    return parameterPullsOkay(generatedParams, model.getFloatingParameters()) and meanErrorOkay and sigmaErrorOkay

def testGraphViz():

    print('testGraphViz')

    model = externConstrainedGaussiansModel(5300., 20., 100., 100000.)

    v = gl.modelGraphViz(model, 'test')

    return True

def testPriorGaussians():

    print('testPriorGaussians')

    model = priorGaussiansModel(5300., 20., 100., 100000.)
    model.useMinuit()

    dataGen = np.concatenate( (np.random.normal(5400, 30, size = int(100.)),
                              np.random.normal(5500, 15, size = int(100000.)),
                              np.random.normal(5100, 60, size = int(100000.)) ))
    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('priorGaussiansTest.pdf')
    plt.clf()

    generatedParams = {'priorGaussiansTest/mean' : 5400.,
                       'priorGaussiansTest/gauss1/sigma1' : 20.,
                       'priorGaussiansTest/gauss1Yield' : 100.,
                       'priorGaussiansTest/gauss2Yield' : 100000.,
                       'priorGaussiansTest/gauss3Yield' : 100000.}

    # Test that the constraint makes the error on the mean very small

    meanErrorOkay = model.getFloatingParameters()['priorGaussiansTest/mean'].error < 0.01

    return parameterPullsOkay(generatedParams, model.getFloatingParameters()) and meanErrorOkay

def testSimpleARGaus():

    print('testSimpleARGaus')

    # Test generating and fitting back with the same model

    model = simpleARGausModel(5400., 0.5, 10.0, 30., 100)
    model.useMinuit()

    dataGen = model.sample(minVal = 4800., maxVal = 5700.)

    plt.hist(dataGen, bins = 150)
    plt.savefig('dataHist.pdf')

    fitter = gl.Fitter(model, backend = 'minuit')
    res = fitter.fit(dataGen, verbose = True)

    plotter = gl.Plotter(model, dataGen)
    plotter.plotDataModel(nDataBins = 100)
    plt.savefig('simpleARGausTest.pdf')
    plt.clf()

if __name__ == '__main__':

    print(testSimultaneousModelLarge())
    # print(testSimpleARGaus())
    # print(testHierarchicalGaussians())
    # print(testDoubleGaussianFracMCMC())
