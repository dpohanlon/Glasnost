Glasnost [![Build Status](https://travis-ci.com/dpohanlon/Glasnost.svg?token=6U7ubPocKxuFEpjJP4aK&branch=master)](https://travis-ci.com/dpohanlon/Glasnost) [![codecov](https://codecov.io/gh/dpohanlon/Glasnost/branch/master/graph/badge.svg?token=THBlL3wY3b)](https://codecov.io/gh/dpohanlon/Glasnost)
===

Minimalist maximum likelihood fitter in (mostly) NumPy, with extensibility for Bayesian MCMC fits. Direct access to parameters hopefully simplifies simultaneous fits where parameters are shared, and additional constraints are implemented directly into the likelihood.

Model specification
---
Model specification is done within name_scopes (à la TensorFlow), such that each model component and parameter is named according to the scope in which it resides, but this doesn't have to be manually constructed by the user.

``` python

with gl.name_scope("massFit"):
    with gl.name_scope("coreGaussian"):
        m = gl.Parameter(name = 'mean') # Full name is 'massFit/coreGaussian/mean'
        s = gl.Parameter(name = 'sigma') # Full name is 'massFit/coreGaussian/sigma'

        gCore = gl.Gaussian(m, s) # Full name is 'massFit/coreGaussian'
```

Composite models can be specifed using relative normalisations

``` python

def doubleGaussianFracModel(mean1, width1, frac, mean2, width2, nEvents):

    with gl.name_scope('doubleGaussianFracModel'):

        with gl.name_scope('gauss1'):

            m1 = gl.Parameter(mean1, name = 'mean', minVal = 4200, maxVal = 5700)
            s1 = gl.Parameter(width1, name = 'sigma', minVal = 0, maxVal = width1 * 5)

            gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

        with gl.name_scope('gauss2'):

            m2 = gl.Parameter(mean2, name = 'mean', minVal = 4200, maxVal = 5700)
            s2 = gl.Parameter(width2, name = 'sigma', minVal = 0, maxVal = width2 * 5)

            gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

        gaussFrac = gl.Parameter(frac, name = 'gaussFrac', minVal = 0.0, maxVal = 1.0)
        totalYield = gl.Parameter(nEvents, name = 'totalYield',
                                  minVal = 0.8 * nEvents,
                                  maxVal = 1.2 * nEvents)

    fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}
    doubleGaussian = gl.Model(initialFitFracs = {gauss1.name : gaussFrac},
                              initialFitComponents = fitComponents,
                              minVal = 5280, maxVal = 5700)

    model = gl.Model(initialFitYields = {doubleGaussian.name : totalYield},
                                         initialFitComponents = {doubleGaussian.name : doubleGaussian},
                                         minVal = 5280, maxVal = 5700)

    return model
    
```

or with explicit yields

``` python

with gl.name_scope('doubleGaussianYieldsModel'):

...

    with gl.name_scope('gauss1'):
    
        ...

        gauss1Yield = gl.Parameter(nEvents1, name = 'gauss1Yield',
                                   minVal = 0.8 * nEvents1, maxVal = 1.2 * nEvents1)

    with gl.name_scope('gauss2'):

        ...

        gauss2Yield = gl.Parameter(nEvents2, name = 'gauss2Yield',
                                   minVal = 0.8 * nEvents2, maxVal = 1.2 * nEvents2)

fitYields = {gauss1.name : gauss1Yield, gauss2.name : gauss2Yield}
fitComponents = {gauss1.name : gauss1, gauss2.name : gauss2}

model = gl.Model(initialFitYields = fitYields,
                 initialFitComponents = fitComponents,
                 minVal = 5300, maxVal = 5700)


```

Simultaneous models
---

Simultaneous fits to multiple (one-dimensional) spectra can be performed using ```SimultaneousModel``` and passing a dictionary of models and yield parameters.

``` python

fitYields1 = {gauss1.name : gauss1Yield}
fitComponents1 = {gauss1.name : gauss1}

model1 = gl.Model(name = 's1',
                  initialFitYields = fitYields1,
                  initialFitComponents = fitComponents1,
                  minVal = 5000, maxVal = 5600)

fitYields2 = {gauss2.name : gauss2Yield}
fitComponents2 = {gauss2.name : gauss2}

model2 = gl.Model(name = 's2',
                  initialFitYields = fitYields2,
                  initialFitComponents = fitComponents2,
                  minVal = 5000, maxVal = 5600)

model = gl.SimultaneousModel(name = 's',
                             initialFitComponents = {model1.name : model1, model2.name : model2})

```

Constraints
---

Constraints on model parameters can either be external, by providing a prior distribution on the parameter that reduces to a float passed at construction time

``` python

m = gl.Parameter(mean, name = 'mean', minVal = 4200, maxVal = 5700)
meanConstraint = gl.Gaussian({'mean' : 5400., 'sigma' : 0.01})
m.priorDistribution = meanConstraint # Constrain the parameter according to a Gaussian distribution

```

They can also be defined relative to another model parameter, by passing an arithmetic function string that describes a transformation of the other parameter, and the definition of the parameter, to the constructor

``` python

with gl.name_scope('gauss1'):

    m1 = gl.Parameter('m', name = 'mean1')
    s1 = gl.Parameter(width, name = 'sigma1', minVal = 0, maxVal = width * 5)

    gauss1 = gl.Gaussian({'mean' : m1, 'sigma' : s1})

with gl.name_scope('gauss2'):

    m2 = gl.Parameter('m1 + 100.', name = 'mean2', m1 = m1) # m2 is defined to be m1 + 100
    s2 = gl.Parameter('s1 / 2.', name = 'sigma2', s1 = s1) # s2 is defined to be s1 / 2

    gauss2 = gl.Gaussian({'mean' : m2, 'sigma' : s2})

```

Inference
---

Model parameters can be inferred from the data using ```Fitter```, which takes backend arguments of ```minuit``` for maximum likelihood fitting with Minuit, or ```emcee``` for Bayesian MCMC with Emcee.

``` python

fitterML = gl.Fitter(model, backend = 'minuit') # Maximum likelihood
fitterBayes = gl.Fitter(model, backend = 'emcee') # MCMC

resML = fitterML.fit(data)
resBayes = fitterBayes.fit(data, nIterations = 1000, nWalkers = 10)

```

By default, the maxium likelihood value or the maximum of the posterior distribution for each model parameter can be accessed using ```param.value``` and the 68% interval around the maximum likelihood value (or the maxium of the posterior) can be accessed using ```param.error```. For Bayesian MCMC, the MCMC chains can be accessed using ```res.chain```

```python

samples = res.chain[:, 200:, :].reshape((-1, model.getNFloatingParameters()))

```

Plotting
---

Plotting functionality is provided by the ```Plotter``` class, which plots normalised models on data histograms

```python

plotter = gl.Plotter(model, data)
plotter.plotDataModel(nDataBins = 30)

```

Supported distributions
---
* Gaussian
* Uniform
* Crystal-Ball
* Exponential
* Student's t
* Beta
