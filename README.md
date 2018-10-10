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
