Glasnost
===

Minimalist maximum likelihood fitter in (mostly) NumPy, with extensibility for Bayesian MCMC fits. Direct access to parameters hopefully simplifies simultaneous fits where parameters are shared, and additional constraints are implemented directly into the likelihood.

Model specification
---
Model specification is done within name_scopes (a la TensorFlow), such that each model component and parameter is named according to the scope in which it resides, but this doesn't have to be manually constructed by the user.

``` python

with gl.name_scope("massFit"):
    with gl.name_scope("coreGaussian"):
        m = gl.Parameter(name = 'mean') # Full name is 'massFit/coreGaussian/mean'
        s = gl.Parameter(name = 'sigma') # Full name is 'massFit/coreGaussian/sigma'

        gCore = gl.Gaussian(m, s) # Full name is 'massFit/coreGaussian'
`
