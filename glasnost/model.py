from glasnost.distribution import Distribution

from iminuit.util import Struct

import numpy as np

from collections import OrderedDict

class Model(Distribution):

    """

    Class corresponding to a composite likelihood model. Inherits from Distribution (implements prob
    and log-prob functions). Initialised with yields (dictionary of names to Parameters) and fit
    components (dictionary of names to Distributions). By default, assume extended maximum likelihood
    fit, where all input distributions are summed.

    """

    def __init__(self, initialFitYields = None, initialFitComponents = None, data = None, simultaneous = False, name = ''):

        super(Model, self).__init__(name)

        # Dictionary of distribution names to yield Parameters
        self.fitYields = initialFitYields

        # If this is a model for a simultaneous fit to multiple datasets - in that case data is expected to be a list
        # of datasets, and in conjunction with 'initialFitYields = None' so that each model is equally weighted in the
        # log-likelihood
        self.simultaneous = simultaneous

        # Dictionary of distribution names to distributions
        self.fitComponents = initialFitComponents

        self.fitComponentParameterNames = {}

        for componentName, component in initialFitComponents.items():
            self.fitComponentParameterNames[componentName] = component.getParameterNames()

        self.parameters = {}

        for component in initialFitComponents.values():
            for parameter in component.getParameters().values():
                self.parameters[parameter.name] = parameter

        if initialFitYields:
            for y in initialFitYields.values():
                self.parameters[y.name] = y

        self.data = data

        floatingParameterNames = self.getFloatingParameterNames()

        # For iminuit's parameter introspection
        # Gets screwed up if parameters are changed between fixed and floating
        # Make sure this is propagated (somehow?)

        self.func_code = Struct(co_varnames = floatingParameterNames,
                                co_argcount = len(floatingParameterNames)
                                )

    # Only floating
    def getComponentFloatingParameterNames(self):

        names = []

        for c in self.fitComponents.values():

            names += list(map(lambda x : x, c.getFloatingParameterNames()))

        return names

    def getTotalYield(self):
        return np.sum(list(self.fitYields.values())) if self.fitYields else 0.

    def getFloatingParameterNames(self):

        names = set(self.getComponentFloatingParameterNames())

        if self.fitYields:
            # Add yields from the model
            for y in self.fitYields.values():
                if not y.isFixed:
                    names.add(y.name)

        return sorted(list(names))

    def getFloatingParameterValues(self):

        values = {}

        # If we're simultaneous, we have no parameters here
        if self.fitYields and not self.simultaneous:
            for y in self.fitYields.values():
                values[y.name] = y.value

        for c in self.fitComponents.values():
            for v in c.getParameters().values():
                values[v.name] = v.value

        return values

    def parameterRangeLnPriors(self):

        # Fill prior for ranges, doesn't depend on data

        if any([y.value_ > y.max or y.value_ < y.min for y in self.fitYields.values()]):
            return -np.inf

        # Use np.zeros(1) as dummy data -> won't be used anyway if it's inf (which is true for
        # parameter range priors)
        if any([np.isinf(c.lnprior(np.zeros(1))) for c in self.fitComponents.values()]):
            return -np.inf

        return 0

    def prob(self, data):

        return np.exp(self.lnprob(data))

    def lnprob(self, data):

        # This assumes that the total likelihood is a sum over components

        # COPIES of dictionary values
        # In future: https://docs.python.org/2/library/stdtypes.html#dictionary-view-objects
        components = list(self.fitComponents.values())

        # Explicitly use y.value_ otherwise this fills the parameter with an array
        # Would be nice only to use the Parameter operations when specified
        # FIX ME!

        # If no yields are given, assume that all are weighted equally (for example if the components
        # are each individual models for a particular dataset)

        # yields = list([y.value_ for y in self.fitYields.values()]) if self.fitYields else [1. for i in range(len(components))]

        yields = None
        if self.fitYields:
            yields = {n : y.value_ for n, y in self.fitYields.items()}
        else:
            yields = {c.name : 1.0 for c in self.fitComponents.values()}

        # If simultaneous, get probVal to avoid having mismatched data lengths (for each dataset),
        # we don't need the prob except for minimisation

        if self.simultaneous:

            # Already lnProb - just sum these to get the total likelihood

            z = np.vstack([ components[i].lnprobVal(data[i]) for i in range(len(components)) ])
            return np.sum(z)

        else:
            # z = [ yields[i] * components[i].prob(data) for i in range(len(components)) ]
            z = [ yields[component.name] * component.prob(data) for component in components ]

            # Matrix of (nComponents, nData) -> uses lots of memory, rewrite using einsum?
            p = np.vstack(z)

            # Sum across component axis, vector of length nData
            p = np.sum(p, 0)

            # normalise
            # p *= (1. / np.sum(yields))
            p *= (1. / np.sum(list(yields.values())))

            # Take log of each component, (sum over data axis to get total log-likelihood)
            p = np.log(p)

            return p

    def probVal(self, data):

        return np.exp(self.lnprobVal(data))

    def lnprobVal(self, data):

        if self.simultaneous:
            # No EML
            return self.lnprob(data)

        # With EML criteria

        nObs = len(data)
        totalYield = self.getTotalYield()

        lnPriors = self.parameterRangeLnPriors()

        # Short circuit to avoid returning nan
        if np.isinf(lnPriors):
            return -np.inf

        lnP = np.sum(self.lnprob(data) + self.lnprior(data)) + nObs * np.log(totalYield) - totalYield

        lnP += self.parameterRangeLnPriors()

        return lnP

    def setData(self, data):

        # For using __call__ with no data

        self.data = data

    def getData(self, data):

        if self.hasData:
            return self.data
        else:
            return None

    def getNFloatingParameters(self):
        return len(self.getFloatingParameterNames())

    @property
    def hasData(self):

        return self.data is not None

    def getInitialParameterValues(self):

        # Return initial parameters so that __call__ can be called initially with the correct number
        # and with the parameters in the correct order

        params = self.getFloatingParameterValues()

        return params

    def getInitialParameterValuesAndStepSizes(self):

        out = self.getFloatingParameterValues()

        # Maybe one day set this more intelligently

        for k, v in self.getFloatingParameterValues().items():
            out['error_' + k] = abs(0.1 * v) if 'yield' not in k else abs(1.)

        out = OrderedDict(sorted(out.items(), key = lambda x : x[0]))

        return out

    def __call__(self, *params):

        # params is a tuple of parameter values according to the ordering given by getFloatingParameterNames()

        if self.getNFloatingParameters() != len(params):
            print('Number of parameters differs from the number of floating parameters of the model.')
            exit(1)

        names = self.getFloatingParameterNames()
        # print(names)
        # Check that this ordering will always be maintained - could be buggy

        for i, n in enumerate(names):
            self.parameters[names[i]].updateValue(params[i])

        return -self.lnprobVal(self.data)
