from glasnost.distribution import Distribution

from iminuit.util import Struct

import numpy as np

class Model(Distribution):

    """

    Class corresponding to a composite likelihood model. Inherits from Distribution (implements prob
    and log-prob functions). Initialised with yields (dictionary of names to Parameters) and fit
    components (dictionary of names to Distributions). By default, assume extended maximum likelihood
    fit, where all input distributions are summed.

    """

    def __init__(self, initialFitYields = None, initialFitComponents = None, data = None, name = ''):

        super(Model, self).__init__(name)

        # Dictionary of distribution names to yield Parameters

        self.fitYields = initialFitYields

        # Dictionary of distribution names to distributions
        self.fitComponents = initialFitComponents

        self.fitComponentParameterNames = {}

        for componentName, component in initialFitComponents.items():
            self.fitComponentParameterNames[componentName] = component.getParameterNames()

        self.parameters = {}

        for component in initialFitComponents.values():
            for parameter in component.getParameters().values():
                self.parameters[parameter.name] = parameter

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

    def getFloatingParameterNames(self):

        names = self.getComponentFloatingParameterNames()

        # Add yields from the model
        for y in self.fitYields.values():

            if y.isFixed:
                continue

            names.append(y.name)

        return names

    def getFloatingParameterValues(self):

        values = {}

        for y in self.fitYields.values():
            values[y.name] = y.value

        for c in self.fitComponents.values():
            for v in c.getParameters().values():
                values[v.name] = v.value

        return values

    def prob(self, data):

        return np.exp(self.lnprob(data))

    def lnprob(self, data):

        # This assumes that the total likelihood is a sum over components

        nObs = len(data)
        totalYield = np.sum(list(self.fitYields.values()))

        # COPIES of dictionary values
        # In future: https://docs.python.org/2/library/stdtypes.html#dictionary-view-objects
        components = list(self.fitComponents.values())

        # Explicitly use y.value_ otherwise this fills the parameter with an array
        # Would be nice only to use the Parameter operations when specified
        # FIX ME!

        yields = list([y.value_ for y in self.fitYields.values()])

        # Matrix of (nComponents, nData) -> uses lots of memory, rewrite using einsum?
        p = np.vstack([ yields[i] * components[i].prob(data) for i in range(len(components)) ])

        # Sum across component axis, vector of length nData
        p = np.sum(p, 0)

        # Take log of each component, (sum over data axis to get total log-likelihood)
        p = np.log(p)

        return p

    def probVal(self, data):

        return np.exp(self.lnprobVal(data))

    def lnprobVal(self, data):

        # With EML criteria

        nObs = len(data)
        totalYield = np.sum(list(self.fitYields.values()))

        return np.sum(self.lnprob(data)) + nObs * np.log(totalYield) - totalYield

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

        return self.getFloatingParameterValues()

    def getInitialParameterValuesAndStepSizes(self):
        out = self.getFloatingParameterValues()

        # Maybe one day set this more intelligently

        for k, v in self.getFloatingParameterValues().items():
            out['error_' + k] = 0.1 * v

        return out

    def __call__(self, **params):

        # params is a dictionary of parameter names to floats (representing the initial configuration)

        if self.getNFloatingParameters() != len(params):
            print('Number of parameters differs from the number of floating parameters of the model.')
            exit(1)

        for n, v in params.items():
            self.parameters[n].updateValue(v)

        return self.lnprobVal(self.data)
