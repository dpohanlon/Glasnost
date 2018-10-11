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

    def __init__(self, initialFitYields = {}, initialFitComponents = {}, initialFitFracs = {}, minVal = None, maxVal = None, data = None, name = ''):

        super(Model, self).__init__(name = name)

        if initialFitFracs and initialFitYields:
            print('Use either fit fractions (of size nComponents-1), or fit yields (of size nComponents).')
            exit(1)

        if initialFitFracs and len(initialFitFracs) != len(initialFitComponents) - 1:
            print('When using fit fractions, the total number of fractions must be nComponents-1 (to enforce the sum-to-one requirement).')
            exit(1)

        if initialFitYields and len(initialFitYields) != len(initialFitComponents):
            print('When using fit yields, the total number of fractions must be nComponents.')
            exit(1)

        # If neither fitYields or fitFracs are filled, assume the weighting in the likelihood is democratic

        # Dictionary of distribution names to yield Parameters
        self.fitYields = initialFitYields

        # Dictionary of distribution names to frac Parameters
        self.fitFracs = initialFitFracs

        # Dictionary of distribution names to distributions
        self.fitComponents = initialFitComponents

        self.fitComponentParameterNames = {}

        for componentName, component in initialFitComponents.items():
            self.fitComponentParameterNames[componentName] = component.getFloatingParameterNames()

        self.parameters = {}

        self.min = minVal
        self.max = maxVal

        for component in initialFitComponents.values():
            for parameter in component.getParameters().values():
                self.parameters[parameter.name] = parameter

        if initialFitYields:
            for y in initialFitYields.values():
                self.parameters[y.name] = y

        if initialFitFracs:
            for y in initialFitFracs.values():
                self.parameters[y.name] = y

        self.data = data

        self.totalYield_ = 0.

        self.floatingParameterNames = self.getFloatingParameterNames()

        # For iminuit's parameter introspection
        # Gets screwed up if parameters are changed between fixed and floating
        # Make sure this is propagated (somehow?)

        self.func_code = Struct(co_varnames = self.floatingParameterNames,
                                co_argcount = len(self.floatingParameterNames)
                                )

    # Only floating
    def getComponentFloatingParameterNames(self):

        names = []

        for c in self.fitComponents.values():

            names += list(map(lambda x : x, c.getFloatingParameterNames()))

        return names

    def setTotalYield(self, y):
        self.totalYield_ = y

    def testFracsSumToOne(self):

        s = np.sum(self.fitFracs.values())

        return np.isclose(s, 1.0)

    def getTotalFracs(self):
        return np.sum(list(self.fitFracs.values())) if self.fitFracs else 0.

    def getTotalYield(self):
        return np.sum(list(self.fitYields.values())) if self.fitYields else self.totalYield_

    def getFloatingParameterNames(self):

        names = set(self.getComponentFloatingParameterNames())

        if self.fitYields:
            # Add yields from the model
            for y in self.fitYields.values():
                if not y.isFixed:
                    names.add(y.name)

        if self.fitFracs:
            # Add fracs from the model
            for y in self.fitFracs.values():
                if not y.isFixed:
                    names.add(y.name)

        return sorted(list(names))

    def getFloatingParameterValues(self):

        values = {}

        if self.fitYields:
            for y in self.fitYields.values():
                if not y.isFixed:
                    values[y.name] = y.value

        if self.fitFracs:
            for y in self.fitFracs.values():
                if not y.isFixed:
                    values[y.name] = y.value

        for c in self.fitComponents.values():
            for v in c.getParameters().values():
                if not v.isFixed:
                    values[v.name] = v.value

        return values

    def getFloatingParameters(self):

        values = {}

        if self.fitYields:
            for y in self.fitYields.values():
                if not y.isFixed:
                    values[y.name] = y

        if self.fitFracs:
            for y in self.fitFracs.values():
                if not y.isFixed:
                    values[y.name] = y

        for c in self.fitComponents.values():
            for v in c.getParameters().values():
                if not v.isFixed:
                    values[v.name] = v

        return values

    def parameterRangeLnPriors(self):

        # Fill prior for ranges, doesn't depend on data
        # (Only if these ranges aren't none)

        if any([(y.min and y.value_ < y.min) or (y.max and y.value_ > y.max) for y in self.fitYields.values()]):
            return -np.inf

        if any([(y.min and y.value_ < y.min) or (y.max and y.value_ > y.max) for y in self.fitFracs.values()]):
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

        lnPriors = self.parameterRangeLnPriors()

        # Short circuit to avoid returning nan
        if np.isinf(lnPriors):
            return np.full_like(data, -np.inf)

        yields = list([y.value_ for y in self.fitYields.values()]) if self.fitYields else [1. for i in range(len(components))]

        # Call this yields, but really it can represent either yields or fracs
        yields = None
        if self.fitYields:

            yields = {n : y.value_ for n, y in self.fitYields.items()}

        elif self.fitFracs:

            # Get the fracs for those provided
            fracsPresent = set([c.name for c in components])
            yields = {}

            for n, f in self.fitFracs.items():
                yields[n] = f.value_
                fracsPresent.remove(n)

            if len(fracsPresent) != 1:
                print('Number of fracs with no value is not one, something is wrong here.')
                exit(1)

            sumFracs = np.sum(list(yields.values()))
            yields[fracsPresent.pop()] = 1. - sumFracs

        else:
            # equal weight
            yields = {c.name : (1.0 / len(components)) for c in components}

        # totalNorm = self.integral(self.min, self.max)
        # norm = {n : c / totalNorm for (n, c) in self.getComponentIntegrals(self.min, self.max).items()}
        norm = {n : c for (n, c) in self.getComponentIntegrals(self.min, self.max).items()}
        totalYield = np.sum(list(yields.values()))

        # z = [ (1. / norm[component.name]) * yields[component.name] * component.prob(data) for component in components ]
        z = [ (yields[component.name] / (norm[component.name] * totalYield)) * component.prob(data) for component in components ]

        # Matrix of (nComponents, nData) -> uses lots of memory, rewrite using einsum?
        p = np.vstack(z)

        # Sum across component axis, vector of length nData
        p = np.sum(p, 0)

        # Take log of each component, (sum over data axis to get total log-likelihood)
        p = np.log(p)

        return p

    def probVal(self, data):

        return np.exp(self.lnprobVal(data))

    def lnprobVal(self, data):

        lnPriors = self.parameterRangeLnPriors()

        # Short circuit to avoid returning nan
        if np.isinf(lnPriors):
            return -np.inf

        lnP = np.sum(self.lnprob(data) + self.lnprior(data))
        lnP += lnPriors

        if self.fitYields:

            # With EML criteria

            nObs = len(data)
            totalYield = self.getTotalYield()

            lnP += nObs * np.log(totalYield) - totalYield

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

        # Set limits also, rather than using the prior
        # (This is probably slow, try and merge with above as it gets similar things)

        for c in self.fitComponents.values():
            for v in c.getParameters().values():
                if not v.isFixed:
                    if v.min != None and v.max != None :
                        out['limit_' + v.name] = (v.min, v.max)
                    elif v.min != None:
                        out['limit_' + v.name] = (v.min, abs(v.value_) * 100.)
                    elif v.max != None:
                        out['limit_' + v.name] = (-abs(v.value_) * 100, v.max)

        for v in list(self.fitYields.values()) + list(self.fitFracs.values()):
            if not v.isFixed:
                if v.min != None and v.max != None :
                    out['limit_' + v.name] = (v.min, v.max)
                elif v.min != None:
                    out['limit_' + v.name] = (v.min, abs(v.value_) * 100.)
                elif v.max != None:
                    out['limit_' + v.name] = (-abs(v.value_) * 100, v.max)

        # Might be slow - try another way
        out = OrderedDict(sorted(out.items(), key = lambda x : x[0]))

        return out

    def getComponentIntegrals(self, minVal, maxVal):
        return {name : c.integral(minVal, maxVal) for (name, c) in self.fitComponents.items()}

    def integral(self, minVal, maxVal):
        # Operate on already normalised PDFs
        # Screws up normalisation over range?!?!????!?!?
        # return 1.0

        return np.sum(list(self.getComponentIntegrals(minVal, maxVal).values()))

    def sample(self, nEvents = None, minVal = None, maxVal = None):
        # Generate according to yields and component models
        # Pass min and max ranges - ideally these would be separate for each 1D fit

        components = list(self.fitComponents.values())

        if not self.fitYields and nEvents:

            # Construct yields from fractions and total nEvents

            fracs = {n : f.value for n, f in self.fitFracs.items()}

            sumOfFracs = np.sum(list(fracs.values()))

            if len(fracs) != len(self.fitComponents) - 1:
                print("Too few fracs for components!")

            for component in components:
                if component.name not in fracs:
                    fracs[component.name] = 1. - sumOfFracs
                    break

            yields = {component.name : fracs[component.name] * nEvents for component in components}

            z = np.concatenate([ component.sample(yields[component.name], minVal = minVal, maxVal = maxVal) for component in components ])

            return z

        else:

            z = np.concatenate([ component.sample(self.fitYields[component.name].value, minVal = minVal, maxVal = maxVal) for component in components ])

            return z

    def logL(self, params): # for emcee

        # params is a tuple of parameter values according to the ordering given by getFloatingParameterNames()

        if self.getNFloatingParameters() != len(params):
            print('Number of parameters passed (%s) differs from the number of floating parameters of the model (%s).' %(len(params), self.getNFloatingParameters()))
            exit(1)

        names = self.floatingParameterNames

        # Check that this ordering will always be maintained - could be buggy

        for i, n in enumerate(names):
            self.parameters[n].updateValue(params[i])

        return self.lnprobVal(self.data)

    def __call__(self, *params):

        # params is a tuple of parameter values according to the ordering given by getFloatingParameterNames()

        if self.getNFloatingParameters() != len(params):
            print('Number of parameters passed (%s) differs from the number of floating parameters of the model (%s).' %(len(params), self.getNFloatingParameters()))
            exit(1)

        names = self.getFloatingParameterNames()

        # Check that this ordering will always be maintained - could be buggy

        for i, n in enumerate(names):
            self.parameters[names[i]].updateValue(params[i])

        return -self.lnprobVal(self.data)
