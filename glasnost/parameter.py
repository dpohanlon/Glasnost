import numpy as np

import re

import glasnost as gl

class Parameter(np.lib.mixins.NDArrayOperatorsMixin, object):

    """

    Parameter class for storing parameter information. This includes the name, value, ranges, whether the
    parameter is fixed, and the prior probability distribution of the parameter. Contains member functions
    to access and modify the value of the parameter.

    """

    def __init__(self, value, name = '', minVal = None, maxVal = None, fixed = False, priorDistribution = None, **kwargs):

        self.name_ = gl.utils.nameScope + name

        self.kw = kwargs

        self.initialValue = value

        self.derived_ = type(value) == str

        if self.derived_:
            value = self.transformToInternalRep(value)

        self.transform_ = compile(value, '', 'eval') if self.derived_ else None

        self.value_ = self.initialValue if not self.derived_ else eval(self.transform_)

        self.priorDistribution = priorDistribution

        self.fixed_ = fixed or (minVal == None and maxVal == None) or self.transform_ != None

        # Leave these unconstrained unless we have to have strict minima and maxima

        self.min = minVal #if minVal is not None else self.initialValue - 3. * self.initialValue
        self.max = maxVal #if maxVal is not None else self.initialValue + 3. * self.initialValue

        # Can envision blinding, errors, etc

        self.error_ = None

        self.rangeOK = self.testRangeOK()

    def __repr__(self):
        if not self.error_:
            if self.fixed_:
                return "%s: %s" % (self.name, self.value)
            else:
                return "%s: %s, [%s, %s]" % (self.name, self.value, self.min, self.max)
        else:
            if self.fixed_:
                return "%s: %s +/- %s" % (self.name, self.value, self.error_)
            else:
                return "%s: %s +/- %s, [%s, %s]" % (self.name, self.value, self.error_, self.min, self.max)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Magic function to catch numpy array operations and decay the Parameter to the base
        # float representation first (using np.lib.mixins.NDArrayOperatorsMixin)

        # Convert Parameter instances to their value_ equivalent before performing numpy
        # operations

        inputs = [i.value if isinstance(i, Parameter) else i for i in inputs]

        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def name(self):
        return self.name_

    @property
    def value(self):
        if self.derived_ == False:
            return self.value_
        else:
            return eval(self.transform_).value

    def testRangeOK(self):
        return not ( (self.min and self.value < self.min) or (self.max and self.value > self.max) )

    def updateValue(self, value):
        self.value_ = value
        self.rangeOK = self.testRangeOK()

    @property
    def error(self):
        return self.error_

    def updateError(self, error):
        self.error_ = error

    @property
    def isFixed(self):
        return self.fixed_

    def __float__(self):
        return self.value

    def isNumeric(self, t):
        return (type(t) == float) or (type(t) == int)

    def transformToInternalRep(self, rep):
        for v in self.kw.keys():
            rep = re.sub(v, "self.kw['" + v + "']", rep)

        return rep

    def lnPrior(self):
        if self.priorDistribution:
            return self.priorDistribution.lnprob(self.value)
        else:
            return 0.

    # Built in Parameter operations that can return Parameters. If numpy operations, the operation gets
    # deferred to __array_ufunc__ to return floats/numpy arrays

    # These are intended to operate on scalars ('numeric') only, to improve performance

    def __neg__(self):
        return Parameter(-self.value, name = self.name + '-add-float')

    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value + other.value, name = self.name + '-add-' + other.name)
        elif self.isNumeric(other):
            return Parameter(self.value + other, name = self.name + '-add-float')
        else:
            return self.value + other

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value * other.value, name = self.name + '-mul-' + other.name)
        elif self.isNumeric(other):
            return Parameter(self.value * other, name = self.name + '-mul-float')
        else:
            return self.value * other

    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value - other.value, name = self.name + '-sub-' + other.name)
        elif self.isNumeric(other):
            return Parameter(self.value - other, name = self.name + '-sub-float')
        else:
            return self.value - other

    # Not == __sub__!
    def __rsub__(self, other):
            return Parameter(other - self.value_, name = self.name + '-rsub-float')

    def __div__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value / other.value, name = self.name + '-div-' + other.name)
        elif self.isNumeric(other):
            return Parameter(self.value / other, name = self.name + '-div-float')
        else:
            return self.value / other

    __truediv__ = __div__
    __floordiv__ = __div__

    # Not == __div__!
    def __rdiv__(self, other):
            return Parameter(other / self.value_, name = self.name + '-rdiv-float')

    def __pow__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value ** other.value, name = self.name + '-pow-' + other.name)
        elif self.isNumeric(other):
            return Parameter(self.value ** other, name = self.name + '-pow-float')

    def __lt__(self, other):
        if isinstance(other, Parameter):
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        if isinstance(other, Parameter):
            return self.value > other.value
        else:
            return self.value > other

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.value == other.value
        else:
            return self.value == other
