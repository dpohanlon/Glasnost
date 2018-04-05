import numpy as np

import glasnost as gl

class Parameter(np.lib.mixins.NDArrayOperatorsMixin, object):

    """

    Parameter class for storing parameter information. This includes the name, value, ranges, whether the
    parameter is fixed, and the prior probability distribution of the parameter. Contains member functions
    to access and modify the value of the parameter.

    """

    def __init__(self, initialValue, name = '', minVal = None, maxVal = None, fixed = False, priorDistribution = None):

        self.name_ = gl.utils.nameScope + name

        self.initialValue = initialValue
        self.value_ = self.initialValue

        self.min = minVal if minVal else self.initialValue - 3. * self.initialValue
        self.max = maxVal if maxVal else self.initialValue + 3. * self.initialValue

        self.priorDistribution = priorDistribution

        self.fixed_ = False

        # Can envision blinding, errors, etc

    def __repr__(self):
        return "%s: %s" % (self.name, self.value_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Magic function to catch numpy array operations and decay the Parameter to the base
        # float representation first (using np.lib.mixins.NDArrayOperatorsMixin)

        # Convert Parameter instances to their value_ equivalent before performing numpy
        # operations

        inputs = [i.value_ if isinstance(i, Parameter) else i for i in inputs]

        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def name(self):
        return self.name_

    @property
    def value(self):
        if self.priorDistribution == None:
            return self.value_
        else:
            return self.priorDistribution.randomSample() # TODO: Implement me

    def updateValue(self, value):
        self.value_ = value

    @property
    def isFixed(self):
        return self.fixed_

    def __float__(self):
        return self.value_

    # Built in Parameter operations that can return Parameters. If numpy operations, the operation gets
    # deferred to __array_ufunc__ to return floats/numpy arrays

    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value_ + other.value_, name = self.name + '-add-' + other.name)
        else:
            return Parameter(self.value_ + other, name = self.name + '-add-float')

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value_ * other.value_, name = self.name + '-mul-' + other.name)
        else:
            return Parameter(self.value_ * other, name = self.name + '-mul-float')

    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value_ - other.value_, name = self.name + '-sub-' + other.name)
        else:
            return Parameter(self.value_ - other, name = self.name + '-sub-float')

    __rsub__ = __sub__

    def _div__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value_ / other.value_, name = self.name + '-div-' + other.name)
        else:
            return Parameter(self.value_ / other, name = self.name + '-div-float')

    __rdiv_ = _div__

    def __pow__(self, other):
        if isinstance(other, Parameter):
            return Parameter(self.value_ ** other.value_, name = self.name + '-pow-' + other.name)
        else:
            return Parameter(self.value_ ** other, name = self.name + '-pow-float')

    def __lt__(self, other):
        if isinstance(other, Parameter):
            return self.value_ < other.value_
        else:
            return self.value_ < other

    def __gt__(self, other):
        if isinstance(other, Parameter):
            return self.value_ > other.value_
        else:
            return self.value_ > other

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.value_ == other.value_
        else:
            return self.value_ == other
