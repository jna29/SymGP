from sympy import S, MatMul
from sympy.core.decorators import call_highest_priority

class SuperMatBase(object):
    """
        The Base class for the redefined matrix symbols we have here
    """
    
    def __neg__(self):
        from .supermatmul import SuperMatMul
        return SuperMatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    
    @call_highest_priority('__radd__')
    def __add__(self, other):
        from .supermatadd import SuperMatAdd
        return SuperMatAdd(self, other).doit()

    
    @call_highest_priority('__add__')
    def __radd__(self, other):
        from .supermatadd import SuperMatAdd
        return SuperMatAdd(other, self).doit()

    
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        from .supermatadd import SuperMatAdd
        return SuperMatAdd(self, -other).doit()

    
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        from .supermatadd import SuperMatAdd
        return SuperMatAdd(other, -self).doit()

    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        from .supermatmul import SuperMatMul
        if isinstance(other, MatMul):
            return SuperMatMul(self, *other.args).doit()
        else:
            return SuperMatMul(self, other).doit()

    
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        from .supermatmul import SuperMatMul
        if isinstance(other, MatMul):
            return SuperMatMul(self, *other.args).doit()
        else:
            return SuperMatMul(self, other).doit()

    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        from .supermatmul import SuperMatMul
        if isinstance(other, MatMul):
            return SuperMatMul(*other.args, self).doit()
        else:
            return SuperMatMul(other, self).doit()

    
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        from .supermatmul import SuperMatMul
        if isinstance(other, MatMul):
            return SuperMatMul(*other.args, self).doit()
        else:
            return SuperMatMul(other, self).doit()
