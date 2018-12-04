from sympy import MatMul, MatAdd, ZeroMatrix, MatrixBase, S
from sympy.core.decorators import call_highest_priority
from sympy.core.compatibility import reduce
from operator import add
from sympy.strategies import (rm_id, unpack, typed, flatten, sort, condition, exhaust,
        do_one, new, glom)
from sympy.utilities import default_sort_key, sift

from .supermatbase import SuperMatBase
 
class SuperMatAdd(SuperMatBase, MatAdd):
    """
    Redefines some methods of MatAdd so as to make them amenable to our application
    """

    _op_priority = 10000

    def __new__(cls, *args, **kwargs):
        return MatAdd.__new__(cls, *args, **kwargs)
        
    def _eval_transpose(self):
        return SuperMatAdd(*[arg.T for arg in self.args]).doit()

    #def transpose(self):
    #    from .supermatexpr import SuperMatTranspose
    #    return SuperMatTranspose(self).doit()
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        
        # Fix to stop first negative term being rendered as -1 in LaTex i.e. we want
        #   A - BCB^{T} in LaTex instead of  -1BCB^{T} + A
        #if args[0].args[0] == S.NegativeOne:
            #print("Before: ",args)
            #args = args[1:] + args[:1]
            #print("After: ",args)
        
        return canonicalize(SuperMatAdd(*args))

factor_of = lambda arg: arg.as_coeff_mmul()[0]
matrix_of = lambda arg: unpack(arg.as_coeff_mmul()[1])
def combine(cnt, mat):
    if cnt == 1:
        return mat
    else:
        return cnt * mat


def merge_explicit(matadd):
    """ Merge explicit MatrixBase arguments
    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = eye(2)
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatAdd(A, B, C)
    >>> pprint(X)
        [1  0]   [1  2]
    A + [    ] + [    ]
        [0  1]   [3  4]
    >>> pprint(merge_explicit(X))
        [2  2]
    A + [    ]
        [3  5]
    """
    groups = sift(matadd.args, lambda arg: isinstance(arg, MatrixBase))
    if len(groups[True]) > 1:
        return SuperMatAdd(*(groups[False] + [reduce(add, groups[True])]))
    else:
        return matadd

         
# MatAdd
rules = (rm_id(lambda x: x == 0 or isinstance(x, ZeroMatrix)),
         unpack,
         flatten,
         glom(matrix_of, factor_of, combine),
         merge_explicit,
         sort(default_sort_key))

canonicalize = exhaust(condition(lambda x: isinstance(x, SuperMatAdd),
                                 do_one(*rules)))

from .supermatmul import SuperMatMul