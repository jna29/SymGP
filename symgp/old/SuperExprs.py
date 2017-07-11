from sympy import Symbol, Mul, Add, Function, Integer
from sympy.core import S
from sympy.core.decorators import call_highest_priority
from sympy.matrices.matrices import ShapeError

class AbstractMatSymbol(object):
    
    def transpose(self):
        pass
                
    def inverse(self):
        pass
    
    @call_highest_priority('__neg__')
    def __neg__(self):
        return SuperMul(S.NegativeOne, self)
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        pass
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        pass
    
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return SuperAdd(self, other)
    
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return SuperAdd(other, self)
    
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return SuperAdd(self, -other)
    
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        print('__rsub__ called')
        return SuperAdd(other, -self) 
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        pass
    
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")
    
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return self*other**S.NegativeOne
    
    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        raise NotImplementedError()

class inv(AbstractMatSymbol, Function):
    
    is_SuperMul = False
    is_SuperAdd = False
    is_SuperSymbol = False
    is_t = False
    is_inv = True
    is_commutative = False
    _op_priority = 10000.0
    
    def __new__(cls, arg):
        return Function.__new__(cls, arg)
    
    def __init__(self, arg):
        self.invSym = None   # Only store for SuperSymbols
        self.sym = arg
        self.shape = arg.shape
        if arg.is_SuperSymbol:
            if arg.is_Square:
                expanded_inv = None
                blockform_inv = None
                if arg.expanded is not None:
                    expanded_inv = arg.expanded.I
                if arg.blockform is not None:
                    blockform_inv = [[0,0], [0,0]]
                    P, Q = arg.blockform[0][0], arg.blockform[0][1]
                    R, S = arg.blockform[1][0], arg.blockform[1][1]
                    blockform_inv[0][0] = SuperAdd(P,-Q*S.I*R).I
                    blockform_inv[0][1] = SuperMul(-blockform_inv[0][0]*Q*S.I)
                    blockform_inv[1][0] = -S.I*R*blockform_inv[0][0]
                    blockform_inv[1][1] = S.I + S.I*R*blockform_inv[0][0]*Q*S.I 
                self.invSym = SuperSymbol(arg.name+"^-1", arg.shape[0], arg.shape[1],
                                        expanded=expanded_inv, blockform=blockform_inv)
            else:
                raise ShapeError("Inverse of non-square matrix %s" % arg)
                
    def inverse(self):
        return self.sym
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    def transpose(self):
        return t(self)
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if self.sym.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(self, other)
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        if self.sym.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(other, self)
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.sym.is_Square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif other is S.NegativeOne:
            return inv(self)
        elif other is S.Zero:
            return Integer(1)   # Identity
        elif other is S.One:
            return self
        else:
            raise NotImplementedError("No other operations are implemented")
        

class t(AbstractMatSymbol, Function):
    
    is_SuperMul = False
    is_SuperAdd = False
    is_SuperSymbol = False
    is_t = True
    is_inv = False
    is_commutative = False
    _op_priority = 10000.0
    
    def __new__(cls, arg):
        if arg.is_SuperMul:
            return SuperMul(*[s.T for s in arg.args[::-1]])
        elif arg.is_SuperAdd:
            return SuperAdd(*[s.T for s in arg.args])
        else:
            return Function.__new__(cls, arg)
    
    def __init__(self, arg):
        self.transSym = None   # Only store for SuperSymbols
        self.sym = arg
        self.shape = (arg.shape[1], arg.shape[0])
        if arg.is_SuperSymbol:    
            expanded_trans = None
            blockform_trans = None
            if arg.expanded is not None:
                expanded_trans = arg.expanded.T
            if arg.blockform is not None:
                blockform_trans = [[0,0],[0,0]]
                blockform_trans[0][0] = arg.blockform[0][0].T
                blockform_trans[0][1] = arg.blockform[1][0].T
                blockform_trans[1][0] = arg.blockform[0][1].T
                blockform_trans[1][1] = arg.blockform[1][1].T 
            self.transSym = SuperSymbol(arg.name+"'", arg.shape[1], arg.shape[0],
                                    expanded=expanded_trans, blockform=blockform_trans)
            
    
    def inverse(self):
        return inv(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    def transpose(self):
        return self.sym
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if self.sym.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(self, other)
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        print("__rmul__ called")
        if self.sym.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(other, self)
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.sym.is_Square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif other is S.NegativeOne:
            return inv(self)
        elif other is S.Zero:
            return Integer(1)   # Identity
        elif other is S.One:
            return self
        else:
            raise NotImplementedError("No other operations are implemented")
          
class SuperMul(Mul):
    """ Special form of 'Mul' where new transpose and inverse ops are 
        supported """
    
    is_SuperMul = True
    is_SuperAdd = False
    is_SuperSymbol = False 
    is_t = False
    is_inv = False
    _op_priority = 10000.0
    
    def __init__(self, *args):
        if args
        self.shape = (args[0].shape[0], args[-1].shape[1])
        self.is_Square = (self.shape[0] == self.shape[1])
        
    def transpose(self):
        return t(self)
         
    T = property(transpose, None, None, 'Matrix transposition')
    
    def inverse(self):
        return inv(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(self, other)
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(other, self)
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.is_Square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif other is S.NegativeOne:
            return inv(self)
        elif other is S.Zero:
            return Integer(1)   # Identity
        elif other is S.One:
            return self
        else:
            raise NotImplementedError("No other operations are implemented")
    

class SuperAdd(Add):
    """ Special form of 'Add' where new transpose and inverse ops are 
        supported """
    
    is_SuperMul = False
    is_SuperAdd = True
    is_SuperSymbol = False
    is_t = False
    is_inv = False
    _op_priority = 10000.0
    
    def __init__(self, *args):
        print(args)
        self.shape = (args[0].shape[0], args[-1].shape[1])
        self.is_Square = (self.shape[0] == self.shape[1])
    
    def transpose(self):
        return t(self)
        
    T = property(transpose, None, None, 'Matrix transposition')
    
    def inverse(self):
        return inv(self)
    
    I = property(inverse, None, None, 'Matrix inversion')

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(self, other)
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(other, self)
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.is_Square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif other is S.NegativeOne:
            return inv(self)
        elif other is S.Zero:
            return Integer(1)   # Identity
        elif other is S.One:
            return self
        else:
            raise NotImplementedError("No other operations are implemented")
    
        
class SuperSymbol(Symbol):
    """ Special form of 'Symbol' that is created to act like a matrix """
    
    is_SuperMul = False
    is_SuperAdd = False
    is_SuperSymbol = True
    is_t = False
    is_inv = False
    _op_priority = 10000.0
    
    def __new__(cls, name, m, n, expanded=None, blockform=None):
        return super(SuperSymbol, cls).__new__(cls, name, commutative=False)
        
    def __init__(self, name, m, n, expanded=None, blockform=None):
        self.shape = (m, n)
        self.expanded = expanded
        self.blockform = blockform
        self.is_Square = (m==n)
    
    def transpose(self):
        return t(self)
    
    T = property(transpose, None, None, 'Matrix transposition')
                
    def inverse(self):
        return inv(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        print('__mul__ called')
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(self, other)
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        print("__rmul__ called")
        if self.is_Square and other == self.I:
            return Integer(1)
        return SuperMul(other, self)
    
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        if not self.is_Square:
            raise ShapeError("Power of non-square matrix %s" % self)
        elif other is S.NegativeOne:
            return inv(self)
        elif other is S.Zero:
            return Integer(1)   # Identity
        elif other is S.One:
            return self
        else:
            raise NotImplementedError("No other operations are implemented")
     
        
        