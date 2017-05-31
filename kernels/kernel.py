#import sys
#sys.path.insert(0, "/Users/jaduol/Documents/Uni (original)/Part II/IIB/MEng Project/Code")  # Hack to allow SuperMatExpr to be imported

from sympy import MatrixSymbol, sympify, Basic, HadamardProduct, S
from sympy.core.decorators import call_highest_priority

from symgp.superexpressions.supermatbase import SuperMatBase
from symgp.superexpressions import SuperMatInverse, SuperMatMul, SuperMatAdd

class KernelMatrix(SuperMatBase, MatrixSymbol):
    
    """
        Symbolic realisation of a covariance function, K, with inputs X1, X2 i.e. K(X1, X2) 
    
        Args:
            name - Name of matrix
            m, n - Shape of matrix. Should satisfy m = inputs[0].shape[0], n = inputs[1].shape[1]
            inputs - The two MatrixSymbols to evaluate this kernel for
            kernel - The covariance function that this matrix corresponds to
    """
    
    _op_priority = 10000
    
    def __new__(cls, name, m, n, inputs, kernel):
        
        if name == '':
            name = kernel.name+'(' + ','.join([i.name for i in inputs]) + ')'
        obj = Basic.__new__(cls,name,inputs[0],inputs[1])
        
        return obj
    
    def __init__(self, name, m, n, inputs, kernel):
        
        """
            Initialize a KernelMatrix object
        
            Args:
                name - Name of matrix
                m, n - Shape of matrix. Should satisfy m = inputs[0].shape[0], n = inputs[1].shape[1]
                inputs - The two MatrixSymbols to evaluate this kernel for
                kernel - The covariance function that this matrix corresponds to
        """
        self.K_func = kernel
        
    def doit(self, **hints):
        if hints.get('deep', True):
            m, n = self.args[1].shape[0].doit(**hints), self.args[2].shape[0].doit(**hints)
            return type(self)(self.name, m, n, [self.args[1].doit(**hints),
                    self.args[2].doit(**hints)], self.K_func)
        else:
            return self
    
    @property
    def shape(self):
        return (self.args[1].shape[0], self.args[2].shape[0])
     
    def transpose(self):
        if self.shape[0] == self.shape[1]:
            return self
        else:
            return self.K_func(self.args[2],self.args[1])
    
    T = property(transpose, None, None, 'Matrix transposition.')
    
    def inverse(self):
        return SuperMatInverse(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
           
class Kernel(object):
    
    """
        GP Kernel (covariance function) object.
    """
        
    def __init__(self, sub_kernels=[], kernel_type='nul', mat=None, name='K'):
        """
            Initializes a Kernel object
        
            Args:
                sub_kernels - A list of the constitutive addition/multiplicative Kernels
                kernel_type - Specifies whether the component Kernels are multiplicative ('mul'), additive ('add'), subtractive ('sub') or 'nul' 
                              indicating that there are no components i.e. this is a base kernel
                mat - The centre matrix between kernels e.g. K(xi,xj) = K(xi,u)M_(u,u)K(u,xj) 
                      where K are kernels and M is a matrix. M should be a SuperMatSymbol with one
                      dependent_vars.
                name - Name of the kernel
        """
        
        self.sub_kernels = sub_kernels
        self.type = kernel_type
        self.M = mat
        self.name = name
        
    def K(self, xi, xj):
        """
            Evaluate kernel for given inputs
        
            Args:
                xi, xj - SuperMatSymbols of shapes (d, ni) and (d, nj) respectively where
                         d is the dimension of each vector and ni, nj are the number of vectors
        
            Returns:
                A KernelMatrix object that represents the covariance between the inputs
                         
        """
        #print(self.type)
        if self.type == 'mul':
            left_kern = self.sub_kernels[0]
            right_kern = self.sub_kernels[1]
            
            if self.M is not None:
                u = self.M.dep_vars[0]
                #print("u: ",u)
                left_kern_mat = left_kern.K(xi,u)
                right_kern_mat = right_kern.K(u,xj)
                #print("M: ",self.M)
                M = self.M if self.M.expanded is None else self.M.expanded
                return left_kern_mat*M*right_kern_mat
            else:
                left_kern_mat = left_kern.K(xi,xj)
                right_kern_mat = right_kern.K(xi,xj)
                return HadamardProduct(left_kern_mat, right_kern_mat)
                        
        elif self.type == 'add' or self.type == 'sub':
            left_kern = self.sub_kernels[0]
            right_kern = self.sub_kernels[1]
            
            left_kern_mat = left_kern.K(xi,xj)
            right_kern_mat = right_kern.K(xi,xj)
            
            if self.type == 'add':
                return left_kern_mat + right_kern_mat
            else:
                return left_kern_mat - right_kern_mat
            
        else: # self.type == 'nul'
            return KernelMatrix('',xi.shape[1],xj.shape[1],[xi, xj],kernel=self)
        
    def __call__(self, xi, xj):
        
        return self.K(xi,xj)
    
    def __repr__(self):
        if self.M and self.type == 'mul':
            return self.sub_kernels[0].__repr__() + '*{' + self.M.name + '}*'+self.sub_kernels[1].__repr__()
        elif self.type == 'add' or self.type == 'sub':
            lparen, rparen = '', '' 
            if len(self.sub_kernels) > 0:
                lparen = '('
                rparen = ')'
            sign = '+' if self.type == 'add' else '-'
            return lparen+self.sub_kernels[0].__repr__() + sign + self.sub_kernels[1].__repr__()+rparen
        else:
            return self.name
        
    def __add__(self, other):
        name = self.__repr__() + '+' + other.__repr__()
        return Kernel([self, other], 'add', name=name)
    
    def __radd__(self, other):
        name = other.__repr__() + '+' + self.__repr__()
        return Kernel([other, self], 'add', name=name)
    
    def __sub__(self, other):
        name = self.__repr__() + '-' + other.__repr__()
        return Kernel([self, other], 'sub', name=name)
    
    def __rsub__(self, other):
        name = other.__repr__() + '-' + self.__repr__()
        return Kernel([other, self], 'sub', name=name)
        
    def __mul__(self, other):
        name = self.__repr__() + '*' + other.__repr__()
        return Kernel([self, other], 'mul', name=name)
    
    def __rmul__(self, other):
        name = other.__repr__() + '*' + self.__repr__()
        return Kernel([other, self], 'mul', name=name)