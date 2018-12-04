from typing import Union, List
from copy import copy

from sympy import MatrixSymbol, Basic, HadamardProduct, Symbol, MatrixExpr

from symgp.superexpressions.supermatbase import SuperMatBase
from symgp.superexpressions.supermatexpr import SuperMatSymbol, Covariance, Variable, \
    CompositeVariable
from symgp.superexpressions import SuperMatInverse


Vector_t = List[MatrixExpr]
Matrix_t = List[List[MatrixExpr]]

class Kernel(object):
    """
    GP Kernel (covariance function).

    This class allows for composition of kernels using the standard mathematical operators '+',
    '-' and '*'. When we do this, we create a new 
    """

    ALLOWED_KERNEL_TYPES = ['nul', 'mul', 'sub', 'add']
        
    def __init__(self, sub_kernels=None, kernel_type='nul', mat=None, name='K'):
        """
        Initializes a Kernel object
        :param sub_kernels: A list of the constitutive addition/multiplicative Kernels.
        :param kernel_type: Specifies whether the component Kernels are multiplicative ('mul'),
        additive ('add'), subtractive ('sub'). If ``sub_kernels`` isn't specified, then we
        should leave this as `nul` indicating that there are no components i.e.
        this is a base kernel.
        :param mat: The centre matrix between kernels e.g. K(xi,xj) = K(xi,u)M_(u,u)K(u,xj).
        where K are the sub_kernels and M is a matrix. M should be a SuperMatSymbol with one
        ``dep_vars`` and ``kernel_type`` should be 'mul'.
        :param name: Name of the kernel. Determines the name of the kernel when printed.
        """

        assert kernel_type in Kernel.ALLOWED_KERNEL_TYPES, ("``kernel_type`` must be one of {}".
            format(Kernel.ALLOWED_KERNEL_TYPES))
        
        # Check validity of arguments
        if sub_kernels is not None and len(sub_kernels) != 2 and len(sub_kernels) != 0:
            raise Exception("``sub_kernels`` should have only two elements or "
                            "be empty (for a base ``Kernel``)")

        if (sub_kernels is None and kernel_type != 'nul') or (sub_kernels is not None and
                                                                  kernel_type == 'nul'):
            raise Exception("If ``sub_kernels`` is ``None``, ``kernel_type`` should be ``nul`` "
                            "and vice versa.")

        # Check validity of mat
        if mat is not None and isinstance(mat, SuperMatSymbol):
            assert len(mat.dep_vars) == 1, "mat.dep_vars should have one element"
            assert kernel_type == 'mul', "``kernel_type`` should be 'mul'"

        self._sub_kernels = sub_kernels if sub_kernels is not None else []
        self._type = kernel_type
        self._M = mat
        self._name = name

    def __call__(self, xi, xj):
        return self.K(xi, xj)

    def __repr__(self):
        return "KernelMatrix(sub_kernels={},kernel_type={},mat={},name={})".format(
            self.sub_kernels, self.type, self._M, self.name)

    def __str__(self):
        if self._M and self.type == 'mul':
            return str(self.sub_kernels[0])+'*{'+self._M.name+'}*'+str(self.sub_kernels[1])
        elif self.type == 'add' or self.type == 'sub':
            lparen, rparen = '', '' 
            if len(self.sub_kernels) > 0:
                lparen = '('
                rparen = ')'
            sign = '+' if self.type == 'add' else '-'
            return lparen + str(self.sub_kernels[0]) + sign + str(self.sub_kernels[1]) + rparen
        else:
            return self.name
    
    # TODO: Find better naming convention    
    def __add__(self, other):
        name = "K_{" + str(self) + '+' + str(other) + "}"
        return Kernel([self, other], 'add', name=name)
    
    def __radd__(self, other):
        name = "K_{" + str(other) + '+' + str(self) + "}"
        return Kernel([other, self], 'add', name=name)
    
    def __sub__(self, other):
        name = "K_{" + str(self) + '-' + str(other) + "}"
        return Kernel([self, other], 'sub', name=name)
    
    def __rsub__(self, other):
        name = "K_{" + str(other) + '-' + str(self) + "}"
        return Kernel([other, self], 'sub', name=name)
        
    def __mul__(self, other):
        name = "K_{" + str(self) + '*' + str(other) + "}"
        return Kernel([self, other], 'mul', name=name)
    
    def __rmul__(self, other):
        name = "K_{" + str(other) + '*' + str(self) + "}"
        return Kernel([other, self], 'mul', name=name)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def sub_kernels(self):
        return self._sub_kernels
    
    def K(self, xi: MatrixSymbol, xj: MatrixSymbol):
        """
        Evaluate kernel for given inputs.
        :param xi: Kernel input of shape (N_i, D) where N_i is the number of data points and D is
        the dimensionality.
        :param xj: Kernel input of shape (N_j, D) where N_j is the number of data points and D is
        the dimensionality.
        :return: A KernelMatrix object that represents the covariance between the inputs. Of
        shape (N_i, N_j)
        """

        if self.type == 'mul':
            left_kern = self.sub_kernels[0]
            right_kern = self.sub_kernels[1]
            
            if self._M is not None:
                u = self._M.dep_vars[0]

                left_kern_mat = left_kern.K(xi,u)
                right_kern_mat = right_kern.K(u,xj)
                M = self.get_M()

                expanded = left_kern_mat*M*right_kern_mat
            else:
                left_kern_mat = left_kern.K(xi,xj)
                right_kern_mat = right_kern.K(xi,xj)

                expanded = HadamardProduct(left_kern_mat, right_kern_mat)

            return KernelMatrix(xi, xj, kernel=self, full_expr=expanded)
        elif self.type == 'add' or self.type == 'sub':
            left_kern = self.sub_kernels[0]
            right_kern = self.sub_kernels[1]
            
            left_kern_mat = left_kern.K(xi,xj)
            right_kern_mat = right_kern.K(xi,xj)
            
            if self.type == 'add':
                expanded = left_kern_mat + right_kern_mat
            else:
                expanded = left_kern_mat - right_kern_mat
            
            return KernelMatrix(xi, xj, kernel=self, full_expr=expanded)
            
        else: # self.type == 'nul'
            return KernelMatrix(xi, xj, kernel=self)
    
    def get_M(self) -> SuperMatSymbol:
        """
        Get the middle M matrix if it exists
        """
        M = self._M if self._M.expanded is None else self._M.to_full_expr()
        return M

class KernelMatrix(Covariance):
    _op_priority = 10000  # Specifies the priority of this class in matrix operations

    def __new__(cls, v1: Union[Variable, CompositeVariable], v2: Union[Variable, CompositeVariable],
                kernel: Kernel, cond_vars: List[Union[Variable, CompositeVariable]] = None,
                name: str = '', full_expr: Union[MatrixExpr, Matrix_t] = None):
        inputs = [v1, v2]

        if name == '':
            name = kernel.name + '_{' + ','.join([i.name for i in inputs]) + '}'

        return Covariance.__new__(cls, v1=v1, v2=v2, cond_vars=cond_vars, name=name,
                                  full_expr=full_expr)


    def __init__(self, v1: Union[Variable, CompositeVariable], v2: Union[Variable, CompositeVariable],
                kernel: Kernel, cond_vars: List[Union[Variable, CompositeVariable]] = None,
                name: str = '', full_expr: Union[MatrixExpr, Matrix_t] = None):
        """
        See ``__new__`` above.
        """
        assert v1.shape[1] == v2.shape[1], "Both inputs must have same dimensionality."
        self._K = kernel
        self._inputs = [v1, v2]
        super(KernelMatrix, self).__init__(v1=v1, v2=v2, cond_vars=cond_vars, name=name,
                                           full_expr=full_expr)

    @property
    def K(self):
        return self._K

    @property
    def inputs(self):
        return self._inputs

    def transpose(self):
        if self.shape[0] == self.shape[1]:
            return self
        else:
            return self._K(self.dep_vars[1], self.dep_vars[0])

    T = property(transpose, None, None, 'Matrix transposition.')