import string

from sympy import MatrixSymbol, BlockMatrix, Symbol, Inverse, Transpose, MatMul, MatAdd, ZeroMatrix, MatrixExpr, S
from sympy.core.decorators import call_highest_priority
from sympy.strategies import (rm_id, unpack, typed, flatten, sort, condition, exhaust,
        do_one, new, glom)
        
from .supermatbase import SuperMatBase
from .supermatmul import SuperMatMul
from .supermatadd import SuperMatAdd

from symgp.utils import utils

# Some unicode symbols
SMALL_MU_GREEK = '\u03bc'
BIG_SIGMA_GREEK = '\u03a3'
BIG_OMEGA_GREEK = '\u03a9'
BIG_LAMBDA_GREEK = '\u039b'
SMALL_ETA_GREEK = '\u03b7' 

class SuperMatSymbol(SuperMatBase, MatrixSymbol):
    
    """mean_count = 0
    covar_count = 0
    invcovar_count = 0
    var_count = 0"""
    _op_priority = 99
    _used_names = []
    #_available_names = [l for l in string.ascii_letters]
    
    ## TODO: Maybe change constructor to get rid of name  
    def __new__(cls, m, n, name='', mat_type='other', dep_vars=[], cond_vars=[], expanded=None, blockform=None):
        """
            The SuperMatSymbol constructor.
        
            It selects the name of the symbol based on the parameters given.
        
            Args:
                (See '__init__' below)
    
            Returns:
                An object of class MatrixSymbol
        """
        
        # Create name of symbol based on dep_vars and cond_vars if this is a
        # 'mean', 'covar', 'invcovar', 'natmean' or 'precision' symbol
        if (mat_type == 'mean' or mat_type == 'covar' or mat_type == 'invcovar' or
            mat_type == 'natmean' or mat_type == 'precision') and name == '':
            if mat_type == 'mean':
                if blockform is None and expanded is None:
                    pre_sym = SMALL_MU_GREEK
                else:
                    pre_sym = 'm'
                
                if not isinstance(dep_vars[0],list):
                    name += pre_sym+'_{'+','.join([v.name for v in dep_vars])
                else:
                    dep_vars_x = dep_vars[0]
                    dep_vars_y = dep_vars[1]
                    name += pre_sym+'_{'+','.join([v.name for v in dep_vars_x+dep_vars_y])
            elif mat_type == 'covar':
                if blockform is None and expanded is None:
                    pre_sym = BIG_SIGMA_GREEK
                else:
                    pre_sym = 'S'
                    
                if not isinstance(dep_vars[0],list):
                    name += pre_sym+'_{'+','.join([v.name for v in dep_vars])
                else:
                    dep_vars_x = dep_vars[0]
                    dep_vars_y = dep_vars[1]
                    name += pre_sym+'_{'+','.join([v.name for v in dep_vars_x+dep_vars_y]) 
            elif mat_type == 'invcovar':
                if blockform is None and expanded is None:
                    pre_sym = BIG_SIGMA_GREEK
                else:
                    pre_sym = 'S'
                    
                if not isinstance(dep_vars[0],list):
                    name += pre_sym+'_i_{'+','.join([v.name for v in dep_vars])
                else:
                    dep_vars_x = dep_vars[0]
                    dep_vars_y = dep_vars[1]
                    name += pre_sym+'_i_{'+','.join([v.name for v in dep_vars_x+dep_vars_y])
            elif mat_type == 'natmean':
                if blockform is None and expanded is None:
                    pre_sym = SMALL_ETA_GREEK
                else:
                    pre_sym = 'n'
                    
                if not isinstance(dep_vars[0],list):
                    name += pre_sym+'_1_{'+','.join([v.name for v in dep_vars])
                else:
                    dep_vars_x = dep_vars[0]
                    dep_vars_y = dep_vars[1]
                    name += pre_sym+'_1_{'+','.join([v.name for v in dep_vars_x+dep_vars_y])
            else: # mat_type == 'precision'
                if blockform is None and expanded is None:
                    pre_sym = SMALL_ETA_GREEK
                else:
                    pre_sym = 'n'
                    
                if not isinstance(dep_vars[0],list):
                    name += pre_sym+'_2_{'+','.join([v.name for v in dep_vars])
                else:
                    dep_vars_x = dep_vars[0]
                    dep_vars_y = dep_vars[1]
                    name += pre_sym+'_2_{'+','.join([v.name for v in dep_vars_x+dep_vars_y])  
            
            if len(cond_vars) > 0:
                name += '|'+','.join([v.name for v in cond_vars])
            
            name += '}'
        else:
            if len(cond_vars) > 0 and len(cond_vars) < 0:
                raise Warning("cond_vars should only be set for mat_type = {'covar', 'invcovar', 'mean', 'natmean', 'precision'}")

        SuperMatSymbol._used_names.append(name)
        
        return MatrixSymbol.__new__(cls, name, m, n)
    
    def __init__(self, m, n, name='', mat_type='other', dep_vars=[], cond_vars=[], expanded=None, blockform=None):
        """
            The SuperMatSymbol initialiser.
        
            Args:
                - 'name' - The name of the symbol. Only used with mat_type == 'var' or 'other' (See '__new__'). Otherwise specify use: name=''.
                - 'm,n' - Shape of matrix
                - 'mat_type' - Type of matrix. Can be 'covar', 'invcovar', 'mean', 'natmean', 'precision', 'var' or 'other'
              (Optional)
                - 'dep_vars' - The variables that this symbol depends on. They are used to locate the submatrices in 'blockform'. 
                                     Give no argument if there is only one dependent variable.
                - 'cond_vars' - The variables this parameter is conditioned on. Can only be used for 'covar', 'invcovar', 'mean', 'natmean', 'precision'
                - 'expanded' - The full expression of a matrix
                - 'blockform' - A block matrix representation of the matrix
        """
        
        self.mat_type = mat_type
        self.dep_vars = dep_vars
        self.cond_vars = list(cond_vars)
        self.expanded = expanded
        self.blockform = blockform
        self.variables_dim1 = {}
        self.variables_dim2 = {}
        if len(dep_vars) > 0:
            dep_vars = list(dep_vars)
            if isinstance(dep_vars[0],list): # Not square covar case
                for i in range(len(dep_vars[0])):
                    if dep_vars[0][i] not in self.variables_dim1:
                        self.variables_dim1[dep_vars[0][i]] = i
                
                for j in range(len(dep_vars[1])):
                    if dep_vars[1][j] not in self.variables_dim2:
                        self.variables_dim2[dep_vars[1][j]] = j
            else: # All other cases
                for i in range(len(dep_vars)):
                    if dep_vars[i] not in self.variables_dim1:
                        self.variables_dim1[dep_vars[i]] = i
                
                if n != 1:
                    for j in range(len(dep_vars)):
                        if dep_vars[j] not in self.variables_dim2:
                            self.variables_dim2[dep_vars[j]] = j
        
        #print("Creating: ",self.name)
        #print("expanded: ",self.expanded)
        #print("blockform: ",self.blockform)
        
    
    # We have to change MatrixSymbol.doit so that objects are constructed appropriately
    # Modeled off MatrixSymbol.doit in https://github.com/sympy/sympy/blob/master/sympy/matrices/expressions/matexpr.py
    def doit(self, **hints):
        return self
        
    def inverse(self):
        return SuperMatInverse(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    def transpose(self):
        if ((self.mat_type == 'covar' or self.mat_type == 'invcovar' or self.mat_type == 'sym') and
            (self.shape[0] == self.shape[1])):
            return self
        else:
            return SuperMatTranspose(self)
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    def partition(self, indices):
        """ 
            Partition a 'blockform' into sections defined by 'indices'.
                - With a mxn matrix, four partitions are created.
                - With a mx1 or 1xm matrix, two partitions are created
        
            Args:
                - 'indices' - The indices that define where to partition the blockform. 2-D list of lists for mxn matrices and 1-D
                              list for mx1 or 1xm matrices where each list specifies the indices for each dimension
        
            Returns:
                - P, Q (, R, S) - The partitions of the matrix. R and S are returned if matrix is mxn for m != 1 and n != 1 
        """
        
        if self.blockform is not None:
            # First check indices are valid
            if all([isinstance(l,list) for l in indices]) and len(indices)==2:
                assert (all([isinstance(l, list) for l in self.blockform])), "self.blockform must be a 2-D list of lists for these indices"
                assert (all([i>=0 and i < len(self.blockform) for i in indices[0]])), "Invalid first set of indices"
                assert (all([i>=0 and i < len(self.blockform[0]) for i in indices[1]])), "Invalid first set of indices"
                
                P, Q, R, S = utils.partition_block(self.blockform, indices)
                #print("P: ",P)
                #print("Q: ",Q)
                #print("R: ",R)
                #print("S: ",S)
                    
                # Sort the variables based on their index in 'self.blockform'
                variables_dim1_keys = sorted(self.variables_dim1.keys(),key=lambda m: self.variables_dim1[m])
                variables_dim2_keys = sorted(self.variables_dim2.keys(),key=lambda m: self.variables_dim2[m])
                    
                # Get variables for both sets in both dimensions 
                indices_vars_dim1 = [v for v in variables_dim1_keys if self.variables_dim1[v] in indices[0]]
                #print("indices_vars_dim1: ",indices_vars_dim1)
                indices_vars_dim2 = [v for v in variables_dim2_keys if self.variables_dim2[v] in indices[1]]
                #print("indices_vars_dim2: ",indices_vars_dim2)
                rem_vars_dim1 = [v for v in variables_dim1_keys if self.variables_dim1[v] not in indices[0]]
                #print("rem_vars_dim1: ",rem_vars_dim1)
                rem_vars_dim2 = [v for v in variables_dim2_keys if self.variables_dim2[v] not in indices[1]]
                #print("rem_vars_dim2: ",rem_vars_dim2)
                    
                # Get shapes of two sets in both dimensions
                m1 = sum([v.shape[0] for v in rem_vars_dim1])
                n1 = sum([v.shape[0] for v in rem_vars_dim2])
                m2 = sum([v.shape[0] for v in indices_vars_dim1])
                n2 = sum([v.shape[0] for v in indices_vars_dim2])
                    
                # Create partition symbols. self.mat_type = {'covar','invcovar','precision','other'} ('other' must be 2-D)
                #print("self.covar.name: ",self.name)
                P = SuperMatSymbol(m1,n1,mat_type=self.mat_type, dep_vars=rem_vars_dim1, blockform=P)
                #print("P.name: ",P.name)
                Q = SuperMatSymbol(m1,n2,mat_type=self.mat_type, dep_vars=[rem_vars_dim1, indices_vars_dim2], blockform=Q)
                #print("Q.name: ",Q.name)
                R = SuperMatSymbol(m2,n1,mat_type=self.mat_type, dep_vars=[indices_vars_dim1, rem_vars_dim2], blockform=R)
                #print("R.name: ",R.name)
                S = SuperMatSymbol(m2,n2,mat_type=self.mat_type, dep_vars=indices_vars_dim1, blockform=S)
                #print("S.name: ",S.name)
                    
                return P,Q,R,S  
                        
                    
            elif all([isinstance(l,int) for l in indices]):
                assert (all([not isinstance(l, list) for l in self.blockform])), "self.blockform must be a 1-D list"
                assert (all([i>=0 and i < len(self.blockform) for i in indices])), "Invalid set of indices"
                
                P, S = utils.partition_block(self.blockform, indices)
                
                # Sort variables based on their index in 'self.blockform'
                variables_keys = sorted(self.variables_dim1.keys(),key=lambda m: self.variables_dim1[m])
                
                # Get variables for both sets
                indices_vars = [v for v in variables_keys if self.variables_dim1[v] in indices]
                rem_vars = [v for v in variables_keys if self.variables_dim1[v] not in indices]
                
                # Get shapes of two sets
                m1 = sum([v.shape[0] for v in rem_vars])
                m2 = sum([v.shape[0] for v in indices_vars])
                
                # If we are partition a variable, we need to create a new name for each half
                if self.mat_type == 'var':
                    name1 = 'v_('+','.join([v.name for v in rem_vars])+')'
                    name2 = 'v_('+','.join([v.name for v in indices_vars])+')'
                else:
                    name1 = ''
                    name2 = ''
                    
                P = SuperMatSymbol(m1,1,name1,mat_type=self.mat_type,dep_vars=rem_vars,blockform=P)
                S = SuperMatSymbol(m2,1,name2,mat_type=self.mat_type,dep_vars=indices_vars,blockform=S)
                
                return P, S
            else:
                raise Exception("Invalid set of indices")
        else:
            raise Exception("Operation can't be performed as there is no blockform.")

    def expand_partition(self):
        """
            Expands blockform to replace SuperMatSymbol representations with their blockforms.
            Only works with 2x2 matrices or 1x2 vectors (i.e. [a,b])
        """
        
        if self.blockform is not None:
            
            # Check size of blockform
            if len(self.blockform) == 2 and isinstance(self.blockform[0],list) and len(self.blockform[0]) == 2:
                P_exp, Q_exp = self.blockform[0][0].blockform, self.blockform[0][1].blockform
                R_exp, S_exp = self.blockform[1][0].blockform, self.blockform[1][1].blockform
                
                # Check blockforms exist
                if P_exp is not None or Q_exp is not None or R_exp is not None or S_exp is not None:
                    raise Exception("One of the blocks doesn't have a blockform")
                
                # Check that the shapes match
                assert(len(P_exp)==len(Q_exp))
                assert(len(P_exp[0])==len(R_exp[0]))
                assert(len(Q_exp[0])==len(S_exp[0]))
                assert(len(R_exp)==len(S_exp))
                
                # Create the top and bottom part of the larger matrix i.e. top = [P_exp, Q_exp], bottom = [R_exp, S_exp] 
                top = []
                for row1, row2 in zip(P_exp,Q_exp):
                    top.append(row1+row2)
                
                bottom = []
                for row1, row2 in zip(R_exp,S_exp):
                    bottom.append(row1+row2)
                
                self.blockform = top+bottom
                
            elif not isinstance(self.blockform[0],list) and len(self.blockform) > 1:
                P_exp = self.blockform[0].blockform
                S_exp = self.blockform[1].blockform
                
                self.blockform = P_exp+S_exp
                
            else:
                raise Exception("self.blockform isn't a 2x2 matrix or a 1x2 list")
                
        else:
            raise Exception("This symbol has no blockform")
    
    def to_full_expr(self):
        """
            Returns the full expression for the blockform or expanded of this symbol
        """
        
        if self.blockform is not None:
            if isinstance(self.blockform[0],list):
                m, n = len(self.blockform), len(self.blockform[0])
                
                full_expr_blockform = []
                
                for i in range(m):
                    full_expr_blockform.append([utils.expand_to_fullexpr(s) for s in self.blockform[i]])
                
                return full_expr_blockform
            else:
                return [utils.expand_to_fullexpr(s) for s in self.blockform]
                
        elif self.expanded is not None:
            return utils.expand_to_fullexpr(self)
        else:
            return self
    
    @staticmethod
    def getUsedNames():
        return SuperMatSymbol._used_names
        
    
    @staticmethod
    def used(name):
        """
            Checks whether name is used
        """
        return name in SuperMatSymbol._used_names
            
class SuperMatInverse(SuperMatBase, Inverse):
    
    _op_priority = 10000
    
    def __new__(cls, mat, expanded=None, blockform=None):
        return Inverse.__new__(cls, mat)
        
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = expanded
        self.blockform = blockform
        
        if (isinstance(mat, SuperMatSymbol) or isinstance(mat, SuperMatInverse) or
            isinstance(mat, SuperMatTranspose) or isinstance(mat, SuperDiagMat)):
            self.mat_type = 'invcovar' if mat.mat_type == 'covar' else 'covar'
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim2
            self.variables_dim2 = mat.variables_dim1
            
            if mat.expanded is not None:
                self.expanded = mat.expanded.I
            
            if mat.blockform is not None and len(mat.blockform) > 1:            
                self.blockform = utils.matinv(mat.blockform)
                
                vars_dim1 = sorted(list(mat.variables_dim1.keys()), key=lambda m: mat.variables_dim1[m])
                vars_dim2 = sorted(list(mat.variables_dim2.keys()), key=lambda m: mat.variables_dim2[m])
            
                if mat.mat_type == 'covar':
                    mat_type = 'invcovar'
                else:
                    mat_type = 'covar'
            
                self.mat_type = mat_type
                
                for i in range(len(self.blockform)):
                    for j in range(len(self.blockform[0])):
                        var_i, var_j = vars_dim1[i], vars_dim2[j]
                        self.blockform[i][j] = SuperMatSymbol(var_i.shape[0],var_j.shape[0], mat_type=mat_type, dep_vars=[var_i, var_j], expanded=self.blockform[i][j])
            
        else:
            self.mat_type = None
            self.dep_vars = None
            self.cond_vars = None
            self.variables_dim1 = None
            self.variables_dim2 = None
               
    def transpose(self):
        return self.arg.transpose().doit().inverse()
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    def doit(self, **hints):
        return self.arg.inverse()
    
    @property
    def name(self):
        return self.arg.name
    
class SuperMatTranspose(SuperMatBase, Transpose):
    
    _op_priority = 10000
    
    def __new__(cls, mat, expanded=None, blockform=None):
        return Transpose.__new__(cls, mat)
    
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = expanded
        self.blockform = blockform
        
        if (isinstance(mat, SuperMatSymbol) or isinstance(mat, SuperMatInverse) or
            isinstance(mat, SuperMatTranspose)):
            self.mat_type = mat.mat_type
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim2
            self.variables_dim2 = mat.variables_dim1
            
            if mat.expanded is not None:
                self.expanded = mat.expanded.T
            if mat.blockform is not None:
                if not isinstance(mat.blockform[0],list):
                    self.blockform = []
                    self.blockform.extend([x.T for x in mat.blockform])
                else:
                    self.blockform = [[0 for _ in range(len(mat.blockform))] for _ in range(len(mat.blockform[0]))]
                
                    for i in range(len(mat.blockform)):
                        for j in range(len(mat.blockform[0])):
                            self.blockform[j][i] = mat.blockform[i][j].T
        else:
            self.mat_type = None
            self.dep_vars = None
            self.cond_vars = None
            self.variables_dim1 = None
            self.variables_dim2 = None
            
    def doit(self, **hints):
        return self.arg.transpose()
    
    @property
    def name(self):
        return self.arg.name
        
class SuperDiagMat(SuperMatBase, MatrixExpr):
    
    _op_priority = 10005
    
    def __new__(cls, mat, expanded=None, blockform=None):
        if isinstance(mat, SuperDiagMat):
            return mat
        else:
            return MatrixExpr.__new__(cls, mat)
        
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = expanded
        self.blockform = blockform
        
        if (isinstance(mat, SuperMatSymbol) or isinstance(mat, SuperMatInverse) or
            isinstance(mat, SuperMatTranspose) or isinstance(mat, SuperDiagMat) or
            isinstance(mat, SuperBlockDiagMat)):
            self.mat_type = 'diag'
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim1
            self.variables_dim2 = mat.variables_dim2
            
        else:
            self.mat_type = 'diag'
            self.dep_vars = None
            self.cond_vars = None
            self.variables_dim1 = None
            self.variables_dim2 = None
            
    def __repr__(self):
        return 'diag['+repr(self.arg)+']'
        
    def __str__(self):
        return 'diag['+repr(self.arg)+']'
    
    def __neg__(self):
        return SuperMatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError
        
    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, SuperDiagMat):
            return SuperDiagMat(SuperMatAdd(self.arg, other.arg).doit()).doit()
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperDiagMat(SuperMatAdd(self.arg, other).doit()).doit()
        else:
            return SuperMatAdd(self, other).doit()
    
    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, SuperDiagMat):
            return SuperDiagMat(SuperMatAdd(other.arg, self.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperDiagMat(SuperMatAdd(other, self.arg).doit()).doit()
        else:
            return SuperMatAdd(other, self).doit()
    
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, SuperDiagMat):
            return SuperDiagMat(SuperMatAdd(self.arg, -other.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperDiagMat(SuperMatAdd(self.arg, -other).doit()).doit()
        else:
            return SuperMatAdd(self, -other).doit()

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, SuperDiagMat):
            return SuperDiagMat(SuperMatAdd(other.arg, -self.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperDiagMat(SuperMatAdd(other, -self.arg).doit()).doit()
        else:
            return SuperMatAdd(other, -self).doit()
    
    """@call_highest_priority('__rmul__')
    def __mul__(self, other):
        return SuperMatMul(self, other).doit()
    
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return SuperMatMul(self, other).doit()
   
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return SuperMatMul(other, self).doit()
    
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return SuperMatMul(other, self).doit()"""
        
    def inverse(self):
        return SuperMatInverse(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
            
    def transpose(self):
        return self
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    def doit(self, **hints):
        return self
    
    @property
    def arg(self):
        return self.args[0]
    
    @property
    def name(self):
        return self.arg.name
    
    @property
    def shape(self):
        return (self.arg.shape[0], self.arg.shape[1])

class SuperBlockDiagMat(SuperMatBase, MatrixExpr):
    
    _op_priority = 10010
    
    def __new__(cls, mat, expanded=None, blockform=None):
        if isinstance(mat, SuperBlockDiagMat):
            return mat
        else:
            return MatrixExpr.__new__(cls, mat)
        
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = expanded
        self.blockform = blockform
        
        if (isinstance(mat, SuperMatSymbol) or isinstance(mat, SuperMatInverse) or
            isinstance(mat, SuperMatTranspose) or isinstance(mat, SuperDiagMat) or
            isinstance(mat, SuperBlockDiagMat)):
            self.mat_type = 'diag'
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim1
            self.variables_dim2 = mat.variables_dim2
            
        else:
            self.mat_type = None
            self.dep_vars = None
            self.cond_vars = None
            self.variables_dim1 = None
            self.variables_dim2 = None
            
    def __repr__(self):
        return 'blockdiag['+repr(self.arg)+']'
        
    def __str__(self):
        return 'blockdiag['+repr(self.arg)+']'
    
    def __neg__(self):
        return SuperMatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError
        
    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, SuperBlockDiagMat):
            return SuperBlockDiagMat(SuperMatAdd(self.arg, other.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperBlockDiagMat(SuperMatAdd(self.arg, other).doit()).doit()
        else:
            return SuperMatAdd(self, other).doit()
 
    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, SuperBlockDiagMat):
            return SuperBlockDiagMat(SuperMatAdd(other.arg, self.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperBlockDiagMat(SuperMatAdd(other, self.arg).doit()).doit()
        else:
            return SuperMatAdd(other, self).doit()
 
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, SuperBlockDiagMat):
            return SuperBlockDiagMat(SuperMatAdd(self.arg, -other.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperBlockDiagMat(SuperMatAdd(self.arg, -other).doit()).doit()
        else:
            return SuperMatAdd(self, -other).doit()
  
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, SuperBlockDiagMat):
            return SuperBlockDiagMat(SuperMatAdd(other.arg, -self.arg).doit())
        elif isinstance(other, MatMul) and any([isinstance(a, Identity) for a in other.args]):
            return SuperBlockDiagMat(SuperMatAdd(other, -self.arg).doit()).doit()
        else:
            return SuperMatAdd(other, -self).doit()

    """@call_highest_priority('__rmul__')
    def __mul__(self, other):
        return SuperMatMul(self, other).doit()
  
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return SuperMatMul(self, other).doit()
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return SuperMatMul(other, self).doit()
    
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return SuperMatMul(other, self).doit()"""
    
    def inverse(self):
        return SuperMatInverse(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
            
    def transpose(self):
        return SuperBlockDiagMat(self.arg.T).doit()
    
    T = property(transpose, None, None, 'Matrix transposition')
    
    def doit(self, **hints):
        return self
    
    @property
    def arg(self):
        return self.args[0]
    
    @property
    def name(self):
        return self.arg.name
    
    @property
    def shape(self):
        return (self.arg.shape[0], self.arg.shape[1])
        
class Variable(SuperMatSymbol):
    
    def __new__(cls, name, m, n):
        return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='var')
    
    def __init__(self, name, m, n):
        """
            Constructor for a Variable symbol
        
            Args:
                name - The variable name
                m, n - Its shape
        """
        SuperMatSymbol.__init__(self, m, n, name=name, mat_type='var')
                 
class Mean(SuperMatSymbol):
    
    def __new__(cls, v, name='', full_expr=None):
        if isinstance(full_expr, list):
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v], blockform=full_expr)
        elif full_expr is not None:
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v], expanded=full_expr)
        else:
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v])
        
    
    def __init__(self, v, name='', full_expr=None):
        """
            Constructor for a Mean symbol. This only works for distributions where variables aren't
            conditioned on others
        
            Args:
                v - The variable this symbol is for
                name - The name we supply to override the internal one
                full_expr - The expanded or blockform for this Mean symbol if it exists
                cond_vars - The variables
        """
        
        if isinstance(full_expr, list):
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v], blockform=full_expr)
        elif full_expr is not None:
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v], expanded=full_expr)
        else:
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[1], name=name, mat_type='mean', dep_vars=[v])

class Covariance(SuperMatSymbol):
    
    def __new__(cls, v, name='', full_expr=None):
        if isinstance(full_expr, list):
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v], blockform=full_expr)
        elif full_expr is not None:
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v], expanded=full_expr)
        else:
            return SuperMatSymbol.__new__(cls, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v])
    
    def __init__(self, v, name='', full_expr=None):
        """
            Constructor for a Covariance symbol. This only works for distributions where variables aren't
            conditioned on others
        
            Args:
                v - The variable this symbol is for
                name - The name we supply to override the internal one
                full_expr - The expanded or blockform for this Covariance symbol if it exists
        """
        
        if isinstance(full_expr, list):
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v], blockform=full_expr)
        elif full_expr is not None:
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v], expanded=full_expr)
        else:
            SuperMatSymbol.__init__(self, v.shape[0], v.shape[0], name=name, mat_type='covar', dep_vars=[v])
      
class Constant(SuperMatSymbol):
    
    def __new__(cls, name, m, n, full_expr=None):
        if isinstance(full_expr, list):
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other', blockform=full_expr)
        elif full_expr is not None:
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other', expanded=full_expr)
        else:
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other')
    
    def __init__(self, name, m, n, full_expr=None):
        """
            Constructor for a Constant symbol.
        
            Args:
                name - The variable name
                m, n - Its shape
        """
        
        if isinstance(full_expr, list):
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other', blockform=full_expr)
        elif full_expr is not None:
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other', expanded=full_expr)
        else:
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other')
    
    

        

