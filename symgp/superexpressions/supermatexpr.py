import string
from typing import List, Union, Any, Tuple, Iterable

from sympy import MatrixSymbol, BlockMatrix, Symbol, Inverse, Transpose, MatMul, MatAdd, ZeroMatrix, \
    MatrixExpr, S, Identity
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

Vector_t = List[MatrixExpr]
Matrix_t = List[List[MatrixExpr]]

class SuperMatSymbol(SuperMatBase, MatrixSymbol):
    _op_priority = 99
    _used_names = []

    ALLOWED_PARAM_TYPES = ['mean', 'cov', 'invcov']
    ALLOWED_MAT_TYPES = ALLOWED_PARAM_TYPES + ['var', 'sym', 'other', 'diag', 'blkdiag']

    def __new__(cls, m, n, name='', mat_type='other', dep_vars=None, cond_vars=None,
                expanded=None, blockform=None):
        """
        The SuperMatSymbol constructor.
        
        It is mainly used to automatically select the name of the symbol based on the parameters
        given. We have to do it here because this is the only way to set the name.

        (See ``__init__`` below for arguments)

        :return: An object of class type ``MatrixSymbol`` with
        """

        # Create name of symbol based on dep_vars and cond_vars if this is a parameter symbol e.g.
        # 'mean', 'cov', 'invcov', etc.
        if mat_type in SuperMatSymbol.ALLOWED_PARAM_TYPES and name == '':
            if mat_type == 'mean':
                if blockform is None and expanded is None:
                    pre_sym = SMALL_MU_GREEK
                else:
                    pre_sym = 'm'
            elif mat_type == 'cov':
                if blockform is None and expanded is None:
                    pre_sym = BIG_SIGMA_GREEK
                else:
                    pre_sym = 'S'
            else: # mat_type == 'invcov':
                if blockform is None and expanded is None:
                    pre_sym = BIG_SIGMA_GREEK + '_i'
                else:
                    pre_sym = 'S_i'

            name += pre_sym + '_{'

            if n != 1 and not isinstance(dep_vars[0], list):
                name += utils.create_distr_name([dep_vars, dep_vars], cond_vars) + '}'
            else:
                name += utils.create_distr_name(dep_vars, cond_vars) + '}'
        else:
            if cond_vars and mat_type not in SuperMatSymbol.ALLOWED_PARAM_TYPES:
                raise Warning("cond_vars should only be set for mat_type = {}".format(
                    SuperMatSymbol.ALLOWED_PARAM_TYPES))

        #if name not in SuperMatSymbol._used_names:
        SuperMatSymbol._used_names.append(name)
        #else:
        #    raise Exception("This name has been used: {}".format(name))

        return MatrixSymbol.__new__(cls, name, m, n)

    def __init__(self, m: Symbol, n: Symbol, name: str = '', mat_type: str = 'other',
                 dep_vars: Union[List[Any], List[List[Any]]] = None,
                 cond_vars: List[Any] = None, expanded: MatrixExpr = None,
                 blockform: Union[List[MatrixExpr], List[List[MatrixExpr]]] = None):
        """
        The SuperMatSymbol initialiser.
        :param name: The name of the symbol. Only used with mat_type == 'var' or 'other' (See
        '__new__'). Otherwise specify use: name=''. An error is raised if this is violated.
        :param m: Number of rows of matrix
        :param n: Number of columns of matrix
        :param mat_type: Type of matrix. Can be 'cov', 'invcov', 'mean', 'var' or 'other'
        :param dep_vars: The variables that this symbol depends on. These are the variables
        that this symbol is a function of which appear in the ``expanded/blockform`` directly
        or through other matrices.

        Leaving this to the default value means that there are no
        dependent variables and so ``expanded/blockform`` shouldn't depend on any variables.
        When ``blockform`` is specified, these variables are used to match the entries of the
        block matrix.

        For example, if ``blockform = [[S_xx, S_xy], [S_yx, S_yy]]`` (see ``blockform`` doc below)
        where the entries are in general ``MatrixExpr``s then ``dep_vars = [x, y]`` or ``dep_vars =[[
        x, y], [x, y]]`` where ``x`` and ``y`` are ``Variable``s (see below). The second form
        where we have a list of lists is useful when the ``blockform`` isn't necessarily square.
        The first list matches the first dimension of ``blockform`` and similarly for the second.
        :param cond_vars: The variables this parameter is conditioned on. Can only be used for
        'cov', 'invcov', 'mean', 'natmean', 'precision'.
        :param expanded: The full algebraic expression represented by this symbol. For example,
        ``expanded = A - B*C.I*D`` where the letters are ``MatrixExpr``s in general.
        :param blockform: A block matrix representation of the matrix. This contains the
        ``MatrixExpr``s in the block matrix stored as a list of lists. For example,
        if a ``SuperMatSymbol`` represents a 2x2 block matrix, ``blockform = [[A, B], [C,D]]``
        where the entries are ``MatrixExpr``s.
        """

        self.mat_type = mat_type
        self.dep_vars = dep_vars
        self.cond_vars = list(cond_vars) if cond_vars else []
        self.expanded = expanded
        self.blockform = blockform  # type: Union[List[MatrixExpr], List[List[MatrixExpr]]]
        self.variables_dim1 = {}
        self.variables_dim2 = {}

        if dep_vars:
            dep_vars = list(dep_vars)
            if isinstance(dep_vars[0], list):  # List of lists case
                for i in range(len(dep_vars[0])):
                    if dep_vars[0][i] not in self.variables_dim1:
                        self.variables_dim1[dep_vars[0][i]] = i

                for i in range(len(dep_vars[1])):
                    if dep_vars[1][i] not in self.variables_dim2:
                        self.variables_dim2[dep_vars[1][i]] = i
            else:  # All other cases
                for i in range(len(dep_vars)):
                    if dep_vars[i] not in self.variables_dim1:
                        self.variables_dim1[dep_vars[i]] = i

                if n != 1:
                    for i in range(len(dep_vars)):
                        if dep_vars[i] not in self.variables_dim2:
                            self.variables_dim2[dep_vars[i]] = i

    def doit(self, **hints):
        """
        We have to change MatrixSymbol.doit so that objects are constructed appropriately
        Modeled off MatrixSymbol.doit in https://github.com/sympy/sympy/blob/master/sympy/matrices/
        expressions/matexpr.py
        """
        return self

    def inverse(self):
        return SuperMatInverse(self)

    I = property(inverse, None, None, 'Matrix inversion')

    def transpose(self):
        if ((self.mat_type == 'cov' or self.mat_type == 'invcov' or self.mat_type == 'sym') and
                (self.shape[0] == self.shape[1])):
            return self
        else:
            return SuperMatTranspose(self)

    T = property(transpose, None, None, 'Matrix transposition')

    def partition(self, indices: Union[List[int], List[List[int]]]) -> Iterable:
        """ 
        Partition a ``blockform`` into sections defined by ``indices``.
            - With a mxn matrix, four partitions are created.
            - With a mx1 or 1xm matrix, two partitions are created
        :param indices: The indices that define which elements to group from the blockform. This
        is a 2-D list of lists for mxn matrices and 1-D list for mx1 or 1xm matrices where each
        list specifies the indices for each dimension
        :return: P, Q (, R, S) The partitions of the matrix. R and S are returned if matrix is mxn
        for m != 1 and n != 1.
        """

        if self.blockform is not None:
            # First check indices are valid
            if all([isinstance(l, list) for l in indices]) and len(indices) == 2:
                assert (all([isinstance(l, list) for l in self.blockform])), \
                    "self.blockform must be a 2-D list of lists for these indices"
                assert (all([i >= 0 and i < len(self.blockform) for i in indices[0]])), \
                    "Invalid first set of indices"
                assert (all([i >= 0 and i < len(self.blockform[0]) for i in indices[1]])), \
                    "Invalid first set of indices"

                P, Q, R, S = utils.partition_block(self.blockform, indices)

                # Sort the variables based on their index in 'self.blockform'
                variables_dim1_keys = sorted(self.variables_dim1.keys(),
                                             key=lambda m: self.variables_dim1[m])
                variables_dim2_keys = sorted(self.variables_dim2.keys(),
                                             key=lambda m: self.variables_dim2[m])

                # Get variables for both sets in both dimensions 
                indices_vars_dim1 = [v for v in variables_dim1_keys
                                     if self.variables_dim1[v] in indices[0]]
                indices_vars_dim2 = [v for v in variables_dim2_keys if
                                     self.variables_dim2[v] in indices[1]]
                rem_vars_dim1 = [v for v in variables_dim1_keys if
                                 self.variables_dim1[v] not in indices[0]]
                rem_vars_dim2 = [v for v in variables_dim2_keys if
                                 self.variables_dim2[v] not in indices[1]]

                # Get shapes of two sets in both dimensions
                m1 = sum([v.shape[0] for v in rem_vars_dim1])
                n1 = sum([v.shape[0] for v in rem_vars_dim2])
                m2 = sum([v.shape[0] for v in indices_vars_dim1])
                n2 = sum([v.shape[0] for v in indices_vars_dim2])

                # TODO: Maybe change this to have the type correspond to ``self``
                # Create partition symbols. self.mat_type must be one of {'cov', 'invcov',
                # 'precision', 'other'} ('other' must be 2-D)

                def get_partition(mat, m, n, dep_vars):
                    """Returns a corrected partition"""
                    if len(mat) == 1 and len(mat[0]) == 1:
                        return mat[0][0]
                    else:
                        return SuperMatSymbol(m, n, mat_type=self.mat_type, dep_vars=dep_vars,
                                              cond_vars=self.cond_vars, blockform=mat)

                P = get_partition(P, m1, n1, rem_vars_dim1)
                Q = get_partition(Q, m1, n2, [rem_vars_dim1, indices_vars_dim2])
                R = get_partition(R, m2, n1, [indices_vars_dim1, rem_vars_dim2])
                S = get_partition(S, m2, n2, indices_vars_dim1)

                return P, Q, R, S
            elif all([isinstance(l, int) for l in indices]):
                assert (all([not isinstance(l, list) for l in
                             self.blockform])), "self.blockform must be a 1-D list"
                assert (all([i >= 0 and i < len(self.blockform) for i in
                             indices])), "Invalid set of indices"

                P, S = utils.partition_block(self.blockform, indices)

                # Sort variables based on their index in 'self.blockform'
                variables_keys = sorted(self.variables_dim1.keys(),
                                        key=lambda m: self.variables_dim1[m])

                # Get variables for both sets
                indices_vars = [v for v in variables_keys if self.variables_dim1[v] in indices]
                rem_vars = [v for v in variables_keys if self.variables_dim1[v] not in indices]

                # Get shapes of two sets
                m1 = sum([v.shape[0] for v in rem_vars])
                m2 = sum([v.shape[0] for v in indices_vars])

                # If we partition a ``Variable``, we need to create a new name for each half
                if self.mat_type == 'var':
                    name1 = 'v_(' + ','.join([v.name for v in rem_vars]) + ')'
                    name2 = 'v_(' + ','.join([v.name for v in indices_vars]) + ')'
                else:
                    name1 = ''
                    name2 = ''

                # TODO: Maybe change this to have the type corresponding to ``self``
                if len(P) == 1:
                    P = P[0]
                else:
                    P = SuperMatSymbol(m1, 1, name1, mat_type=self.mat_type, dep_vars=rem_vars,
                                       blockform=P)

                if len(S) == 1:
                    S = S[0]
                else:
                    S = SuperMatSymbol(m2, 1, name2, mat_type=self.mat_type, dep_vars=indices_vars,
                                       blockform=S)

                return P, S
            else:
                raise Exception("Invalid set of indices")
        else:
            raise Exception("Operation can't be performed as there is no blockform.")

    def expand_partition(self):
        """
        Expands blockform to replace SuperMatSymbol representations with their blockforms.
        Only works with 2x2 block matrices or 1x2 block vectors (i.e. [a,b])
        """

        if self.blockform is not None:

            # Check size of blockform
            if len(self.blockform) == 2 and isinstance(self.blockform[0], list) and len(
                    self.blockform[0]) == 2:
                P_exp, Q_exp = self.blockform[0][0].blockform, self.blockform[0][1].blockform
                R_exp, S_exp = self.blockform[1][0].blockform, self.blockform[1][1].blockform

                # Check blockforms exist
                for i, block in enumerate([P_exp, Q_exp, R_exp, S_exp]):
                    if block is None:
                        raise Exception("Block matrix {}, {} doesn't have a blockform".format(
                            i//2, i%2))

                # Check that the shapes match
                assert (len(P_exp) == len(Q_exp))
                assert (len(P_exp[0]) == len(R_exp[0]))
                assert (len(Q_exp[0]) == len(S_exp[0]))
                assert (len(R_exp) == len(S_exp))

                # Create the top and bottom part of the larger matrix i.e.
                # top = [P_exp, Q_exp], bottom = [R_exp, S_exp]
                top = []
                for row1, row2 in zip(P_exp, Q_exp):
                    top.append(row1 + row2)

                bottom = []
                for row1, row2 in zip(R_exp, S_exp):
                    bottom.append(row1 + row2)

                self.blockform = top + bottom

            elif not isinstance(self.blockform[0], list) and len(self.blockform) > 1:
                P_exp = self.blockform[0].blockform
                S_exp = self.blockform[1].blockform

                self.blockform = P_exp + S_exp

            else:
                raise Exception("self.blockform isn't a 2x2 matrix or a 1x2 list")

        else:
            raise Exception("This symbol has no blockform")

    def to_full_expr(self):
        """
        Returns the full expression for the blockform or expanded of this symbol
        """

        if self.blockform is not None:
            if all([isinstance(l, list) for l in self.blockform]) and all([len(l) == len(
                    self.blockform[0]) for l in self.blockform]):
                m, n = len(self.blockform), len(self.blockform[0])

                full_expr_blockform = []

                for i in range(m):
                    full_expr_blockform.append(
                        [utils.expand_to_fullexpr(s) for s in self.blockform[i]])

                return full_expr_blockform
            elif all([isinstance(e, MatrixExpr) for e in self.blockform]):
                return [utils.expand_to_fullexpr(s) for s in self.blockform]
            else:
                raise Exception("self.blockform is invalid: {}".format(self.blockform))
        elif self.expanded is not None:
            return utils.expand_to_fullexpr(self)
        else:
            return self

    @staticmethod
    def getUsedNames():
        return SuperMatSymbol._used_names

    @staticmethod
    def used(name: str):
        """
        Checks whether name is used
        """
        return name in SuperMatSymbol._used_names

class SuperMatInverse(SuperMatBase, Inverse):
    _op_priority = 10000

    def __new__(cls, mat: SuperMatBase, expanded: MatrixExpr=None,
                 blockform: List[List[MatrixExpr]]=None):
        """
        See '__init__' below for definition of parameters
        """
        return Inverse.__new__(cls, mat)

    def __init__(self, mat: SuperMatBase, expanded: MatrixExpr=None,
                 blockform: List[List[MatrixExpr]]=None):
        """
        Creates an inverse matrix for the given ``SuperMatBase`` derived symbol
        :param mat: The matrix/matrix expression to calculate the inverse of.
        :param expanded: The expanded expression if it exists. If this isn't specified, we
        calculate the ``self.expanded`` automatically from that in ``mat``.
        :param blockform: The block matrix expression. Must be an N x N list of lists. If this
        isn't specified, we calculate the ``self.blockform`` automatically from that in ``mat``.
        """
        if not blockform is None:
            assert len(blockform) == len(blockform[0]), "blockform must be square"


        self.expanded = expanded
        self.blockform = blockform

        if any([isinstance(mat, sym) for sym in [SuperMatSymbol, SuperMatInverse,
                                                SuperMatTranspose, SuperDiagMat]]):
            if not mat.blockform is None:
                assert len(mat.blockform) == len(mat.blockform[0]), \
                    "``mat`` must have a square blockform"

            if mat.mat_type == 'cov':
                self.mat_type = 'invcov'
            elif mat.mat_type == 'invcov':
                self.mat_type = 'cov'
            else:
                self.mat_type = mat.mat_type

            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim1
            self.variables_dim2 = mat.variables_dim2

            if not mat.expanded is None and not self.expanded is None:
                self.expanded = mat.expanded.I

            if not mat.blockform is None and len(mat.blockform) > 1 and not self.blockform is None:
                self.blockform = utils.matinv(mat.blockform)

                vars_dim1 = sorted(mat.variables_dim1.keys(), key=lambda m: mat.variables_dim1[m])
                vars_dim2 = sorted(mat.variables_dim2.keys(), key=lambda m: mat.variables_dim2[m])

                for i in range(len(self.blockform)):
                    for j in range(len(self.blockform[0])):
                        var_i, var_j = vars_dim1[i], vars_dim2[j]
                        self.blockform[i][j] = SuperMatSymbol(var_i.shape[0], var_j.shape[0],
                                                              mat_type=self.mat_type,
                                                              dep_vars=[var_i, var_j],
                                                              cond_vars=mat.cond_vars,
                                                              expanded=self.blockform[i][j])
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

    def __new__(cls, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        See '__init__' below for definition of parameters
        """
        return Transpose.__new__(cls, mat)

    def __init__(self, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        Creates a transpose matrix for the given ``SuperMatBase`` derived symbol
        :param mat: The matrix/matrix expression to calculate the transpose of.
        :param expanded: The expanded expression if it exists. If this isn't specified, we
        calculate the ``self.expanded`` automatically from that in ``mat``.
        :param blockform: The block matrix expression. If this isn't specified, we calculate the
        ``self.blockform`` automatically from that in ``mat``.
        """
        self.expanded = expanded
        self.blockform = blockform

        if any([isinstance(mat, sym) for sym in [SuperMatSymbol, SuperMatInverse,
                                                 SuperMatTranspose]]):
            self.mat_type = mat.mat_type
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim2
            self.variables_dim2 = mat.variables_dim1

            if not mat.expanded is None and not self.expanded is None:
                self.expanded = mat.expanded.T

            if not mat.blockform is None and not self.blockform is None:
                if isinstance(mat.blockform, Vector_t):
                    self.blockform = []
                    self.blockform.extend([x.T for x in mat.blockform])
                elif isinstance(mat.blockform, Matrix_t):
                    self.blockform = [[0 for _ in range(len(mat.blockform))] for _ in
                                      range(len(mat.blockform[0]))]

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

    def __new__(cls, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        See '__init__' below for definition of parameters
        """
        if isinstance(mat, SuperDiagMat):
            return mat
        elif isinstance(mat, SuperBlockDiagMat):
            return SuperDiagMat(mat.arg, expanded=mat.expanded, blockform=mat.blockform)
        else:
            return MatrixExpr.__new__(cls, mat)

    def __init__(self, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        Creates a diagonal matrix for the given ``SuperMatBase`` derived symbol
        :param mat: The matrix/matrix expression to calculate the diagonal matrix of.
        :param expanded: The expanded expression if it exists.
        :param blockform: The block matrix expression.
        """
        self.expanded = expanded
        self.blockform = blockform

        if any([isinstance(mat, sym) for sym in [SuperMatSymbol, SuperMatInverse,
                                                 SuperMatTranspose, SuperDiagMat,
                                                 SuperBlockDiagMat]]):
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
        return 'diag[' + repr(self.arg) + ']'

    def __str__(self):
        return 'diag[' + str(self.arg) + ']'

    def _sympystr(self, *args, **kwargs):
        return self.__str__()

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

    def __new__(cls, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        See '__init__' below for definition of parameters
        """
        if isinstance(mat, SuperBlockDiagMat) or isinstance(mat, SuperDiagMat):
            return mat
        else:
            return MatrixExpr.__new__(cls, mat)

    def __init__(self, mat: SuperMatBase, expanded: MatrixExpr = None,
                blockform: List[List[MatrixExpr]] = None):
        """
        Creates a block diagonal matrix for the given ``SuperMatBase`` derived symbol
        :param mat: The matrix/matrix expression to calculate the block diagonal matrix of.
        :param expanded: The expanded expression if it exists.
        :param blockform: The block matrix expression.
        """
        self.expanded = expanded
        self.blockform = blockform

        if all([isinstance(mat, sym) for sym in [SuperMatSymbol, SuperMatInverse,
                                                 SuperMatTranspose, SuperDiagMat,
                                                 SuperBlockDiagMat]]):
            self.mat_type = 'blkdiag'
            self.dep_vars = mat.dep_vars
            self.cond_vars = mat.cond_vars
            self.variables_dim1 = mat.variables_dim1
            self.variables_dim2 = mat.variables_dim2

        else:
            self.mat_type = 'blkdiag'
            self.dep_vars = None
            self.cond_vars = None
            self.variables_dim1 = None
            self.variables_dim2 = None

    def __repr__(self):
        return 'blkdiag[' + repr(self.arg) + ']'

    def __str__(self):
        return 'blkdiag[' + repr(self.arg) + ']'

    def _sympystr(self, *args, **kwargs):
        return self.__str__()

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
    def __new__(cls, name: str, m: Union[Symbol, int], n: Union[Symbol, int]):
        return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='var')

    def __init__(self, name: str, m: Union[Symbol, int], n: Union[Symbol, int]):
        """
        Constructor for a Variable symbol
        :param name: The variable name
        :param m: Number of rows
        :param n: Number of columns
        """
        SuperMatSymbol.__init__(self, m, n, name=name, mat_type='var')

class CompositeVariable(SuperMatSymbol):
    """
    Represents a vector of individual ``Variable`` or ``CompositeVariable`` objects
    """

    def __new__(cls, name: str, variables):
        assert all([v.shape[1] == 1 for v in variables])
        m = sum([v.shape[0] for v in variables])
        n = variables[0].shape[1]
        return SuperMatSymbol.__new__(cls, m, n, name=name, blockform=variables, mat_type='var')

    def __init__(self, name: str, variables):
        """
        Creates a combined variable from a list of ``Variable``s and/or ``CompositeVariable``s
        :param name: The name for this combined variable.
        :param variables: The list of ``Variable``/``CompositeVariable`` objects. They must all
        have shape (?,1) where ``?`` can vary for each variable
        """
        m = sum([v.shape[0] for v in variables])
        n = variables[0].shape[1]
        SuperMatSymbol.__init__(self, m, n, name=name, blockform=variables, mat_type='var')

class Mean(SuperMatSymbol):
    def __new__(cls, v: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]],
                cond_vars: List[Union[Variable, CompositeVariable]]=None, name: str='',
                full_expr: Union[MatrixExpr, Vector_t]=None):
        # Create name
        if name == '':
            if full_expr is None:
                pre_sym = SMALL_MU_GREEK
            else:
                pre_sym = 'm'

            name += pre_sym + '_{'
            if isinstance(v, list):
                name += utils.create_distr_name(v, cond_vars) + '}'
            else:
                name += utils.create_distr_name([v], cond_vars) + '}'

        full_full_expr = utils.expand_to_fullexpr(full_expr) if isinstance(full_expr, MatrixExpr) else \
            full_expr
        if not full_expr is None and isinstance(full_full_expr, Mean) and \
                        full_full_expr.name == name:
            return full_expr

        if isinstance(v, list):
            shape = sum([t.shape[0] for t in v])
            shape = (shape, 1)
            variables = v
        else:
            shape = v.shape
            variables = [v]

        if isinstance(full_expr, list):
            assert utils.is_vector(full_expr), "full_expr must be a 1-D vector (i.e. a list)"
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='mean',
                                          dep_vars=variables, cond_vars=cond_vars, blockform=full_expr)
        elif isinstance(full_expr, MatrixExpr):
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='mean',
                                          dep_vars=variables, cond_vars=cond_vars, expanded=full_expr)
        elif full_expr is None:
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='mean',
                                          dep_vars=variables, cond_vars=cond_vars)
        else:
            raise Exception("Invalid full_expr provided: {}. Must be a list ("
                            "blockform), MatrixExpr (expanded) or None (no "
                            "expanded/blockform)".format(full_expr))

    def __init__(self, v: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]],
                 cond_vars: List[Union[Variable, CompositeVariable]]=None, name: str='',
                 full_expr: Union[MatrixExpr, Vector_t]=None):
        """
        Constructor for a Mean symbol. This only works for distributions where variables aren't
        conditioned on others.
        :param v: The random variable this symbol is a mean for
        :param cond_vars: The optional conditioned-on variables for the (implicit) distribution
        that this mean symbol is a parameter of.
        :param name: Optional name for this mean symbol. If left as is, a name is created
        automatically.
        :param full_expr: The expanded or blockform for this Mean symbol if it exists.

        """
        full_full_expr = utils.expand_to_fullexpr(full_expr) if isinstance(full_expr, MatrixExpr) else \
            full_expr
        if not full_expr is None and isinstance(full_full_expr, Mean) and full_full_expr.name == \
                self.name:
            print("self.name: ", self.name)
            print("name: ", name)
            return

        if isinstance(v, list):
            shape = sum([t.shape[0] for t in v])
            shape = (shape, 1)
            variables = v
        else:
            shape = v.shape
            variables = [v]

        if isinstance(full_expr, list):
            assert utils.is_vector(full_expr), "full_expr must be a 1-D vector (i.e. a list)"
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='mean',
                                    dep_vars=variables, cond_vars=cond_vars, blockform=full_expr)
        elif isinstance(full_expr, MatrixExpr):
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='mean',
                                    dep_vars=variables, cond_vars=cond_vars, expanded=full_expr)
        elif full_expr is None:
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='mean',
                                    dep_vars=variables, cond_vars=cond_vars)
        else:
            raise Exception("Invalid full_expr provided: {}. Must be a list ("
                            "blockform), MatrixExpr (expanded) or None (no "
                            "expanded/blockform)".format(full_expr))

class Covariance(SuperMatSymbol):
    def __new__(cls,
                v1: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]],
                v2: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]] = None,
                cond_vars: List[Union[Variable, CompositeVariable]] = None, name: str='',
                full_expr: Union[MatrixExpr, Matrix_t]=None):

        if v2 is None:
            v2 = v1

        variables = [v1, v2]
        # Create name
        if name == '':
            if full_expr is None:
                pre_sym = BIG_SIGMA_GREEK
            else:
                pre_sym = 'S'

            name += pre_sym + '_{'
            if isinstance(v1, list) and isinstance(v2, list):
                name += utils.create_distr_name(variables, cond_vars) + '}'
            elif ((isinstance(v1, Variable) or isinstance(v1, CompositeVariable)) and
                      (isinstance(v2, Variable) or isinstance(v2, CompositeVariable))):
                name += utils.create_distr_name(variables, cond_vars) + '}'
            else:
                raise Exception("v1 and v2 must be the same. They can either be a list of "
                                "CompositeVariable/Variable or CompositeVariable/Variables themselves.")

        full_full_expr = utils.expand_to_fullexpr(full_expr) if isinstance(full_expr, MatrixExpr) else \
            full_expr

        if not full_expr is None and isinstance(full_full_expr, Covariance) and \
                        full_full_expr.name == name:
            return full_expr

        if isinstance(v1, list) and isinstance(v2, list):
            shape_v1 = sum([v.shape[0] for v in v1])
            shape_v2 = sum([v.shape[0] for v in v2])
            #assert shape_v1 == shape_v2, "Both lists of variables must have same shape"
            shape = (shape_v1, shape_v2)
        elif ((isinstance(v1, Variable) or isinstance(v1, CompositeVariable)) and
                (isinstance(v2, Variable) or isinstance(v2, CompositeVariable))):
            #assert v1.shape[0] == v2.shape[0], "Both variables must have same shape"
            shape = (v1.shape[0], v2.shape[0])

            # Get unique variables
            variables = [v1] if v1 == v2 else [v1, v2]
        else:
            raise Exception("v1 and v2 must be the same. They can either be a list of "
                            "CompositeVariable/Variable or CompositeVariable/Variables themselves.")

        if isinstance(full_expr, list):
            assert utils.is_square(full_expr), "full_expr must be a square matrix"
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='cov',
                                          dep_vars=variables, cond_vars=cond_vars,
                                          blockform=full_expr)
        elif isinstance(full_expr, MatrixExpr):
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='cov',
                                          dep_vars=variables, cond_vars=cond_vars,
                                          expanded=full_expr)
        elif full_expr is None:
            return SuperMatSymbol.__new__(cls, shape[0], shape[1], name=name, mat_type='cov',
                                          dep_vars=variables, cond_vars=cond_vars)
        else:
            raise Exception("Invalid full_expr provided: {}. Must be a list of lists ("
                            "blockform), MatrixExpr (expanded) or None (no "
                            "expanded/blockform)".format(full_expr))

    def __init__(self,
                 v1: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]],
                 v2: Union[Variable, CompositeVariable, List[Union[Variable, CompositeVariable]]]=None,
                 cond_vars: List[Union[Variable, CompositeVariable]] = None, name: str='',
                 full_expr: Union[MatrixExpr, Matrix_t]=None):
        """
        Constructor for a Covariance symbol. This only works for distributions where variables aren't
        conditioned on others.
        :param v1: The first argument of the covariance matrix.
        :param v2: The second argument of the covariance matrix. If this isn't specified,
        then we set v2 = v1.
        :param cond_vars: The optional conditioned-on variables for the (implicit) distribution
        that this covariance symbol is a parameter of.
        :param name: Optional name for this covariance symbol. If left as is, a name is created
        automatically
        :param full_expr: The expanded or blockform for this Covariance symbol if it exists.
        """
        full_full_expr = utils.expand_to_fullexpr(full_expr) if isinstance(full_expr, MatrixExpr) \
            else full_expr
        if not full_expr is None and isinstance(full_full_expr, Covariance) and \
                        full_full_expr.name == self.name:
            print("self.name: ", self.name)
            print("name: ", name)
            return

        if v2 is None:
            v2 = v1

        variables = [v1, v2]
        if isinstance(v1, list) and isinstance(v2, list):
            shape_v1 = sum([v.shape[0] for v in v1])
            shape_v2 = sum([v.shape[0] for v in v2])
            #assert shape_v1 == shape_v2, "Both lists of variables must have same shape"
            shape = (shape_v1, shape_v2)
        elif ((isinstance(v1, Variable) or isinstance(v1, CompositeVariable)) and
                (isinstance(v2, Variable) or isinstance(v2, CompositeVariable))):
            #assert v1.shape[0] == v2.shape[0], "Both variables must have same shape"
            shape = (v1.shape[0], v2.shape[0])

            # Get unique variables
            variables = [v1] if v1 == v2 else [v1, v2]
        else:
            raise Exception("v1 and v2 must be the same. They can either be a list of "
                            "CompositeVariable/Variable or CompositeVariable/Variables themselves.")


        if isinstance(full_expr, list):
            assert utils.is_square(full_expr), "full_expr must be a square matrix"
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='cov',
                                    dep_vars=variables, cond_vars=cond_vars, blockform=full_expr)
        elif isinstance(full_expr, MatrixExpr):
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='cov',
                                    dep_vars=variables, cond_vars=cond_vars, expanded=full_expr)
        elif full_expr is None:
            SuperMatSymbol.__init__(self, shape[0], shape[1], name=name, mat_type='cov',
                                    dep_vars=variables, cond_vars=cond_vars)
        else:
            raise Exception("Invalid full_expr provided: {}. Must be a list of lists ("
                            "blockform), MatrixExpr (expanded) or None (no "
                            "expanded/blockform)".format(full_expr))

class Constant(SuperMatSymbol):
    def __new__(cls, name: str, m: Union[Symbol, int], n: Union[Symbol, int],
                full_expr: Union[MatrixExpr, Vector_t, Matrix_t]=None):
        if isinstance(full_expr, list):
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other',
                                          blockform=full_expr)
        elif full_expr is not None:
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other',
                                          expanded=full_expr)
        else:
            return SuperMatSymbol.__new__(cls, m, n, name=name, mat_type='other')

    def __init__(self, name: str, m: Union[Symbol, int], n: Union[Symbol, int],
                 full_expr: Union[MatrixExpr, Vector_t, Matrix_t]=None):
        """
        Constructor for a Constant symbol.
        :param name: The variable name
        :param m: The number of rows
        :param n: The number of columns
        :param full_expr: The detailed expression that this symbol represents. This can be a
        standard ``MatrixExpr``, a list of ``MatrixExpr``s (representing a 1-D vector) or a list of
        lists of ``MatrixExpr``s (representing a block matrix of matrix expressions)
        """

        if isinstance(full_expr, list):
            assert utils.is_vector(full_expr) or utils.is_matrix(full_expr), \
                "Invalid full_expr list. Must be a 1-D list or if it is 2-D (list of lists), " \
                "length of each list must be the same"
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other', blockform=full_expr)
        elif isinstance(full_expr, MatrixExpr):
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other', expanded=full_expr)
        elif full_expr is None:
            SuperMatSymbol.__init__(self, m, n, name=name, mat_type='other')
        else:
            raise Exception("Invalid full_expr provided: {}. Must be a list or list of "
                            "lists (blockform), MatrixExpr (expanded) or None (no "
                            "expanded/blockform)".format(full_expr))
