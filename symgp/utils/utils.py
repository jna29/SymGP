
from __future__ import print_function, division

from typing import Union, Optional, List, Iterable, Dict, Any
from collections import defaultdict
from abc import ABCMeta
import copy
import re
import string
import math
import logging
import sys

import numpy as np

from sympy import (MatMul, MatAdd, Basic, MatrixExpr, MatrixSymbol, ZeroMatrix, Symbol, Identity, Transpose,
                   Inverse, Number, Rational, ln, Determinant, pi, sympify, srepr, S, Expr, Matrix)
from sympy.printing.latex import LatexPrinter
from sympy.core.evaluate import global_evaluate
from sympy.core.compatibility import iterable, ordered, default_sort_key


# GREEK symbols
SMALL_MU_GREEK = '\u03bc'
BIG_SIGMA_GREEK = '\u03a3'
SMALL_SIGMA_GREEK = '\u03c3'
BIG_OMEGA_GREEK = '\u03a9'
BIG_LAMBDA_GREEK = '\u039b'
SMALL_ETA_GREEK = '\u03b7' 

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOG.addHandler(ch)

######## Matrix operations with lists as matrices ########
def _mul_with_num(num, mat):
    """
        Used by 'matmul'
    
        Multiplies a matrix/vector represented as a list (see 'matmul') with a number.
    
        Args:
            num - A number to multiply all elements of mat with
            mat - A list/list of lists representing a vector/matrix (see 'matmul')
    
        Returns:
            mat of same shape with elements multiplied with num
    """
    
    from symgp.superexpressions import SuperMatMul
    
    if isinstance(mat[0], list):
        new_mat = [[SuperMatMul(num,mat[i][j]).doit() for j in range(len(mat[0]))] for i in range(len(mat))]
    else:
        new_mat = [SuperMatMul(num, v).doit() for v in mat]
    
    return new_mat

def _check_shape_matmul(mat, order):
    """
        Checks size of 'mat' and reshapes if necessary.
    
        Args:
            mat - A list/list of lists representing a vector/matrix (see 'matmul')
            order - Indicates whether mat is a left/right matrix/vector such that
                    we can broadcast appropriately.
    
        Returns:
            (m,n) - Tuple giving the shape of mat
            broadcast - Boolean indicating whether we should broadcast list
    """
    
    broadcast_list = False
    if isinstance(mat[0],list):
        m = len(mat)
        n = len(mat[0])
    elif order == 'left':
        m = 1
        n = len(mat)
        broadcast_list = True
    else:  # order == 'right'
        m = len(mat)
        n = 1
        broadcast_list = True
    
    return m, n, broadcast_list
     
def matmul(list1, list2):
    """
        Multiply two lists in a matrix fashion.
    
        Similar to numpy's matrix multiplication of arrays:
            - If list1 has shape (m1,) (i.e. it is a 1-D list) it is broadcast to (1,m1).
              Here we take the transpose of all the elements as we assume.
              list2 must have shapes (m1,n2) or (m1,) otherwise an Exception is raised.
              A list of shape (n2,) or (1,) is returned.
            - If list2 has shape (m2,) it is broadcast to (m2,1).
              list1 must have shapes (m2,) or (m1,m2) otherwise an Exception is raised.
              A list of shape (1,) or (m1,) is returned.
            - Any other case requires the shapes to match.

        For example, we can call this as:
            - matmul([[A, B], [C, D]], [a, b])
            - matmul([[A, B], [C, D]], [[a], [b]])
        
        All elements (A, B, C, D, a, b) are all SuperMatSymbols where the shapes must match.
    
        Multiplying all elements in a list by a number is also supported e.g. matmul(a,5) or matmul(5,a).     
    """
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
        
    # Handle multiplication by integers
    if isinstance(list1, int):
        return _mul_with_num(list1, list2)
        
    if isinstance(list2, int):
        return _mul_with_num(list2, list1)

    # Check sizes and reshape if necessary
    m1, n1, broadcast_list1 = _check_shape_matmul(list1, 'left')
    m2, n2, broadcast_list2 = _check_shape_matmul(list2, 'right')
    
    # Check shapes
    if n1 != m2:
        raise Exception("Shapes don't match: %s, %s" % ((m1, n1), (m2, n2)))
    
    # Multiply based on types of lists
    if broadcast_list1 and broadcast_list2: # (1,n1) x (m2,1)
        out_list = [SuperMatAdd(*[SuperMatMul(list1[i],list2[i]).doit() for i in range(n1)]).doit()]
    elif broadcast_list1:  # (1,n1) x (m2,n2)
        out_list = [0 for _ in range(n2)]
        for i in range(n2):
            out_list[i] = SuperMatAdd(*[SuperMatMul(list1[j],list2[j][i]).doit() for j in range(m2)]).doit()
    elif broadcast_list2:  # (m1,n1) x (m2,1)
        out_list = [0 for _ in range(m1)]
        for i in range(m1):
            out_list[i] = SuperMatAdd(*[SuperMatMul(list1[i][j],list2[j]).doit() for j in range(m2)]).doit()
    else: # (m1,n1) x (m2,n2) 
        out_list = [[0 for _ in range(n2)] for _ in range(m1)]
        for i in range(m1):
            for j in range(n2):
                out_list[i][j] = SuperMatAdd(*[SuperMatMul(list1[i][k],list2[k][j]).doit() for k in range(n1)]).doit()
           
    return out_list

def _check_shape_matadd(mat):
    """
        Determines matrix shape of given matrix (as defined in matmul) 'mat'
    
        Args:
            mat - A list/list of lists representing a vector/matrix (see 'matmul')
    
        Returns:
            m, n - The shape of mat
    """
    
    if isinstance(mat[0],list):
        m = len(mat)
        n = len(mat[0])
    else:
        m = 0
        n = len(mat)
    
    return m, n

def _assert_shapes(m1,n1,m2,n2):
    """
        Checks whether shapes match
    """
    
    if m1 != m2 or n1 != n2:
        raise Exception("Shapes don't match: %s, %s" % ((m1, n1), (m2, n2)))
        
def matadd(list1, list2):
    """
        Adds two lists that must be the same shape. We reshape list of (m,) to (0,m).
    
        Returns a list of the same shape as the lists.
    """
    from symgp.superexpressions import SuperMatAdd
    
        
    # Check sizes
    m1, n1 = _check_shape_matadd(list1)
    m2, n2 = _check_shape_matadd(list2)

    # Check shapes match
    _assert_shapes(m1,n1,m2,n2)
    
    # Shape out_list based on whether list1 is 1-D.
    if m1 == 0:
        out_list = [SuperMatAdd(list1[i],list2[i]).doit() for i in range(n1)]
    else:
        out_list = [[SuperMatAdd(list1[i][j],list2[i][j]).doit() for j in range(n1)] for i in range(m1)]
    
        
    return out_list

def mattrans(mat):
    """
        Returns the transpose of an mxn matrix (list of lists)
    
        Arg:
            mat - A list/list of lists representing a vector/matrix (see 'matmul')
    
        Returns:
            mat_T - A transpose of shape n x m where mat has shape m x n. If mat has
                    shape (m,) we simply return mat where each element is the 
                    transpose of its corresponding element in mat.
    """
    
    if all([not isinstance(e,list) for e in mat]): # (m,) case
        return [e.T.doit() for e in mat]
    else: # Other case
        if any([not isinstance(e,list) for e in mat]):
            raise Exception("mat is not a regular matrix")
        m_T = len(mat[0])
        n_T = len(mat)
        mat_T = [[mat[j][i].T.doit() for j in range(n_T)] for i in range(m_T)]  
        return mat_T
    
def matinv(mat):
    """
        Inverts nxn matrices. 
    
        Args:
            mat - A list/list of lists representing a vector/matrix (see 'matmul') of
                  shape (n,n)
        
        Returns:
            If n > 2, we first partition then apply the algorithm again.
            If n == 1, we simply return the SuperMatInverse of the element.
    """
    
    if any([not isinstance(e,list) for e in mat]):
        raise Exception("This is not a suitable matrix")
    
    if len(mat) != len(mat[0]):
        raise Exception("This isn't a square matrix.")
        
    n = len(mat)
    
    # Recursively calculate the inverse to get the large untidy expression
    if n == 1:
        return [[mat[0][0].I]]
    else:
        if n == 2:
            P, Q = [[mat[0][0]]], [[mat[0][1]]]
            R, S = [[mat[1][0]]], [[mat[1][1]]]
        else:
            P, Q, R, S = partition_block(mat,[len(mat)-1,len(mat[0])-1])
        
        P_bar = matinv(matadd(P,matmul(matmul(matmul(-1,Q),matinv(S)),R)))
        Q_bar = matmul(matmul(matmul(-1,P_bar),Q),matinv(S))
        R_bar = matmul(matmul(matmul(-1,matinv(S)),R),P_bar)
        S_bar = matadd(matinv(S),matmul(matmul(matmul(matmul(matinv(S),R),P_bar),Q),matinv(S)))
        
        # Create new matrix by top bottom method i.e. create top of matrix then create bottom
        top = []
        for row1, row2 in zip(P_bar,Q_bar):
            top.append(row1+row2)
        
        bottom = []
        for row1, row2 in zip(R_bar,S_bar):
            bottom.append(row1+row2)
        
        return top+bottom

def _copy_block(block):
    """
        Makes a copy of block as used by 'partition_block'
    """
    
    new_block = []
    if isinstance(block[0], list):
        for row in block:
            new_block.append(list(row))
    else: #isinstance(block, list)
        new_block = list(block)
    
    return new_block

def is_matrix(block):
    """
        Returns true if block is a matrix.
    
        A matrix must be a Python list of lists where each list has length
        greater than 1 and all lists must be same length
    """
    
    return (all([isinstance(r,list) for r in block]) and all([len(block[0])==len(r) for r in block]))

def is_1d_vector(block):
    """
    Returns True if ``block is a 1-d list.
    :param block: A list.
    """
    return all([not isinstance(e, list) for e in block])

def is_2d_vector(block):
    """
    Returns True if ``block is a 2-d list.
    :param block: A list.
    """
    return all([isinstance(r, list) for r in block]) and all([len(r) == 1 for r in block])

def is_vector(block):
    """
        Returns true if block is a vector.
    
        A vector must be:
            - A Python list where each element is not a list e.g. [a,b,c]
            - A Python list of lists where each list has length 1 e.g. [[a],[b],[c]]
    """
    
    return is_1d_vector(block) or is_2d_vector(block)

def is_square(block):
    """
        Determines whether block is a square matrix.
    """    
    
    return is_matrix(block) and (len(block[0]) == len(block))

def _move_cols_to_end(block, indices):
    """
        Moves the columns given by indices to the end of block 
        preserving the order of the columns
    
        For example if:
    
            block = [[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9,10,12,13]] 
    
            indices=[1,2]
    
        we get
    
            block = [[1, 4, 2, 3],
                     [5, 8, 6, 7],
                     [9,13,10,12]]
    """
    
    num_rows, num_cols = len(block), len(block[0])
    
    indices = sorted(indices,reverse=True)
    new_block = _copy_block(block)
    
    for idx, col in enumerate(indices):
        if col == num_cols-1:
            continue
        else:
            c = col
            # Shifts column to last available column i.e. 
            while c < num_cols-(idx+1):
                for row in range(num_rows):
                    temp = new_block[row][c]
                    new_block[row][c] = new_block[row][c+1]
                    new_block[row][c+1] = temp
                c += 1
    
    return new_block

def _move_elems_to_end(block, indices):
    """
        Moves the elements in vector 'block' at locations given in 'indices' to the end
        of the block whilst preserving the order of the elements
    """
    
    indices = sorted(indices,reverse=True)
    
    block_size = len(block)
    
    new_block = _copy_block(block)
    
    # Push elements corresponding to indices to end of list
    for idx, k in enumerate(indices):
        if k == block_size-1:
            continue
        else:
            i = k
            while i < block_size-(idx+1):
                temp = new_block[i]
                new_block[i] = new_block[i+1]
                new_block[i+1] = temp
                i += 1
    
    return new_block
                    
def partition_block(block, indices):
    """
        Partitions a list into four or two sections based on the indices
    
        Args:
            block - The input block to be partitioned: 
                        - If block is 2-D, we partition it into [[P, Q], [R, S]]
                        - If block is 1-D, we partition it into [a, b] (shape=(m,)) or 
                          [[a],[b]] (shape=(m,1))
            indices - The indices that form one partition:
                        - If block is 2-D, this can be 2-D (e.g. [[1,2],[0,1]]) or 1-D (e.g. [1,2,3]). 
                          If 1-D, block needs to be square.
                        - If block is 1-D, this should also be a vector (e.g. [1,2,3] or [[1],[2],[3]])
                      Repeat indices are removed automatically. The order of the columns/rows are
                      preserved.
    
                      For example for block = [[A,B,C,D],[E,F,G,H],[I,J,K,L],[M,N,O,Z]] and 
                      indices = [[0,2],[1,3]], we get:
    
                            P = [[E,G],[M,O]]       Q = [[F,H],[N,Z]]
                            R = [[A,C],[I,K]]       S = [[B,D],[J,L]]
                                
        Returns:
          Either
            P, Q, R, S - The four partitions for 2-D blocks
          Or
            a, b - The two partitions for 1-D blocks
    """
    
    # Checks validity of indices values
    _is_valid_idx = lambda idx, max_idx: all([(i >= 0 and i < max_idx) for i in idx])
    
    
    
    # Check block is a correct block matrix/vector ([[...],[...]] or [...])
    if not (is_matrix(block) or is_vector(block)):
        raise Exception("The block to be partitioned must be a matrix ([[A,B], [C,D]]) or \
                         vector ([a,b] or [[a],[b]])")
    
    # Copy block    
    new_block = _copy_block(block)
    
    if is_matrix(new_block) and not is_vector(new_block):
        num_rows, num_cols = len(new_block), len(new_block[0])
        
        # Check indices are appropriate for matrix
        if (all([isinstance(e,int) for e in indices]) and is_square(new_block)):
            indices = [indices, indices]   # Convert to 2-D
        else:
            if not all([isinstance(e,list) for e in indices]):
                raise Exception("Incorrect form for indices for a matrix. Must be a list of lists e.g.\
                                 [[1,2],[3]] or a 1-D list [1,2] if the matrix is square")
        
        # Remove repeat set of indices
        row_indices = list(set(indices[0]))
        col_indices = list(set(indices[1]))
                      
        # Check for 1x1 case
        if num_rows == 1 and num_cols == 1:
            raise Exception("Can't partition a 1x1 block. Minimum size is 2x2")
            
        # Check that all indices are in appropriate range
        if not _is_valid_idx(row_indices,num_rows):
            raise Exception("Invalid row indices. Must be in range: [%s,%s]" % (0,num_rows-1))
        
        if not _is_valid_idx(col_indices,num_cols):
            raise Exception("Invalid column indices. Must be in range: [%s,%s]" % (0,num_cols-1))
        
          
        # First push columns indicated by indices to end
        new_block = _move_cols_to_end(new_block, col_indices)
        # Do same for rows
        new_block = list(map(list,zip(*new_block)))   # Flip rows and columns
        new_block = _move_cols_to_end(new_block, row_indices)
        new_block = list(map(list,zip(*new_block)))
           
        m = num_rows - len(row_indices)   # Number of rows of partition not given by indices
        n = num_cols - len(col_indices)   # Number of columns of partition not given by indices 
            
        # Create partitions
        P = [new_block[i][:n] for i in range(m)]  # No row and col indices
        Q = [new_block[i][n:] for i in range(m)]  # No row but col indices
        R = [new_block[i][:n] for i in range(m, num_rows)]  # No col but row indices
        S = [new_block[i][n:] for i in range(m, num_rows)]  # Intersection of row and col indices
                
        return P, Q, R, S
    else:   # Vector
        block_size = len(new_block)
        
        # Check for 1x1 case
        if block_size == 1:
            raise Exception("Can't partition a 1x1 block")
        
        # Check indices are appropriate for vector
        if is_vector(indices):
            if all([isinstance(e,list) for e in indices]):   # Convert to 1-D list
                indices = [e[0] for e in indices]
        else:
            raise Exception("Incorrect form of indices. Must be 1-D e.g. [1,2]")
        
        # Check that all indices are in appropriate range
        if not _is_valid_idx(indices,block_size):
            raise Exception("Invalid indices. Must be in range: [%s,%s]" % (0,block_size-1))
            
        # Remove duplicates
        indices = list(set(indices))
        
        new_block = _move_elems_to_end(new_block,indices)
        
        # Partition    
        m1 = block_size - len(indices)
        a = new_block[:m1]
        b = new_block[m1:]
        
        return a, b


######## MVG helper functions ########
def get_logZ(cov):
    """
    Calculates log-normalising constant symbol using cov
    """
    return -cov.shape[0]/2*ln(2*pi) - Rational(1,2)*ln(Determinant(cov))


######### Search and replace functions ########
def replace_with_num(expr, d):
    """
        Replaces matrix symbols with numerical matrices using a DFS search through the
        expression tree.
    
        Args:
            - 'expr': The expression which we want to evaluate.
            - 'd': A dictionary mapping the matrix symbols to numerical matrices (these can be
                   SymPy 'Matrix' objects or 'numpy.ndarray' arrays).

        Returns:
            - A 'numpy.ndarray' that is the evaluation of the expr with the numerical
              matrices.
        
    """

    import numpy as np
    
    
    # Determine what to return based on type(expr)
    if isinstance(expr, MatrixSymbol):
        try:
            return d[expr.name]
        except KeyError as e:
            print("Error: No numerical matrix was specified for %s" % (e))
    elif isinstance(expr, Number):
        return expr
    elif isinstance(expr, MatrixExpr):
        
        sub_exprs = []
        for arg in expr.args:
            sub_exprs.append(replace_with_num(arg, d))
              
        if expr.is_MatMul:
            for e in sub_exprs:
                if not isinstance(e,Number):
                    shape = e.shape[0]
                    break
                    
            out = np.eye(shape)
            for e in sub_exprs:
                if isinstance(e,Number):
                    out *= np.float(e)
                elif isinstance(e,Matrix):
                    out = np.dot(out,np.array(e.tolist(),dtype=np.float32))
                else:
                    out = np.dot(out,e)
            return out
        elif expr.is_MatAdd:
            if len(sub_exprs[0].shape) == 2:
                out = np.zeros(sub_exprs[0].shape)
            else:
                out = np.zeros(sub_exprs[0].shape[0])
            
            for e in sub_exprs:
                if isinstance(e,Matrix):
                    out += np.array(e.tolist(),dtype=np.float32).reshape(out.shape)
                else:
                    out += e.reshape(out.shape)
            return out
        elif expr.is_Inverse:
            if isinstance(sub_exprs[0],Matrix):
                out = np.linalg.inv(np.array(sub_exprs[0].tolist(),dtype=np.float32))
            else:
                out = np.linalg.inv(sub_exprs[0])
            return out
        else: # expr.is_Transpose
            if isinstance(sub_exprs[0],Matrix):
                out = np.array(sub_exprs[0].T.tolist(),dtype=np.float32)
            else:
                out = sub_exprs[0].T
            return out
    else:
        raise Exception("Expression should be a MatrixExpr")

def evaluate_expr(expr, d):
    """
        Evaluates a matrix expression with the given numerical matrices
    
        Args:
            - 'expr' - The symbolic matrix expression
            - 'd' - A dictionary mapping the matrix symbols to numerical matrices
    
        Returns:
            - 'r' - The result of all the matrix calculations
    """
    
    r = replace_with_num(expr, d)
    
    return r
    
def replace_with_expanded(expr, done=True, excluded=None):
    """
        Similar to 'replace_with_num' above except we replace SuperMatrixSymbols
        with their expanded forms if they exist
    
        Args:
            expr - The current MatrixExpr
        
        Returns:
            expr - The expanded MatrixExpr
            done - Boolean indicating whether no more expansions can be done
    """
    
    from symgp.superexpressions import (SuperMatSymbol, SuperMatTranspose, SuperMatInverse, SuperMatAdd, 
                                        SuperMatMul, SuperDiagMat, SuperBlockDiagMat)
    from symgp.kernels.kernel import KernelMatrix
     
    if (isinstance(expr, MatMul) or isinstance(expr, MatAdd) or 
        isinstance(expr, Inverse) or isinstance(expr, Transpose)):
        
        sub_exprs = []
        for arg in expr.args:
            expanded, done = replace_with_expanded(arg, done, excluded=excluded)
            sub_exprs.append(expanded)
            
        if expr.is_MatMul:
            e = SuperMatMul(*sub_exprs)
        elif expr.is_MatAdd:
            e = SuperMatAdd(*sub_exprs)
        elif expr.is_Inverse:
            e = SuperMatInverse(*sub_exprs)
        else: # expr.is_Transpose
            e = SuperMatTranspose(*sub_exprs)
        
        return e, done
    elif excluded is not None and any([isinstance(expr, c) for c in excluded]):
        return expr, done
    elif isinstance(expr, SuperMatSymbol) and expr.expanded is not None:
        return expr.expanded, False
    else:
        return expr, done

def expand_to_fullexpr(expr, num_passes=-1, excluded=None):
    """
        Expands a MatrixExpr composed of SuperMatSymbols by substituting any SuperMatSymbol
        with an 'expanded'
    
        Args:
            expr - The expression to expand
            num_passes - The number of passes to make through the expression. -1 indicates that
                         we pass through expression until no more substitutions can be made.
            excluded - The classes (and its subclasses) to exclude substitution with expanded
                       expressions for.
    
        Return:
            e - The expanded expression
    """
    
    e = expr

    # Keep on passing through expression until no more substitutions can be made
    if num_passes == -1:
        done = False
        while not done:
            done = True
            e, done = replace_with_expanded(e, done, excluded=excluded)
    else:
        for _ in range(num_passes):
            e, _ = replace_with_expanded(e, excluded=excluded)
        
    return e.doit().doit()

def _replace_with_MatSym(expr, rule):
    """
        Replaces the MatrixExpr expression in 'expr' given by the replacement rule
    
        Args:
            expr - The expression which we want to replace sub-expressions in.
            rule - A tuple that matches the old expression (old_expr) to the replacement (repl) as
                   (old_expr, repl)
    
        Returns:
            subbed_expr - 'expr' with the substitution made
    """
    
    from collections import deque
    from symgp import SuperDiagMat, SuperBlockDiagMat
    
    old_expr, repl = rule
    
    len_old_expr = len(old_expr.args)   # Number of arguments. TODO: Check for cases where k is a single symbol
    
    # Table used to build back tree. 
    #
    # We pair a key of a sub_expression with an id 'k' that indicates sub_expr was the k'th entry into the table with either:
    #
    #       - A list of (sub_expr.args[i], k) tuples indicating the keys from which to search for the
    #         next expressions in the tree in their correct order:
    #
    #                     {(sub_expr, j): [(sub_expr.args[0],m),(sub_expr.args[1],l), ...]}
    #         
    #       - A Expr that we substitute in for sub_expr when it is retrieved by higher nodes in the expression tree:
    #
    #                     {(sub_expr, j): sub_expr_repl}   
    #
    #         where sub_expr_repl is the expression that we replace sub_expr with. It can be sub_expr itself or a replacement
    #         we define.
    tree_table = defaultdict(list)
    
    queue = deque(((expr, 0, 0),))
    
    #tree_table[(full_expr,0)] = list(zip(list(full_expr.args),[1]*len(full_expr.args)))
    
    curr_id = 1  # An id to uniquely identify each sub-expression i.e. we can have similar expressions at the same level
    while len(queue) > 0:
        sub_expr, level, old_id = queue.pop()
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
        
            # Match current rule to expressions in this sub expression
            
            len_sub_expr = len(sub_expr.args)
            
            i = 0
            while i < len_sub_expr:
                j = 0
                l = 0    # Used when we need to skip over symbols e.g. for addition where we may need to match a subset of args.
                
                matched, skipped = _match_with_pat(sub_expr,i,old_expr)
                      
                if matched:  # Match found: Replace match with pattern
                    # Determine the level of the new replacement expression in the expression tree
                    if len_old_expr == len_sub_expr:
                        new_level = level
                    else:
                        new_level = level + 1
        
                    queue.appendleft((repl, new_level, curr_id))
                                              
                    # We need to re-order sub_expr - mainly for matches in MatAdds with remainders e.g. matching A in A + B + C
                    if skipped:
                        old_sub_expr = sub_expr
                        
                        # Get remainder after removing old_expr
                        rem = sub_expr
                        for c in old_expr.args:
                            rem -= c
                        
                        rem = [rem] if not isinstance(rem,MatAdd) else list(rem.args)

                        # Create new expression
                        new_args = list(old_expr.args) + rem
                        sub_expr = type(sub_expr)(*new_args)
                        
                        # As we changed the sub_expr we have to reassign the elements of the old one
                        if tree_table.get((old_sub_expr, level, old_id)):
                            old_values = tree_table.pop((old_sub_expr, level, old_id))
                            tree_table[(sub_expr, level, old_id)] = old_values + [(repl, new_level, curr_id)]
                        else:
                            tree_table[(sub_expr, level, old_id)] = [(repl, new_level, curr_id)]
                        
                        curr_id += 1
                            
                    else:    
                        # Check entry for sub_expr exists
                        tree_table[(sub_expr, level, old_id)].append((repl, new_level, curr_id))
                        
                        curr_id += 1
                          
                    # Start after pattern     
                    i += len_old_expr
                else:
                    queue.appendleft((sub_expr.args[i], level+1, curr_id))
                    
                    # Check entry for sub_expr exists
                    tree_table[(sub_expr, level, old_id)].append((sub_expr.args[i], level+1, curr_id))
                    
                    curr_id += 1
                    
                    # Start at next symbol     
                    i += 1
                
        else:
            # Add expression for this node
            tree_table[(sub_expr, level, old_id)] = sub_expr
                   
    
    # Sort based on level in descending order
    sorted_tree_table = sorted(tree_table.items(), key=lambda elem: elem[0][1], reverse=True) 
    
    # Create expression from table
    for p, c in sorted_tree_table:
        
        # Skip terminal nodes else update tree table for non-terminal nodes
        if p[0] == c:
            continue
        else:
            # Create MatrixExpr using the elements in the value c, which is a list, for the key p and
            # then update 'tree_table'
            tree_table[p] = type(p[0])(*[tree_table[e] for e in c])  
    
    # Rewrite full expression    
    subbed_expr = tree_table[sorted_tree_table[-1][0]]
    
    return subbed_expr
      
def _match_with_pat(expr, start, pat):
    """
        Matches an expression or a portion of it to a pattern.
    
        Args:
            expr - The expression we want to match.
            start - The starting index into expr
            pat - The pattern we want to find in 'expr'. This can be:
                    - A MatrixExpr. Here we aim to find pat in 'expr'
                    - A Kernel. We aim to find KernelMatrix objects/MatrixExprs 
                      composed of KernelMatrix objects that match Kernel
    
        Returns:
            matched - Indicates whether the pattern was found in 'expr'.
            skipped - Indicates whether we had to skip over symbols when matching 
                      in a MatAdd expression.
          (Optional)
            pattern - The pattern that we match. Only returned for when pat is a Kernel
            repl - The replacement expression. Only returned for when pat is a Kernel
        
        Examples:
            - expr = A*B*D, pat = A*B -> matched = True, skipped = False
            - expr = A + B + C, pat = A + C -> matched = True, skipped = True (as we had to skip over B)
              Note that 'skipped' is determined based on the order of expr.args.
            - expr = K(a,u)*K(u,u)*K(u,b), pat = Q (Q.sub_kernels=[K,K], Q.M=K(u,u)) -> matched = True, skipped = True
              (We match the whole expression with Q), pattern = K(a,u)*K(u,u)*K(u,b), repl = Q(a,b)
    """
    
    from symgp import Kernel
    
    len_expr = len(expr.args)
    matched, skipped = False, False
    #print("Pat: ", pat, type(pat))
    #print("Expr: ", expr)
    if isinstance(pat, MatrixExpr):
        if isinstance(pat, MatrixSymbol):
            from symgp import SuperMatMul
            pat = SuperMatMul(pat)

        #print("Expr: ", expr)
        len_pat = len(pat.args)
        j = 0
        l = 0
        while j < len_pat and start + l + j < len_expr:
            if start + l + j >= len_expr:
                break

            #print("Current expr: ", expr.args[start+l+j])
            #print("current pat: ", pat.args[j])
            if (expr.args[start+l+j].doit() != pat.args[j].doit()):# or (sub_expr.args[i+l+j].match(k.args[j])):
                #print("Not matched")
                #foundMatch = False
                # As additions may be stored in any order, we need to skip symbols so that we can match
                # the pattern
                if isinstance(pat, MatAdd) and isinstance(expr, MatAdd):
                    l += 1
                else:
                    break
            else:
                #print("Matched")
                j += 1
        
        if j == len_pat:
            matched = True
            #print("Matched full expr")
            
        if l > 0:
            skipped = True
        
        return matched, skipped
    elif isinstance(pat, Kernel):
        #print("Kernel pat: ", pat)
        kern_vars = get_all_kernel_variables(expr)
        
        # Get all possible kernel patterns
        patterns = []
        #print("kern_vars: ", kern_vars)
        for v1 in kern_vars:
            patterns.extend([pat(v1,v2) for v2 in kern_vars])

        # Sort patterns based on length of underlying expression
        def sort_func(e):
            e_full = e.to_full_expr()
            if isinstance(e_full, MatrixSymbol):
                return 1
            else:
                return len(e_full.args)

        patterns = sorted(patterns, key=sort_func)[::-1]

        #print("patterns: ", patterns)

        # Find a match in our list of patterns
        for i, p in enumerate(patterns):
            #print("p: ",p," patterns[i]: ",patterns[i])
            #print("p.expanded: ", p.expanded)
            #print("p.to_full_expr(): ", p.to_full_expr())
            #print("expr, start: ", expr, start)
            #print("patterns: ", patterns)
            #print("matched_pat: ", p.to_full_expr())
            matched, skipped = _match_with_pat(expr, start, p.to_full_expr())
            #print("matched, skipped, matched_pat, repl: ", matched, skipped, p.to_full_expr(), p)
            if matched:
                return matched, skipped, p.to_full_expr(), p

        return matched, skipped, None, None

            
    else:
        raise Exception("Invalid pattern 'pat': Must be a Kernel object or a MatrixExpr")
      
def _replace_with_Kernel(expr, kern):
    """
        Replaces the kernel expression in 'expr' given by the replacement rule
    
        Args:
            expr - The expression which we want to replace sub-expressions in.
            kern - The Kernel we want to replace an expression in 'expr' with. The expression
                   belongs to the set of expression 'kern' represents.
    
                   For example, if
                         
                         M = Constant('M',n,n,full_expr=K(u,u).I)
                         kern = Kernel(sub_kernels=[K,K],kernel_type='mul',mat=M,name='Q')

                   we replace all expressions of form K({v1},u)*K(u,u).I*K(u,{v2}) where {v1} and {v2} can be 
                   any variable.
    
        Returns:
            subbed_expr - 'expr' with the substitution made
    """
    
    from collections import deque
    from symgp import Kernel, SuperDiagMat, SuperBlockDiagMat
    
    # Table used to build back tree. 
    #
    # We pair a key of a sub_expression with an id 'k' that indicates sub_expr was the k'th entry in the table with either:
    #
    #       - A list of (sub_expr.args[i], k) tuples indicating the keys from which to search for the
    #         next expressions in the tree in their correct order:
    #
    #                     {(sub_expr, j): [(sub_expr.args[0],m),(sub_expr.args[1],l), ...]}
    #         
    #       - A Expr that we substitute in for sub_expr when it is retrieved by higher nodes in the expression tree:
    #
    #                     {(sub_expr, j): sub_expr_repl}   
    #
    #         where sub_expr_repl is the expression that we replace sub_expr with. It can be sub_expr itself or a replacement
    #         we define.
    
    tree_table = defaultdict(list)
    
    if isinstance(kern,Kernel):
        queue = deque(((expr, 0, 0),))    # Tuples of (expression, tree level, expression id)
    
        curr_id = 1  # An id to uniquely identify each sub-expression i.e. we can have similar expressions at the same level
        while len(queue) > 0:
            sub_expr, level, old_id = queue.pop()

            #print("sub_expr: {}, level: {}, old_id: {}".format(sub_expr, level, old_id))

            if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
                isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
                isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
                
                # TODO: Add functionality to replace expressions such as A+D in D + 2*A.
            
                len_sub_expr = len(sub_expr.args)
            
                i = 0
                while i < len_sub_expr:

                    matched, skipped, pattern, repl = _match_with_pat(sub_expr, i, kern)

                    #print("i: ", i)
                    #print("Matched: {}, Skipped: {}, Pattern: {}, Repl: {}".format(
                    #    matched, skipped, pattern, repl))


                    # Update 'tree_table'
                    if matched:  # Match found: Replace match with pattern
                        # Determine the level of the new replacement expression in the expression tree
                        if len(pattern.args) == len_sub_expr:
                            new_level = level
                        else:
                            new_level = level + 1
        
                        queue.appendleft((repl, new_level, curr_id))
                                              
                        # We need to re-order sub_expr - mainly for matches in MatAdds with
                        # remainders e.g. matching A in A + B + C
                        if skipped:
                            old_sub_expr = sub_expr
                        
                            # Get remainder after removing old_expr
                            rem = sub_expr
                            for c in pattern.args:
                                rem -= c
                        
                            rem = [rem] if not isinstance(rem,MatAdd) else list(rem.args)

                            # Create new expression
                            new_args = list(pattern.args) + rem
                            sub_expr = type(sub_expr)(*new_args)
                        
                            # As we changed the sub_expr we have to reassign the elements of the old one
                            if tree_table.get((old_sub_expr, level, old_id)):
                                old_values = tree_table.pop((old_sub_expr, level, old_id))
                                tree_table[(sub_expr, level, old_id)] = old_values + [(repl, new_level, curr_id)]
                            else:
                                tree_table[(sub_expr, level, old_id)] = [(repl, new_level, curr_id)]
                        else:
                            # Check entry for sub_expr exists
                            tree_table[(sub_expr, level, old_id)].append((repl, new_level, curr_id))

                        #print("Key: {}, Val: {}".format((sub_expr, level, old_id),
                        #                                (repl, new_level, curr_id)))

                        # Start after pattern
                        if isinstance(pattern, MatrixSymbol):
                            i += 1
                        else:
                            i += len(pattern.args)

                    else:
                        #print("Key: {}, Val: {}".format((sub_expr, level, old_id),
                        #                                (sub_expr.args[i], level+1, curr_id)))

                        queue.appendleft((sub_expr.args[i], level+1, curr_id))
                    
                        # Check entry for sub_expr exists
                        tree_table[(sub_expr, level, old_id)].append((sub_expr.args[i], level+1, curr_id))
                    
                        # Start at next symbol     
                        i += 1 
                    
                    curr_id += 1
            else:
                # Add expression for this node
                tree_table[(sub_expr, level, old_id)] = sub_expr
    else:
        raise Exception("Invalid 'old_expr': Should be a Kernel, MatMul or MatAdd object")
    
    # Sort based on level in descending order
    sorted_tree_table = sorted(tree_table.items(), key=lambda elem: elem[0][1], reverse=True) 

    #print("\n")

    #for ele in sorted_tree_table:
    #    print(ele)

    #print("\n")

    # Create expression from table
    for p, c in sorted_tree_table:
    
        # Skip terminal nodes else update tree table for non-terminal nodes
        if p[0] == c:
            continue
        else:
            # Create MatrixExpr using the elements in the value c, which is a list, for the key p and
            # then update 'tree_table'
            tree_table[p] = type(p[0])(*[tree_table[e] for e in c])

        #print("p: {}, tree_table[p]: {}".format(p, tree_table[p]))
    
    subbed_expr = tree_table[sorted_tree_table[-1][0]]

    return subbed_expr
              
def replace(expr, rules):
    """
        Replaces expressions in expr with the given rules.
    
        Args:
            expr - The input expression
            rules - A list where elements can be:
                         - A tuple matching an old MatrixExpr to a new MatSym, or
                         - A Kernel object that has an underlying representation of matrix expressions
                           to match e.g. 
                
                                M = Constant('M',n,n,full_expr=K(u,u).I)
                                Q = Kernel(sub_kernels=[K,K],kernel_type='mul',mat=M,name='Q')
    
                           matches expressions of the form 'K({v1},u)*K(u,u).I*K(u,{v2})' where {v1} and {v2} can be 
                           any variable
                    N.B. For an expression of the form -1*A we must replace it with another expression
                         of the form -1*B and not A with B.
    
        Returns:
            The expression with the substitutions made.
    """

    from symgp import Kernel
    
    # Get the full expression
    #full_expr = expand_to_fullexpr(expr)
    full_expr = expr
    
    # For each substitution rule, replace the corresponding sub-expression
    for r in rules:
        if isinstance(r,Kernel):
            full_expr = _replace_with_Kernel(full_expr, r)
        elif isinstance(r,tuple) and isinstance(r[0],MatrixExpr) and isinstance(r[1],MatrixSymbol):
            full_expr = _replace_with_MatSym(full_expr, r)
        else:
            raise Exception("Invalid matching of expressions to replacements. Rule must be (old_expr,repl) or kern")
    
    return full_expr.doit()
             
def replace_with_SuperMat(expr, d):
    """
        Similar to replace_with_num above except we replace symbols with the 
        corresponding SuperMatExpr symbols
    """
    
    from symgp.superexpressions import SuperMatMul, SuperMatAdd, SuperMatInverse, SuperMatTranspose
    
    if isinstance(expr, Symbol) or isinstance(expr, Number):
        if isinstance(expr, Symbol):
            try:
                return d[expr.name]
            except KeyError as e:
                print("Error: No SuperMatSymbol substitute was specified for %s" % (e))
        else:
            return expr
        
    r = []
    for arg in expr.args:
        r.append(replace_with_SuperMat(arg, d))
        
    if isinstance(expr, Expr):
        if expr.is_Mul:
            return SuperMatMul(*r)
        elif expr.is_Add:
            return SuperMatAdd(*r)
        elif expr.is_Inverse:
            return SuperMatInverse(*r)
        else: # expr.is_Transpose
            return SuperMatTranspose(*r)
    else:
        raise Exception("Expression should be a MatrixExpr")
             
    
######## LaTeX printing ########
class matLatPrinter(LatexPrinter):
        
    def _print_Symbol(self, expr):
        if expr.name[0] == SMALL_SIGMA_GREEK:
            return self._print(Symbol('\sigma'+expr.name[1:]))
        else:
            return LatexPrinter().doprint(expr)
    
    def _print_SuperMatSymbol(self, expr):
        mat_type = expr.mat_type
        #print("mat_type: ",mat_type)
        #print("expr: ",expr)
        name = expr.name
        """if (mat_type == 'mean' or mat_type == 'covar' or mat_type == 'invcovar' or
            mat_type == 'natmean' or mat_type == 'precision'):
            dep_vars = expr.dep_vars
            cond_vars = expr.cond_vars
            if mat_type == 'mean':
                if not isinstance(dep_vars[0],list):
                    name = '\mu_{'+','.join([self._print(v) for v in dep_vars])
                else:
                    dep_vars = list(set(dep_vars[0]).union(set(dep_vars[1])))
                    name = '\mu_{'+','.join([self._print(v) for v in dep_vars])
            elif mat_type == 'covar':
                if not isinstance(dep_vars[0],list):
                    name = '\Sigma_{'+','.join([self._print(v) for v in dep_vars])
                else:
                    dep_vars = list(set(dep_vars[0]).union(set(dep_vars[1])))
                    name = '\Sigma_{'+','.join([self._print(v) for v in dep_vars]) 
            elif mat_type == 'invcovar':
                if not isinstance(dep_vars[0],list):
                    name = '\Sigma^{-1}_{'+','.join([self._print(v) for v in dep_vars])
                else:
                    dep_vars = list(set(dep_vars[0]).union(set(dep_vars[1])))
                    name = '\Sigma^{-1}_{'+','.join([self._print(v) for v in dep_vars])
            elif mat_type == 'natmean':
                if not isinstance(dep_vars[0],list):
                    name = '\eta_{1,'+','.join([self._print(v) for v in dep_vars])
                else:
                    dep_vars = list(set(dep_vars[0]).union(set(dep_vars[1])))
                    name = '\eta_{1,'+','.join([self._print(v) for v in dep_vars])
            else: # mat_type == 'precision'
                if not isinstance(dep_vars[0],list):
                    name = '\eta_{2,'+','.join([self._print(v) for v in dep_vars])
                else:
                    dep_vars = list(set(dep_vars[0]).union(set(dep_vars[1])))
                    name = '\eta_{2,'+','.join([self._print(v) for v in dep_vars])   
            
            if len(cond_vars) > 0:
                name += '|'+','.join([self._print(v) for v in cond_vars])
            
            name += '}'
        
            return name
        else: # Just use the Symbol converted latex form
            if expr.name[-2:] == '_s':
                return r'\mathbf{'+expr.name[:-2]+'}_{*}'
            else:
                return r'\mathbf{'+expr.name+'}'"""
        return r'\mathbf{' + expr.name + '}'

    def _print_SuperMatInverse(self, expr):
        return self._print(expr.args[0]) +'^{-1}'
                 
    def _print_SuperMatTranspose(self, expr):
        return  self._print(expr.args[0]) +'^T'
    
    def _print_SuperDiagMat(self, expr):
        return r"\text{diag}["+self._print(expr.arg)+"]"
    
    def _print_SuperBlockDiagMat(self, expr):
        return r"\text{blockdiag}["+self._print(expr.arg)+"]"
    
    def _print_SuperMatAdd(self, expr):
        terms = list(expr.args)
        
        # Fix to stop first negative term being rendered as -1 in LaTex i.e. we want
        #   A - BCB^{T} in LaTex instead of  -1BCB^{T} + A
        if terms[0].args[0] == S.NegativeOne:
            terms = terms[1:] + terms[:1]
        
        tex = " + ".join(map(self._print, terms))
        return tex
            
    
    def _print_MVG(self, expr):
        from symgp.kernels.kernel import KernelMatrix

        excluded_classes = [KernelMatrix]
        
        # Form MVG name
        latex_name = r'\begin{align*}' + "\n"
        latex_name += expr.prefix+r'\left('
        vars_name_pre = ','.join([self._print(v) for v in expr.variables])   # Name of vars
        if len(expr.cond_vars) > 0:
            vars_name_pre += '|'+','.join([self._print(v) for v in expr.cond_vars])
        latex_name += vars_name_pre + r'\right)'
        
        # N(mean, covar)
        latex_name += r'&= \mathcal{N}\left('
        if len(expr.variables) > 1:
            vars_name_N = r'\left[\begin{smallmatrix}'
            for i in range(len(expr.variables)-1):
                vars_name_N += self._print(expr.variables[i])+r'\\'
            vars_name_N += self._print(expr.variables[-1])+r'\end{smallmatrix}\right]'
            
            # Mean
            mean_short_name = r'\mathbf{m}_{'+vars_name_pre+r'}'
            
            if expr.mean.blockform is not None:
                mean_name = r'\left[\begin{smallmatrix}'
                for i in range(len(expr.mean.blockform)-1):
                    mean_name += self._print(expand_to_fullexpr(
                        expr.mean.blockform[i], excluded=excluded_classes).doit())+r'\\'
                mean_name += self._print(expand_to_fullexpr(
                    expr.mean.blockform[-1], excluded=excluded_classes).doit())+r'\end{smallmatrix}\right]'
            
            # Covariance
            covar_short_name = r'\mathbf{\Sigma}_{'+vars_name_pre+r'}'
            
            if expr.covar.blockform is not None:
                covar_name = r'\left[\begin{smallmatrix}'
                for i in range(len(expr.covar.blockform)-1):
                    for j in range(len(expr.covar.blockform[i])-1):
                        covar_name += self._print(expand_to_fullexpr(
                            expr.covar.blockform[i][j], excluded=excluded_classes).doit())+r'&'
                    covar_name += self._print(expand_to_fullexpr(
                        expr.covar.blockform[i][-1], excluded=excluded_classes).doit())+r'\\'
            
                # Add last row
                for j in range(len(expr.covar.blockform[-1])-1):
                    covar_name += self._print(expand_to_fullexpr(
                        expr.covar.blockform[-1][j], excluded=excluded_classes).doit())+r'&'
                covar_name += self._print(expand_to_fullexpr(
                    expr.covar.blockform[-1][-1], excluded=excluded_classes).doit())+r'\end{smallmatrix}\right]'
            
            # Write shortened distribution expression
            latex_name += vars_name_N + r';' + mean_short_name + r',' + covar_short_name + r'\right)\\'+"\n"
            
            
        else:
            mean_short_name = r'\mathbf{m}_{'+vars_name_pre+r'}'
            mean_name = self._print(expand_to_fullexpr(
                expr.mean.expanded, excluded=excluded_classes).doit()) if expr.mean.expanded is not None else ''
            covar_short_name = r'\mathbf{\Sigma}_{'+vars_name_pre+r'}'
            covar_name = self._print(expand_to_fullexpr(
                expr.covar.expanded, excluded=excluded_classes).doit()) if expr.covar.expanded is not None else ''
            
            # Write shortened distribution expression
            var_name_N = self._print(expr.variables[0])
            latex_name += var_name_N + r';' + mean_short_name+r','+covar_short_name+r'\right)\\' + "\n"
        
        
        # Add full expressions for mean and covariance below
        if mean_name != '' and covar_name != '':
            latex_name += mean_short_name + r' &= ' + mean_name + r'\\' + "\n" + \
                          covar_short_name + r' &= ' + covar_name + r'\\' + "\n"
        
        latex_name += r'\end{align*}'
            
        return latex_name
    
    def _print_Identity(self, expr):
        return r'\mathbf{I}'
    
    #def _print_NegativeOne(self, expr):
    #    return r'-'
    
    def _print_ZeroMatrix(self, expr):
        return r'\mathbf{0}'
    
    def _print_KernelMatrix(self, expr):
        latex_name = r'\mathbf{'+expr.K.name+'}_{'+matLatex(expr.inputs[0])+','+\
                     matLatex(expr.inputs[1])+'}'
        return latex_name
             
def matLatex(expr, profile=None, **kwargs):
    """
        Returns the LaTeX code for the given expression
    """
    
    if profile is not None:
        profile.update(kwargs)
    else:
        profile = kwargs
    out_latex = matLatPrinter(profile).doprint(expr)
    
    #Clean up string
    out_latex = re.sub('(\+.\-1)','-',out_latex) # Change '+ -1' to '-'
    
    return out_latex

def updateLatexDoc(filename, expr):
    """
        Updates the latex filename with the given expression.
    
        This function is mainly used to typeset the LaTeX code that is produced by calling
        utils.matLatex(expr). 
    
        We append the expression to the list of 'dmath' environments from the breqn package.
        For MVGs we also display the full expressions for the mean and covariance below the 
        expression for the distribution.
    
        Args:
            filename - The '.tex' file to which we write the LaTeX code.
            expr - The expression (or list of expressions) for which we want to generate LaTeX. 
                   This can be any native SymPy expression (and the subclasses in this library) 
                   or an MVG. 
                
    """

    import subprocess
    from MVG import MVG
    
    with open(filename,'r+') as f:
        contents = f.read()
        
        split_contents = re.split(r"(.+\\begin\{document\}\n)(.+)(\\end\{document\}.*)", contents, flags=re.DOTALL)
        
        edited_content = split_contents[2]
        
        
        if edited_content == '\n':
            edited_content = ''
            
        if not isinstance(expr, list):
            expr = [expr]
                
        for e in expr:
            # Write our expression to the end of the file
                
            if isinstance(e, MVG):    
                edited_content += r'\section{$'+ matLatex(e.name) + r'$}' + "\n"
                edited_content += r'\begingroup\makeatletter\def\f@size{12}\check@mathfonts'+ "\n" + \
                                  r'\def\maketag@@@#1{\hbox{\m@th\large\normalfont#1}}'+ "\n"
                edited_content += matLatex(e)
                edited_content += r'\endgroup'+ "\n\n"
            else:
                edited_content += r'\section{expression}' + "\n"
                edited_content += "\\begin{align*}\n"
                edited_content += matLatex(e)
                edited_content += "\n\\end{align*}\n"
        
        split_contents[2] = edited_content
        
        f.seek(0)
        f.write(''.join(split_contents))
        f.truncate()
        
    
    subprocess.check_call(["latexmk", "-pdf",str(filename)])
    subprocess.check_call(["open", filename.split(".")[0]+".pdf"])


######## Expression conversion functions ########
def expand_mat_sums(sums):
    """
        Helper method for 'expand_matmul' 
        Based on 'def _expandsums' in sympy.core.mul
    """
    from symgp.superexpressions.supermatadd import SuperMatAdd, SuperMatMul
             
    L = len(sums)
        
    if L == 1:
        return sums[0]
    terms = []
    left = expand_mat_sums(sums[:L//2]).args
    right = expand_mat_sums(sums[L//2:]).args
            
    terms = [a*b for a in left for b in right]
    added = SuperMatAdd(*terms)
        
    return added

def expand_matmul(expr):
    """
        Expands MatMul objects e.g. C*(A+B) -> C*A + C*B
        Based on 'def _eval_expand_mul' in sympy.core.mul
    """
    from symgp.superexpressions import SuperMatAdd
         
    sums, rewrite = [], False
    for factor in expr.args:
        if isinstance(factor, MatrixExpr) and factor.is_MatAdd:
            sums.append(factor)
            rewrite = True
        else:
            sums.append(Basic(factor))
    
    if not rewrite:
        return expr
    else:
        if sums:
            terms = expand_mat_sums(sums).args
                
            args = []
            for term in terms:
                t = term
                if isinstance(t,MatrixExpr) and t.is_MatMul and any(a.is_MatAdd if isinstance(a,MatrixExpr) else False for a in t.args):
                    t = expand_matmul(t)
                    
                args.append(t)
            return SuperMatAdd(*args).doit()
        else:
            return expr
            
def expand_matexpr(expr):
    """
        Expands matrix expressions (MatrixExpr)
    """
    from symgp.superexpressions import SuperMatAdd
        
    if expr.is_MatAdd:
        args = []
        args.extend([expand_matexpr(a) if a.is_MatMul else a for a in expr.args])
        return SuperMatAdd(*args).doit()
    elif expr.is_MatMul:
        return expand_matmul(expr).doit()
    else:
        return expr.doit()
                            
def collect(expr, syms, muls, evaluate=None):
    """
        Collect additive terms of a matrix expression
        Adapted from 'collect' function in SymPy library (https://github.com/sympy/sympy/blob/master/sympy/simplify/radsimp.py)
    
        Args:
            expr - The expression to collect terms for
            syms + muls - List of 1 or 2 symbols corresponding to order of multiplication indicators in 'muls'.
                          e.g. syms=[B,A],muls=['left','right'] corresponds to collecting terms for expressions
                               of the form B*{W1}*A + B*{W2}*A + {W3} where {W1}, {W2} and {W3} are matrix
                               expressions to give B*({W1} + {W2})*A + {W3}
    """
    
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
    if not isinstance(expr, MatAdd):
        return expr          
        
    if evaluate is None:
        evaluate = global_evaluate[0]

    def make_expression(terms):
        product = [term for term in terms]
        return SuperMatMul(*product)
    
    def parse_expression(terms, pattern, mul):
        """Parse terms searching for a pattern.
        terms is a list of MatrixExprs
        pattern is an expression treated as a product of factors
        
        Returns tuple of unmatched and matched terms.
        """ 
        if (not isinstance(pattern, MatrixSymbol) and 
            not isinstance(pattern, Transpose) and 
            not isinstance(pattern, Inverse) and
            not isinstance(pattern, MatAdd)):
            pattern = pattern.args
        else:
            pattern = (pattern,)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            if not isinstance(pattern, MatAdd):      
                pattern = [elem for elem in pattern]

            terms = terms[:]  # need a copy
                
            elems = []

            for elem in pattern:    
                if elem.is_Number:
                    # a constant is a match for everything
                    continue

                for j in range(len(terms)):
                    
                    # Search from right if we have a duplicate of 'pattern' in 'terms'. We only want to match one
                    # based on whether we collect terms on the right or left hand side given by 'mul'.
                    if mul == 'right':
                        k = len(terms)-1 - j
                    else:
                        k = j
                        
                    if terms[k] is None:
                        continue

                    term = terms[k]
                    
                    if (((not (isinstance(term, elem.__class__) and (isinstance(elem, MatrixSymbol) or
                               isinstance(elem, Transpose) or isinstance(elem, Inverse)))) 
                          and term.match(elem) is not None) or
                        (term == elem)):
                        # found common term so remove it from the expression
                        # and try to match next element in the pattern
                        elems.append(terms[k])
                        terms[k] = None

                        break
                else:
                    # pattern element not found
                    return None

            return [_f for _f in terms if _f], elems

    # Check that syms is of length 1 or 2
    if iterable(syms):
        syms = [s for s in syms]
        if len(syms) > 2:
            raise Exception("Too many matching symbols. Maximum is 2")
    else:
        syms = [syms]
    
    # Check that muls is either a list of same length as syms or a string for which
    # syms only has one element   
    if iterable(muls):
        muls = [m for m in muls]
        mul = muls[0]
        if len(muls) != len(syms):
            raise Exception("Number of muls should match syms.")
    else:
        mul = muls
        if not isinstance(mul,str) and len(syms) > 1:
            raise Exception("Number of muls should match syms.")
    
    expr = sympify(expr)
    
    # Get all expressions in summation
    
    # If syms[0] is a MatAdd, collect terms in summa that are equal to to the symbol
    if isinstance(syms[0], MatAdd) and isinstance(expr, MatAdd):
        matched, rejected = ZeroMatrix(expr.shape[0],expr.shape[1]), expr
            
        for s in syms[0].args:
            for t in rejected.args:
                if s == t:
                    matched += t
                    rejected -= t
                    break
            
        summa = [matched]
            
        if matched != expr:
            if isinstance(rejected,MatAdd):
                summa += [i for i in rejected.args]
            else:
                summa += [rejected]
    else:
        summa = [i for i in expr.args]
        
    collected, disliked = defaultdict(list), ZeroMatrix(expr.shape[0],expr.shape[1])
    
    # For each product in the summation, match the first symbol and update collected/
    # disliked depending on whether a match was/wasn't made.
    for product in summa:
        if isinstance(product, MatMul):
            terms = [i for i in product.args]
        else:
            terms = [product]

        # Only look at first symbol
        symbol = syms[0]
        
        result = parse_expression(terms, symbol, mul)
        
        # If symbol matched a pattern in terms, we collect the multiplicative terms for the 
        # symbol into a dictionary 'collected'
        if result is not None:
            terms, elems = result
                
            index = Identity(elems[0].shape[0])
            for elem in elems:
                index *= elem
    
            terms = make_expression(terms)
            if isinstance(terms, Number):
                if mul == 'left':
                    terms = SuperMatMul(Identity(index.shape[1]),terms)
                else:
                    terms = SuperMatMul(Identity(index.shape[0]),terms)
            
            collected[index].append(terms)
        else:
            # none of the patterns matched
            disliked += product
            
    # add terms now for each key
    collected = {k: SuperMatAdd(*v) for k, v in collected.items()}
    if isinstance(syms,list) and isinstance(muls,list):
        second_mul = muls[1]
        first_sym, second_sym = syms 
        collected[first_sym] = collect(collected[first_sym],[second_sym],second_mul)
    
    
    if not disliked.is_ZeroMatrix:
        if mul == 'left':
            collected[Identity(disliked.shape[0])] = disliked
        else:
            collected[Identity(disliked.shape[1])] = disliked

    if evaluate:
        if mul == 'left':
            if len(collected.items()) == 1:
                return [key*val for key, val in collected.items()][0]
            else:
                if expr.is_MatMul:
                    return SuperMatMul(*[key*val for key, val in collected.items()])
                else:
                    return SuperMatAdd(*[key*val for key, val in collected.items()])
        else: # mul == 'right'
            if len(collected.items()) == 1:
                return [val*key for key, val in collected.items()][0]
            else:
                if expr.is_MatMul:
                    return SuperMatMul(*[val*key for key, val in collected.items()])
                else:
                    return SuperMatAdd(*[val*key for key, val in collected.items()])
    else:
        return collected
 
def accept_inv_lemma(e, start, end):
    """
        Checks if expr satisfies the matrix form E^{-1}F(H - GE^{-1}F)^{-1}.
        
        We return True if e matches otherwise return False.
    """
        
    def checkSym(a):
        return isinstance(a, MatrixSymbol) or isinstance(a, Inverse) or isinstance(a, Transpose)
        
    def checkMatExpr(a, class_name):
        return isinstance(a, class_name)
    
    if len(e.args) < 3:
        return False
            
    arg_1, arg_2, arg_3 = e.args[start:end+1]
        
    # Match E^{-1}
    if not checkSym(arg_1):
        return False
            
    # Match E^{-1}F
    if not checkSym(arg_2):
        return False
        
    # Match E^{-1}F({MatExpr})^{-1}    
    if not checkMatExpr(arg_3, Inverse):
        return False
        
    # Match E^{-1}F({MatAdd})^{-1}    
    if not checkMatExpr(arg_3.arg, MatAdd):
        return False
        
    # Match E^{-1}F(A + B)^{-1}    
    if len(arg_3.arg.args) == 2:
        # Check whether it is E^{-1}F(A + MatMul)^{-1} or E^{-1}F(MatMul + B)^{-1}
        if checkSym(arg_3.arg.args[0]) and checkMatExpr(arg_3.arg.args[1], MatMul):
            arg_3_args = arg_3.arg.args[1].args
        elif checkSym(arg_3.arg.args[1]) and checkMatExpr(arg_3.arg.args[0], MatMul):
            arg_3_args = arg_3.arg.args[0].args
        else:
            return False
    else:
        return False
        
    # Match E^{-1}F(A + GCD)^{-1} or E^{-1}F(A + (-1)*GCD)^{-1}
    if len(arg_3_args) == 3 and not isinstance(arg_3_args[0], type(S.NegativeOne)):
        # Check whether CD matches E^{-1}F 
        if not (arg_3_args[1] == arg_1 and arg_3_args[2] == arg_2):
            return False
    elif len(arg_3.arg.args[1].args) == 4 and isinstance(arg_3.arg.args[1].args[0], type(S.NegativeOne)):
        # Check whether CD matches E^{-1}F 
        if not (arg_3_args[2] == arg_1 and arg_3_args[3] == arg_2):
            return False
    else:
        return False
        
    # Successful match
    return True

def check_inv_lemma(expr):
    
    if len(expr.args) == 3 and accept_inv_lemma(expr,0,2):
        return True
    else:
        return False
                                    
def simplify(expr):
    """
        A simplification algorithm
    
        We return a tuple of (simps, subs) (See below)
    """ 
    
    from symgp.superexpressions import SuperMatSymbol
    
    depth = get_max_depth(expand_to_fullexpr(expr))
    
    simps = []    # The simplified expressions we have obtained with the associated substitutions
    subs = {}     # Pairs substituted expressions with the substitutions made
    usedSubs = []   # The expressions we have substituted we have used so far
    
    # Get the expressions at every depth
    #exprs_by_depth = get_exprs_at_depth(expr, range(depth+1))
    
    usedNames = SuperMatSymbol.getUsedNames()
    
    min_expr = expr    
    for d in range(depth, -1, -1):  
        # Get the exprs at each depth for the new shortest expressions
        exprs_by_depth = get_exprs_at_depth(min_expr, range(depth+1))
           
        sub_exprs = exprs_by_depth[d]
        
        min_syms = math.inf
        
        # For each sub expression at level d check for copies in other parts of expressions
        for s in sub_exprs:
            repetitions = 0
            
            # Find other similar expressions to s
            for k in exprs_by_depth.keys():
                if k == d:
                    continue
                
                if s in exprs_by_depth[k]:    
                    repetitions += exprs_by_depth[k].count(s)  
            
            # Make replacements if expression 's' appears more than twice throughout expression or
            # it corresponds to the special matrix inverse lemma
            if (repetitions > 0 or check_inv_lemma(s)) and s not in usedSubs:
                
                # Update the used substituted expressions
                usedSubs.append(s)
                 
                # TODO: Allow for using best or range of simplified exprs from previous depths
                                    
                # Lower case for vectors and upper case for matrices
                if s.shape[0] != 1 and s.shape[1] != 1:
                    avail_prefixes = string.ascii_uppercase
                else:
                    avail_prefixes = string.ascii_lowercase
                
                # Keep on searching for available replacement names  
                for c in avail_prefixes:
                    i = 0
                    r_name = c + '_{' + str(i) + '}'
                    while r_name in usedNames and i < 99:
                        i += 1
                        r_name = c + '_{' + str(i) + '}'
                                 
                    if not r_name in usedNames:
                        r = SuperMatSymbol(s.shape[0], s.shape[1], r_name, expanded=s)
                        
                        repl_list = [(s,r)]         
                        simp_expr = replace(min_expr, repl_list).doit()
                    
                        if not subs.get(s):
                            subs[s] = r
                            
                        simps.append(simp_expr.doit())
                        
                        num_syms = get_num_symbols(simp_expr)
                        if num_syms < min_syms:
                            min_syms = num_syms
                            min_expr = simp_expr.doit()

                        # Check if we can collect any symbols on simp_expr. If we can add to simps.
                        if isinstance(simp_expr, MatAdd):
                            ends_of_expr_collection = get_ends(simp_expr)
                                
                            for ends_of_expr in ends_of_expr_collection:
                                ends_dict_left = defaultdict(list)
                                ends_dict_right = defaultdict(list)
                                ends_dict_both = defaultdict(list)
                            
                                # Collect left ends and right ends
                                for l in range(len(ends_of_expr)):
                                    if len(ends_of_expr[l]) == 2:
                                        ends_dict_left[ends_of_expr[l][0]].append(l)
                                        ends_dict_right[ends_of_expr[l][1]].append(l)
                                        ends_dict_both[ends_of_expr[l]].append(l)
                                    else:
                                        ends_dict_left[ends_of_expr[l][0]].append(l)
                                        ends_dict_right[ends_of_expr[l][0]].append(l)
                                
                                # If there are two or more repetitions of a symbol, collect        
                                for key, val in ends_dict_left.items():
                                    simped = collect(simp_expr,key,'left').doit()
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)
                            
                                for key, val in ends_dict_right.items():
                                    simped = collect(simp_expr,key,'right').doit()
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)
                            
                                # For cases where both ends are repeated two or more times (e.g. A*P*A + A*Q*A + B), collect
                                for key, val in ends_dict_both.items():
                                    simped = collect(simp_expr,[key[0],key[1]],['left','right']).doit()
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)      
                        break

    simps = sorted(simps, key=lambda e: get_num_symbols(e))
    
    return simps, subs            
                                              

######## Quick creation of variables/constants ########
def variables(var_names, var_shapes):
    """
        Creates a tuple of SuperMatSymbol Variables with the given names
    
        Args:
            var_names - The names of each variable. 
                        Can be a string, list or tuple. 
                        For a string, the variable names are separated by spaces e.g. "u f fs" for variables with 
                        names "u", "f" and "fs".
    
            var_shapes - The shapes of each variable. 
                         Can be a list or tuple of tuples. e.g. [(m,n), (p,q), (i,j)] for shapes (m,n), (p,q) and (i,j)
                         If the variable is a column vector, we simply need to specify one dimension e.g. [m, p, i] for shapes
                         (m,1), (p,1) and (i,1).
                         We can also have combinations e.g [m, (p,q), i] 

        Returns:
            output_vars - A tuple of variables for each 
    """
    
    from symgp.superexpressions import Variable
    
    if isinstance(var_names, str):
        var_names = var_names.split(" ")
    
    # Lists must be of same length
    assert(len(var_names) == len(var_shapes))
    
    for i, shape in enumerate(var_shapes):
        if isinstance(shape, Symbol):
            var_shapes[i] = (shape,1)
    
    return (Variable(name, shape[0], shape[1]) for name, shape in zip(var_names, var_shapes))
        
def constants(const_names, const_shapes):

    from symgp.superexpressions import Constant
    
    if isinstance(const_names, str):
        const_names = const_names.split(" ")
    
    # Lists must be of same length
    assert(len(const_names) == len(const_shapes))
    
    for i, shape in enumerate(const_shapes):
        if isinstance(shape, Symbol):
            const_shapes[i] = (shape,1)
    
    return (Constant(name, shape[0], shape[1]) for name, shape in zip(const_names, const_shapes))


######## Useful functions to get info about expressions ########
def get_exprs_at_depth(expr, depths):
    """
        Returns the MatAdd and MatMul expressions in expr at levels given by 'depth' of the expression tree.
    
        The top expression is level 0.
    
        If no expressions at the levels exist, we simply return an empty dict
    """
        
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
        
    if isinstance(depths, int):
        depths = [depths]
    else:
        depths = list(depths)
        
    exprs_at_depths = defaultdict(list)                        
    stack = [{expr: 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
            
        if level in depths and (isinstance(sub_expr, MatAdd) or isinstance(sub_expr, MatMul)):
            if isinstance(sub_expr, MatAdd) and len(sub_expr.args) > 2:    # Substitute all permutations of > 2 arg MatAdds
                sub_expr_perms = get_permutations(sub_expr)
                exprs_at_depths[level].extend(sub_expr_perms)
            elif isinstance(sub_expr, MatMul):    # Substitute 
                # Remove number at head of expression
                if isinstance(sub_expr.args[0], Number):
                    sub_expr = type(sub_expr)(*sub_expr.args[1:])
                
                if len(sub_expr.args) > 2:
                    l = len(sub_expr.args)
                    start, end = 0, 2
                
                    while end < l:
                        if (accept_inv_lemma(sub_expr,start,end)):
                            new_expr = type(sub_expr)(*sub_expr.args[start:end+1])
                            exprs_at_depths[level].append(new_expr)
                            break
                        else:
                            start += 1
                            end += 1
                
                    if end == l:
                        exprs_at_depths[level].append(sub_expr)
                else:
                    exprs_at_depths[level].append(sub_expr)
            else:
                exprs_at_depths[level].append(sub_expr)
                
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                #print(type(arg),arg)
                stack.append({arg: level+1})
                
    return exprs_at_depths

def get_ends(expr):
    """
        Returns the left and right matrices of the args of the MatAdd expression, expr.
    
        For example for A*Q*B + 2*C + D*E, we return [(A,B), (C,), (D,E)]
                 or for (Q+A)*R*(Q+A) + Q + A we return [(Q+A,Q+A), (Q+A,)]
    """
    
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
    # The collections of 'ends' lists where each has different groupings of single symbols. 
    # For example, for an expression (A+B)*Q*(B+C) + A + B + C, the two 'ends' lists we get are:
    #
    #               ends_collection = [[(A+B,B+C), (A+B), (C,)],
    #                                  [(A+B,B+C), (A,), (B+C,)]]
    #
    ends_collection = [] 
    
    # The preliminary list of end arguments of args of expr
    ends = []
    
    expr_args = list(expr.doit().args)
    
    mmul_to_rem = {}   # Pairs a MatMul to the remainder keyed by the ends. We ignore expressions of form {Number}*A where
                       # A is a MatSym, MatTrans or MatInv
                       
    for a in expr_args:
        a_mmul = a.as_coeff_mmul()[1].doit()
        if isinstance(a, MatMul):
            ends.append((a_mmul.args[0],a_mmul.args[-1]))
            mmul_to_rem[(a_mmul.args[0],a_mmul.args[-1])] = (a,(expr - a).doit())
        else:
            ends.append((a_mmul,))
    
    ends_collection.append(ends)
    
    for ends_mmul, val in mmul_to_rem.items():            
        for end in ends_mmul:
            if isinstance(end,MatAdd):
                rem = val[1]
                match = [elem for elem in get_permutations(val[1]) if elem==end]
                    
                if len(match) > 1:
                    raise Exception("More than one match found: %s"%(match))
                
                if len(match) > 0:
                    new_ends = [ends_mmul]
                    new_ends.append((match[0],))
                    for arg in match[0].args:
                        rem = (rem - arg).doit()
                        
                    # Get remaining elements
                    if isinstance(rem, MatMul):
                        for arg in rem.args:
                            if isinstance(arg, MatMul):
                                new_ends.append((arg.args[0],arg.args[-1]))
                            else:
                                new_ends.append((arg,))
                    else:
                        new_ends.append((rem,))
                          
                    if not new_ends in ends_collection:
                        ends_collection.append(new_ends)
                            
    return ends_collection 

def get_num_symbols(expr):
    """
        Returns the number of MatrixSyms in the expression
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
    
    numSyms = 0                          
    stack = [{expr: 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        
        if isinstance(sub_expr, MatrixSymbol):
            numSyms += 1
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                stack.append({arg: level+1})
    
    return numSyms

def display_expr_tree(expr):
    """
        Visualizes the expression tree for the given expression
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
                              
    stack = [{expand_to_fullexpr(expr): 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        
        print("-" + 4*level*"-",sub_expr)
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                stack.append({arg: level+1})

def get_max_depth(expr):
    """
        Get the maximum depth of the expression tree down to the lowest symbol 
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
    
    depth = 0                          
    stack = [{expr: 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                stack.append({arg: level+1})
            
            depth = level + 1 if level+1 > depth else depth
    
    return depth

def get_all_kernel_variables(expr):
    """
        Returns all the variables that are arguments of KernelMatrix objects that are
        in expr
    
        For example, for expr = K(a,u)*K(u,u)*K(u,b), we return kern_vars = [a,u,b]
    """
    
    from symgp import SuperDiagMat, SuperBlockDiagMat, KernelMatrix
    
    kern_vars = []                         
    stack = [(expr,0)]
    
    while len(stack) > 0:
        sub_expr, level = stack.pop()
        
        if isinstance(sub_expr,KernelMatrix):
            if sub_expr.inputs[0] not in kern_vars:
                kern_vars.append(sub_expr.inputs[0])
            if sub_expr.inputs[1] not in kern_vars:
                kern_vars.append(sub_expr.inputs[1])
            
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                stack.append((arg,level+1))
               
    
    return kern_vars
    
def get_permutations(expr):
    """
        Returns the permutations of a MatAdd expression for lengths 2 to len(expr).
    
        For example, for A + B + C + D, we return:
    
             [A+B, A+C, A+D, B+C, B+D, C+D, A+B+C, A+B+D, A+C+D, B+C+D, A+B+C+D]
    
    """
    
    from symgp.superexpressions import SuperMatAdd
    import itertools
    
    if isinstance(expr, MatrixSymbol) or isinstance(expr, Transpose) or isinstance(expr, Inverse):
        return [expr]
    
    if not isinstance(expr, MatAdd):
        raise Exception("Function only works for MatAdd expressions")
        
    expr_args = expr.args
    
    expr_perms = []
    
    for i in range(2,len(expr_args)+1):
        expr_perms.extend([SuperMatAdd(*e).doit() for e in itertools.combinations(expr_args,i)])
    
    return expr_perms

def get_var_coeffs(expr, var):
    """ 
        Returns the coeffs for the given variable and the remainder
        
        Args:
            - 'expr' - The expanded matrix expression
            - 'var' - List of variables for which we find the coefficients
    
        Returns:
            - 'coeffs' - A list of coeffs of the variables. Same size as 'var'
            - 'rem' - The remaining expression (when we subtract the terms corresponding to variables in 'var')
    """
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
    coeffs = [ZeroMatrix(expr.shape[0],v.shape[0]) for v in var]
     
    # Search the expression tree for each variable in var then add coefficient to list
    if expr.is_MatAdd:
        for arg in expr.args:   
            if arg in var:
                for i, v in enumerate(var):
                    if arg == v:
                        coeffs[i] = arg.as_coeff_mmul()[0]
            else:
                for arg2 in arg.args:
                    if arg2 in var:
                        for i, v in enumerate(var):
                            if arg2 == v:
                                coeffs[i] = SuperMatMul(*[c for c in arg.args if c != arg2]).doit()
        rem = SuperMatAdd(*[c for c in expr.args if c not in [c*v for c,v in zip(coeffs,var)]]).doit()
    elif expr.is_MatMul:
        rem = expr
        for arg in expr.args:
            if arg in var:
                for i, v in enumerate(var):
                    if arg == v:
                        coeffs[i] = SuperMatMul(*[c for c in expr.args if c != v]).doit()
                        rem = ZeroMatrix(expr.shape[0], expr.shape[1])
        
    else:
        rem = expr # If no match is made, we leave remainder as expr
        for i, v in enumerate(var):
            if expr == v:
                coeffs[i] = Identity(expr.shape[0])
                rem = ZeroMatrix(expr.shape[0],expr.shape[1])
    
    return coeffs, rem

def create_blockform(A,B,C,D):
    """
        Create new matrix by top bottom method i.e. create top half of matrix then create bottom
    
        Args:
            A, B, C, D - The four partitions of the block matrix. Must be 2-D i.e. all of form [[.]]
    
        Returns:
            The full blockform i.e. [[A, B], [C, D]]
    """ 
    top = []
    for row1, row2 in zip(A,B):
        top.append(row1+row2)

    bottom = []
    for row1, row2 in zip(C,D):
        bottom.append(row1+row2)
            
    return top+bottom

def get_variables(expr):
    """
        Returns a list of all the 'Variable' objects in the given expr.
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat, Variable
    
    variables_in_expr = []                      
    stack = [(expr, 0)]
    
    while len(stack) > 0:
        sub_expr, level = stack.pop()
        
        if isinstance(sub_expr, Variable) and sub_expr not in variables_in_expr:
            variables_in_expr.append(sub_expr)
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                stack.append((arg, level+1))
    
    return variables_in_expr
    
######### Other miscellaneous functions #########
def create_distr_name(dep_vars=None, cond_vars=None) -> str:
    """
    Creates a name based on the given variables of a distribution and the variables it is
    conditioned on
    :param dep_vars: The random variables of a distribution. x in p(x|z)
    :param cond_vars: The conditioned-on variables of a distribution. z in p(x|z)
    :return: A string of the name.
    """
    name = ''

    if dep_vars:
        if not isinstance(dep_vars[0], list):
            name += ','.join([v.name for v in dep_vars])
        else:
            dep_vars_x = dep_vars[0]
            dep_vars_y = dep_vars[1]
            name += ','.join([v.name for v in dep_vars_x + dep_vars_y])

    if cond_vars:
        name += '|' + ','.join([v.name for v in cond_vars])

    return name


######## GUI lexer ########
class Token(object, metaclass=ABCMeta):
    """
    Abstract base class for parser tokens
    """
    def __init__(self, *args) -> None:
        self.value = args[0]

    def __eq__(self, other: 'Token'):
        return self.value == other.value

    def __str__(self):
        return self.value

    def __repr__(self):
        return type(self).__name__ + '(value=' + self.value + ')'
        
class DiagToken(Token):

    ALLOWED_VALUES = ['diag', 'blockdiag', 'blkdiag']

    def __init__(self, t : str) -> None:
        """
        Initialises token for diagonal symbols
        :param t: The name of the token from the range - 'diag'|'blockdiag'|'blkdiag'
        """
        assert t in DiagToken.ALLOWED_VALUES, "t must be one of {}".format(
            DiagToken.ALLOWED_VALUES)
        super(DiagToken, self).__init__(t)

class OperatorToken(Token):

    ALLOWED_VALUES = ['+', '-', '*']

    def __init__(self, op : str) -> None:
        """
        Initialises token for operator symbols
        :param op: The name of the token from the range - '+'|'-'|'*'
        """
        assert op in OperatorToken.ALLOWED_VALUES, "op must be one of {}".format(
            OperatorToken.ALLOWED_VALUES)
        super(OperatorToken, self).__init__(op)

class PlusToken(OperatorToken):

    def __init__(self):
        super().__init__('+')

class MinusToken(OperatorToken):

    def __init__(self):
        super().__init__('-')

class StarToken(OperatorToken):

    def __init__(self):
        super().__init__('*')

class ParenToken(Token):

    ALLOWED_VALUES = ['(', ')', '[', ']', '{', '}']

    def __init__(self, paren : str) -> None:
        """
        Initialises token for parentheses symbols
        :param paren: The name of the token from the range - ')'|'('|']'|'['|'{'|'}'
        """
        assert paren in ParenToken.ALLOWED_VALUES, "paren must be one of {}".format(
            ParenToken.ALLOWED_VALUES)
        super(ParenToken, self).__init__(paren)

class LRoundParenToken(ParenToken):
    def __init__(self):
        super(LRoundParenToken, self).__init__('(')

class RRoundParenToken(ParenToken):
    def __init__(self):
        super(RRoundParenToken, self).__init__(')')

class LBoxParenToken(ParenToken):
    def __init__(self):
        super(LBoxParenToken, self).__init__('[')

class RBoxParenToken(ParenToken):
    def __init__(self):
        super(RBoxParenToken, self).__init__(']')

class LCurlyParenToken(ParenToken):
    def __init__(self):
        super(LCurlyParenToken, self).__init__('{')

class RCurlyParenToken(ParenToken):
    def __init__(self):
        super(RCurlyParenToken, self).__init__('}')
    
class MatIdentifierToken(Token):
    def __init__(self, mat : str) -> None:
        """
        Initialises token for matrix variable identifier symbols
        :param mat: The name of the token. Must start with a upper case letter and only have
        alphanumeric characters and/or '_'.
        """
        super(MatIdentifierToken, self).__init__(mat)

class VecIdentifierToken(Token):
    def __init__(self, vec : str) -> None:
        """
        Initialises token for vector variable identifier symbols
        :param vec: The name of the token. Must start with a lower case letter and only have
        alphanumeric characters and/or '_'.
        """
        super(VecIdentifierToken, self).__init__(vec)

class KernelToken(Token):
    def __init__(self, name : str, arg1 : str, arg2 : str) -> None:
        """
        Initialises token for kernel function symbols.
        :param name: The kernel name. Can start with lower or upper case letters
        :param arg1: The first argument of the kernel
        :param arg2: The second argumnet of the kernel
        """
        super(KernelToken, self).__init__(name)
        self.arg1 = arg1
        self.arg2 = arg2

    def __eq__(self, other: Token):
        return isinstance(other, KernelToken) and \
               (self.value == other.value and self.arg1 == other.arg1 and self.arg2 == other.arg2)

    def __str__(self):
        return self.value + '(' + self.arg1 + ',' + self.arg2 + ')'

    def __repr__(self):
        return type(self).__name__ + '(value=' + self.value + ', arg1=' + self.arg1 + ', arg2=' + \
               self.arg2 + ')'

class GroupToken(Token):
    """
    Groups the supplied tokens into a single token
    """

    def __init__(self, tokens: List[Token]):
        """
        Initialises the token that groups a sequence of tokens together.
        :param tokens: The list of tokens to group
        """
        super(GroupToken, self).__init__(tokens)

    def tokens(self) -> List[Token]:
        return self.value

class InvToken(Token):

    ALLOWED_VALUES = ['.I', '^-1', '^{-1}']

    def __init__(self, op: str) -> None:
        """
        Initialises token representing the inverse operation.
        :param op: Must be one of '.I', '^-1', '^{-1}'
        """
        assert op in InvToken.ALLOWED_VALUES, "op must be one of {}".format(
            InvToken.ALLOWED_VALUES)
        super(InvToken, self).__init__(op)

class TransToken(Token):

    ALLOWED_VALUES = ['.T', '\'', '^t', '^T', '^{t}', '^{T}']

    def __init__(self, op : str) -> None:
        """
        Initialises token representing the inverse operation.
        :param op: Must be one of ".T", "'", "^t", "^T", "^{t}", "^{T}".
        """
        assert op in TransToken.ALLOWED_VALUES, "op must be one of {}".format(
            TransToken.ALLOWED_VALUES)
        super(TransToken, self).__init__(op)

def get_tokens(expr: str) -> List[Token]:
    """
    Converts a string expression into a list of tokens. An exception is raised
    if the expression doesn't give a valid parse
    :param expr: The expression which we want to turn into a list of tokens.
    :return: The list of tokens
    """

    # Useful functions
    def match_to_symbol(s: str) -> Optional[
        Union[MatIdentifierToken, VecIdentifierToken, KernelToken]]:
        """
        Determines whether expr matches to mat_identifier, vec_identifier or kernel
        :param s: The expression which we want to match
        :return: A token if there is a match otherwise we return None
        """

        if mat_identifier.fullmatch(s):
            return MatIdentifierToken(s)
        elif vec_identifier.fullmatch(s):
            return VecIdentifierToken(s)
        elif kernel.fullmatch(s):
            # Break up 's' into the kernel name and the two arguments
            match = s.split("(")
            name = match[0]

            arg1, arg2 = match[1].strip(")").split(",")

            return KernelToken(name, arg1, arg2)
        else:
            return ValueError("Invalid string: {}. Should match regexes: {}, {} or {}".format(
                s, mat_identifier.pattern, vec_identifier.pattern, kernel.pattern))

    def match_to_mat_op(s: str) -> Optional[Union[TransToken, InvToken]]:
        """
        Determines whether s matches inv_sym or trans_sym
        :param s: String to be matched
        :return: A TransToken or InvToken depending on s
        """
        if inv_sym.fullmatch(s):
            return InvToken(s)
        elif trans_sym.fullmatch(s):
            return TransToken(s)
        else:
            raise ValueError("Invalid string: {}. Should match regexes: {} or {}".format(s, inv_sym.pattern, trans_sym.pattern))

    def make_paren_token(s: str) -> Optional[ParenToken]:
        if s == '(':
            return LRoundParenToken()
        elif s == ')':
            return RRoundParenToken()
        elif s == '[':
            return LBoxParenToken()
        elif s == ']':
            return RBoxParenToken()
        elif s == '{':
            return LCurlyParenToken()
        elif s == '}':
            return RCurlyParenToken()
        else:
            raise ValueError("Invalid paren token. Must be one of '(',')','[',']','{','}'. Provided: %s" % (s))

    def make_operator_token(s: str) -> Optional[OperatorToken]:
        if s == '+':
            return PlusToken()
        elif s == '-':
            return MinusToken()
        elif s == '*':
            return StarToken()
        else:
            raise ValueError('Invalid token. Must be one of "+", "-" or "*". Specified: {}'.format(s))

    # Remove meaningless spaces
    expr = expr.replace(" ", "")

    ## Regex expressions ##

    # Low-level expressions
    digit = re.compile(r"[0-9_]")
    lower_char = re.compile(r"[a-z]")
    upper_char = re.compile(r"[A-Z]")
    operators = re.compile(r"[\+\-\*]")
    diag_op = re.compile(r"diag|blkdiag|blockdiag")
    inv_sym = re.compile(r"\.I|\^\-1|\^\{\-1\}")
    trans_sym = re.compile(r"\.T|\'|\^t|\^T|\^\{t\}|\^\{T\}")

    # Matrix and vectors
    mat_identifier = re.compile(r"{1}(?:{0}|{1}|{2})*".format( \
        lower_char.pattern, upper_char.pattern, digit.pattern))
    vec_identifier = re.compile(r"{0}(?:{0}|{1}|{2})*".format( \
        lower_char.pattern, upper_char.pattern, digit.pattern))

    # Kernels
    kernel = re.compile(r"(?:{0}|{1})\((?:{2}|{3}),(?:{2}|{3})\)".format( \
        lower_char.pattern, upper_char.pattern, vec_identifier.pattern, mat_identifier.pattern))

    # Matrices, vectors and kernels
    symbols = re.compile(r"{0}|{1}|{2}".format(
        mat_identifier.pattern, vec_identifier.pattern, kernel.pattern)
    )

    # Full expression to match
    #expr_re = re.compile(
    #    r"^(\()?(?:({0})(\[))?(\()?({1})((?:{3}|{4})|\)|\])?((?:{3}|{4})|\)|\])?((?:{3}|{4})|\)|\])?((?:(?:{2})\(?(?:(?:{0})\[)?(?:{1})(?:(?:{3}|{4})|\)|\])?(?:(?:{3}|{4})|\)|\])?(?:(?:{3}|{4})|\)|\])?)*)(\))?". \
    #        format(diag_op.pattern, symbols.pattern, operators.pattern, inv_sym.pattern,
    #               trans_sym.pattern))

    expr_re = re.compile(
        r"^(\()?(?:({0})(\[))?(\()?({1})((?:(?:[\)\]])?(?:{3}|{4})?)*)((?:(?:{2})\(?(?:(?:{0})\[)?(?:{1})(?:(?:[\)\]])?(?:{3}|{4})?)*)*)(\))?".\
            format(diag_op.pattern, symbols.pattern, operators.pattern, inv_sym.pattern,
                   trans_sym.pattern))


    # First match first part of expression then recursively match remainder
    tokens = []
    expr_match = expr_re.fullmatch(expr)
    if expr_match:
        groups = expr_match.groups()
        #print("groups: ", groups)
        if groups[0]:  # '('
            tokens.append(LRoundParenToken())

        if groups[1]:  # diag_op
            tokens.append(DiagToken(groups[1]))

        if groups[2]:  # '['
            tokens.append(make_paren_token(groups[2]))

        if groups[3]:  # '('
            tokens.append(make_paren_token(groups[3]))

        if groups[4]:  # mat_identifier|vec_identifier|kernel
            tokens.append(match_to_symbol(groups[4]))

        # Alternations between (inv_sym|trans_sym) and ]|)
        #if groups[5]:  # ) | ]
        #    tokens.append(make_paren_token(groups[5]))

        #if groups[6]:  # inv_sym | trans_sym
        #    tokens.append(match_to_mat_op(groups[6]))

        close_expr = groups[5]
        close_expr_pat = re.compile(r"([\)\]])?({0}|{1})?((?:[\)\]]?(?:{0}|{1})?)*)".format(
            inv_sym.pattern, trans_sym.pattern))
        while len(close_expr) > 0:
            close_expr_groups = close_expr_pat.fullmatch(close_expr).groups()

            if close_expr_groups[0]:  # ) | ]
                tokens.append(make_paren_token(close_expr_groups[0]))

            if close_expr_groups[1]:  # inv_sym | trans_sym
                tokens.append(match_to_mat_op(close_expr_groups[1]))

            close_expr = close_expr_groups[2]

        # (inv_sym|trans_sym)|']'|')' (3 times)
        #for i in range(5,8):
        #    if groups[i]:
        #        try:
        #            token = make_paren_token(groups[i])
        #        except ValueError:
        #            token = match_to_mat_op(groups[i])

        #        tokens.append(token)

        ## Repeat for the rest of the expression
        right = groups[6]  # The remainder of the expression if it exists excluding last bracket

        #right_regex = re.compile(
        #    r"^({0})(\()?(?:({4})(\[))?({1})((?:{2}|{3})|\)|\])?((?:{2}|{3})|\)|\])?((?:{2}|{3})|\)|\])?((?:(?:{0})\(?(?:(?:{4})\[)?(?:{1})(?:(?:{2}|{3})|\)|\])?(?:(?:{2}|{3})|\)|\])?(?:(?:{2}|{3})|\)|\])?)*)".format(\
        #        operators.pattern, symbols.pattern, inv_sym.pattern, trans_sym.pattern,
        #        diag_op.pattern))

        right_regex = re.compile(
            r"^({0})(\()?(?:({4})(\[))?({1})((?:(?:[\)\]])?(?:{2}|{3})?)*)((?:(?:{0})\(?(?:(?:{4})\[)?(?:{1})(?:(?:[\)\]])?(?:{2}|{3})?)*)*)".format( \
                operators.pattern, symbols.pattern, inv_sym.pattern, trans_sym.pattern,
                diag_op.pattern))

        while len(right) > 0:
            subgroups = right_regex.fullmatch(right).groups()
            #print("subgroups: ", subgroups)
            if subgroups[0]:  # operators
                tokens.append(make_operator_token(subgroups[0]))
            else:
                raise RuntimeError("Scanning error: Missing operator")

            if subgroups[1]:  # '('
                tokens.append(make_paren_token(subgroups[1]))

            if subgroups[2]:  # 'diag_op'
                tokens.append(DiagToken(subgroups[2]))

            if subgroups[3]:  # '['
                tokens.append(make_paren_token(subgroups[3]))

            if subgroups[4]:  # mat_identifier|vec_identifier|kernel
                tokens.append(match_to_symbol(subgroups[4]))
            else:
                raise RuntimeError("Scanning error: Missing mat_identifier, vec_identifier or kernel.")

            # Alternations between (inv_sym|trans_sym) and ]|)
            # if groups[5]:  # ) | ]
            #    tokens.append(make_paren_token(groups[5]))

            # if groups[6]:  # inv_sym | trans_sym
            #    tokens.append(match_to_mat_op(groups[6]))

            close_expr = subgroups[5]
            close_expr_pat = re.compile(r"([\)\]])?({0}|{1})?((?:[\)\]]?(?:{0}|{1})?)*)".format(
                inv_sym.pattern, trans_sym.pattern))
            while len(close_expr) > 0:
                close_expr_groups = close_expr_pat.fullmatch(close_expr).groups()

                if close_expr_groups[0]:  # ) | ]
                    tokens.append(make_paren_token(close_expr_groups[0]))

                if close_expr_groups[1]:  # inv_sym | trans_sym
                    tokens.append(match_to_mat_op(close_expr_groups[1]))

                close_expr = close_expr_groups[2]

            # (inv_sym|trans_sym)|']'|')' (3 times)
            #for i in range(5, 8):
            #    if subgroups[i]:
            #        try:
            #            token = make_paren_token(subgroups[i])
            #        except ValueError:
            #            token = match_to_mat_op(subgroups[i])
            #
            #        tokens.append(token)

            right = subgroups[6]#[8]

        if groups[7]:
            tokens.append(RRoundParenToken())

        return tokens
    else:
        raise Exception("Invalid input")

def tokens_to_string(tokens : List[Token]) -> str:
    """
    Converts a list of tokens to the string they represent.
    :param tokens: The ordered list of tokens
    :return: The string representation of the list of tokens
    """
    output = ""
    for token in tokens:    
        if any([isinstance(token, token_class) for token_class in \
                [DiagToken, OperatorToken, ParenToken, MatIdentifierToken, VecIdentifierToken]]):
            output += token.value
        elif isinstance(token, InvToken) or isinstance(token, TransToken):
            sym = token.value
            if isinstance(sym, KernelToken):
                output += sym.value + "(" + sym.arg1 + "," + sym.arg2 + ")"
            elif isinstance(sym, GroupToken):
                output += tokens_to_string(sym.tokens())
            else:
                output += sym.value  
            
            if isinstance(token, InvToken):
                output += ".I"
            else:
                output += ".T"
        elif isinstance(token, GroupToken):
            output += tokens_to_string(token.tokens())
        else:
            output += token.value +"(" + token.arg1 + ","+ token.arg2 + ")"
    return output

######## GUI AST classes ########

## AST Printer stuff ##
class VisitorBase(object, metaclass=ABCMeta):
    """
    Abstract class for Visitor from the Visitor pattern.
    """
    def visit_binary(self, binary: 'Binary'):
        raise NotImplementedError()

    def visit_unary(self, unary: 'Unary'):
        raise NotImplementedError()

    def visit_literal(self, literal: 'Literal'):
        raise NotImplementedError()

    def visit_kernel_literal(self, kern_lit: 'KernelLiteral'):
        raise NotImplementedError()

    def visit_grouping(self, grouping: 'Grouping'):
        raise NotImplementedError()

    def visit_diag(self, diag: 'Diag'):
        raise NotImplementedError()

    def visit_matop(self, matop: 'MatOp'):
        raise NotImplementedError()

class ASTPrinter(VisitorBase):

    def print_ast(self, expr: 'ASTNode'):
        return expr.accept(self)

    def visit_binary(self, binary: 'Binary'):
        return self.parenthesise(binary.operator, binary.left, binary.right)

    def visit_unary(self, unary: 'Unary'):
        return self.parenthesise(unary.operator, unary.right)

    def visit_literal(self, literal: 'Literal'):
        return self.parenthesise(literal.value)

    def visit_kernel_literal(self, kern_lit: 'KernelLiteral'):
        return self.parenthesise(kern_lit.name, kern_lit.arg1, kern_lit.arg2)

    def visit_grouping(self, grouping: 'Grouping'):
        return self.parenthesise("group", grouping.expr)

    def visit_diag(self, diag: 'Diag'):
        return self.parenthesise(diag.diag_op, diag.expr)

    def visit_matop(self, matop: 'MatOp'):
        return self.parenthesise(matop.mat_op, matop.expr)

    def parenthesise(self, name: str, *exprs: Iterable['ASTNode']):
        out_str = "( " + name

        for expr in exprs:
            out_str += " "
            if isinstance(expr, ASTNode):
                out_str += expr.accept(self)
            else:
                out_str += expr
        out_str += ")"

        return out_str

## Node classes ##
class ASTNode(object, metaclass=ABCMeta):

    def __eq__(self, other):
        if type(self) == type(other):
            return all([self.__dict__[k] == other.__dict__[k] for k in self.__dict__.keys() if not k.startswith('_')])
        return False

    def accept(self, visitor: VisitorBase):
        raise NotImplementedError("Should be implemented by subclasses.")

class Binary(ASTNode):

    def __init__(self, left, operator: OperatorToken, right):
        self.left = left
        self.operator = operator.value
        self.right = right

    def __str__(self):
        return str(self.left) + self.operator + str(self.right)

    def __repr__(self):
        return "Binary(left={}, operator={}, right={}".format(self.left, self.operator, self.right) + ")"

    def accept(self, visitor: VisitorBase):
        return visitor.visit_binary(self)

class Unary(ASTNode):

    def __init__(self, operator: OperatorToken, right):
        self.operator = operator.value
        self.right = right

    def __str__(self):
        return self.operator + str(self.right)

    def __repr__(self):
        return "Unary(operator={}, right={})".format(self.operator, self.right)

    def accept(self, visitor: VisitorBase):
        return visitor.visit_unary(self)

class Literal(ASTNode):

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return "Literal(value={})".format(repr(self.value))

    def accept(self, visitor: VisitorBase):
        return visitor.visit_literal(self)

class KernelLiteral(ASTNode):

    def __init__(self, name: str, arg1: str, arg2: str):
        self.name = name
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return self.name + "(" + self.arg1 + ", " + self.arg2 + ")"

    def __repr__(self):
        return "KernelLiteral(name={}, arg1={}, arg2={})".format(self.name, self.arg1, self.arg2)

    def accept(self, visitor: VisitorBase):
        return visitor.visit_kernel_literal(self)

class Grouping(ASTNode):

    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "(" + str(self.expr) + ")"

    def __repr__(self):
        return "Grouping(expr={})".format(self.expr)

    def accept(self, visitor: VisitorBase):
        return visitor.visit_grouping(self)

class Diag(ASTNode):

    def __init__(self, diag_op: DiagToken, expr):
        self.diag_op = diag_op.value
        self.expr = expr

    def __repr__(self):
        return "Diag(diag_op={}, expr={})".format(self.diag_op, self.expr)

    def __str__(self):
        return self.diag_op + "[" + str(self.expr) + "]"

    def accept(self, visitor: VisitorBase):
        return visitor.visit_diag(self)

class MatOp(ASTNode):

    def __init__(self, expr, mat_op: Union[InvToken, TransToken]):
        self.expr = expr
        self.mat_op = mat_op.value

    def __repr__(self):
        return "MatOp(expr={}, mat_op={})".format(self.expr, self.mat_op)

    def __str__(self):
        return str(self.expr) + self.mat_op

    def accept(self, visitor: VisitorBase):
        return visitor.visit_matop(self)


# Parsing functions
def parse(tokens):
    """
    Parses a list of tokens to produce an expression with a dictionary
    of the objects created
    """

    current = 0   # Index of current token.

    def previous() -> Token:
        return tokens[current-1]

    def advance() -> Token:
        nonlocal current
        if not is_at_end():
            current += 1
        return previous()

    def peek() -> Token:
        return tokens[current]

    def is_at_end() -> bool:
        return current == len(tokens)

    def match(*token_types) -> bool:
        for token_type in token_types:
            if check(token_type):
                advance()
                return True
        return False

    def check(token_type) -> bool:
        if is_at_end():
            return False
        return isinstance(peek(), token_type)

    def consume(token_type, message: str) -> Token:
        if check(token_type):
            return advance()
        raise error(peek(), message)

    def error(token: Token, message: str):
        return RuntimeError(message + " Actual: " + token.value)

    def primary():
        if match(MatIdentifierToken, VecIdentifierToken):
            return Literal(previous().value)
        elif match(KernelToken):
            kern_tok = previous()  # type: KernelToken
            return KernelLiteral(kern_tok.value, kern_tok.arg1, kern_tok.arg2)
        elif match(LRoundParenToken):
            expr = expression()
            consume(RRoundParenToken, "Expect ')' after expression.")
            return Grouping(expr)
        elif match(DiagToken):
            diag_op = previous()  # type: DiagToken
            consume(LBoxParenToken, "Expect '[' after diag_op and before expression.")
            expr = expression()
            consume(RBoxParenToken, "Expect ']' after expression.")
            return Diag(diag_op, expr)

    def unary():
        if match(MinusToken):
            operator = previous()  # type: OperatorToken
            right = unary()
            out_expr = Unary(operator, right)
        else:
            out_expr = primary()

        if match(TransToken, InvToken):
            matop = previous()  # type: Union[TransToken, InvToken]
            return MatOp(out_expr, matop)
        else:
            return out_expr

    def multiplication():
        expr = unary()

        while match(StarToken):
            operator = previous()  # type: OperatorToken
            right = unary()
            expr = Binary(expr, operator, right)

        return expr

    def addition():
        expr = multiplication()
        while match(PlusToken, MinusToken):
            operator = previous()  # type: OperatorToken
            right = multiplication()
            expr = Binary(expr, operator, right)

        return expr

    def expression():
        return addition()

    return expression()

def print_ast(ast: ASTNode):
    ast_printer = ASTPrinter()
    return ast_printer.print_ast(ast)

# Interpreter
class Interpreter(VisitorBase, metaclass=ABCMeta):

    def __init__(self, namespace: Dict[str, Any]):
        """
        Initialises interpreter.
        :param namespace: Dictionary mapping names to Python objects that are used to evaluate
        expression. For example for a SymGP Constant named 'A', we would have the entry:

                            namespace['A'] = Constant('A')

        For Kernels, we have to append '_kern' to the Kernel name to distinguish it from matrix
        symbols.

        We assume in all the 'visit*' functions below that all the required objects have been defined
        previously.
        """
        self._ns = namespace

    def interpret(self, expr: ASTNode):
        return self.evaluate(expr)

    def evaluate(self, expr: ASTNode):
        return expr.accept(self)

    def visit_binary(self, expr: Binary):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)

        if expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        else:
            return None

    def visit_unary(self, expr: Unary):
        right = self.evaluate(expr.right)

        if expr.operator == '-':
            return -right
        else:
            return None

    def visit_literal(self, expr: Literal):
        return self._ns[expr.value]

    def visit_kernel_literal(self, expr: KernelLiteral):
        from symgp.kernels import Kernel

        arg1, arg2 = self._ns[expr.arg1], self._ns[expr.arg2]
        kern = self._ns[expr.name + '_kern']   # type: Kernel

        return kern(arg1, arg2)

    def visit_grouping(self, expr: Grouping):
        return self.evaluate(expr.expr)

    def visit_diag(self, expr: Diag):
        from symgp.superexpressions.supermatbase import SuperMatBase
        from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat

        arg = self.evaluate(expr.expr)  # type: SuperMatBase

        if expr.diag_op == 'diag':
            return SuperDiagMat(arg)
        elif expr.diag_op == 'blkdiag' or expr.diag_op == 'blockdiag':
            return SuperBlockDiagMat(arg)
        else:
            return None

    def visit_matop(self, expr: MatOp):
        from symgp.superexpressions import SuperMatTranspose, SuperMatInverse

        arg =  self.evaluate(expr.expr)

        trans_ops = [".T", "'", "^t", "^T", "^{t}", '^{T}']
        inv_ops = [".I", "^-1", "^{-1}"]

        if expr.mat_op in trans_ops:
            return SuperMatTranspose(arg)
        elif expr.mat_op in inv_ops:
            return SuperMatInverse(arg)
        else:
            return None
