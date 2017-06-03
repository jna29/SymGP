from __future__ import print_function, division

from collections import defaultdict
import copy
import re
import string
import math

from sympy import (MatMul, MatAdd, Basic, MatrixExpr, MatrixSymbol, ZeroMatrix, Symbol, Identity, Transpose,
                   Inverse, Number, Rational, ln, Determinant, pi, sympify, srepr, S, Expr)
from sympy.printing.latex import LatexPrinter
from sympy.core.evaluate import global_evaluate
from sympy.core.compatibility import iterable, ordered, default_sort_key


SMALL_MU_GREEK = '\u03bc'
BIG_SIGMA_GREEK = '\u03a3'
SMALL_SIGMA_GREEK = '\u03c3'
BIG_OMEGA_GREEK = '\u03a9'
BIG_LAMBDA_GREEK = '\u039b'
SMALL_ETA_GREEK = '\u03b7' 



######## Matrix operations with lists as matrices ########
def matmul(list1, list2, debug=False):
    """
        Multiply two lists in a matrix fashion.
    
        Similar to numpy's matrix multiplication of arrays:
            - If list1 has shape (m1,) (i.e. it is a 1-D list) it is broadcast to (1,m1).
              list2 can have shapes (m1,n2) or (m1,) otherwise a Exception is raised.
              A list of shape (n2,) or (1,) is returned.
            - If list2 has shape (m2,) it is broadcast to (m2,1).
              list1 can have shapes (m2,) or (m1,m2) otherwise a Exception is raised.
              A list of shape (1,) or (m1,) is returned.
            - Any other case requires the shapes to match.
        
        Multiplying all elements in a list by a number is also supported e.g. matmul(a,5) or matmul(5,a).     
    """
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
    if debug:
        print("list1: ",list1)
        print("list2: ",list2)
        
    # Handle multiplication by integers
    if isinstance(list1, int):
        if isinstance(list2[0], list):
            list2 = [[SuperMatMul(list1,list2[i][j]).doit() for j in range(len(list2[0]))] for i in range(len(list2))]
        else:
            list2 = [SuperMatMul(list1, v).doit() for v in list2]
        return list2
        
    if isinstance(list2, int):
        if isinstance(list1[0], list):
            list1 = [[SuperMatMul(list2, list1[i][j]).doit() for j in range(len(list1[0]))] for i in range(len(list1))]
        else:
            list1 = [SuperMatMul(list2, v).doit() for v in list1]
        return list1
    
    if debug:
        print("list1: ",list1)
        print("list2: ",list2)
        
    broadcast_list1 = False
    broadcast_list2 = False
    
    # Check sizes and reshape if necessary
    if isinstance(list1[0],list):
        m1 = len(list1)
        n1 = len(list1[0])
    else:
        m1 = 1
        n1 = len(list1)
        list1 = [e.T for e in list1]
        broadcast_list1 = True
    
    if isinstance(list2[0],list):
        m2 = len(list2)
        n2 = len(list2[0])
    else:
        m2 = len(list2)
        n2 = 1
        broadcast_list2 = True
    
    # Check shapes
    if n1 != m2:
        raise Exception("Shapes don't match: %s, %s" % ((m1, n1), (m2, n2)))
    
    # Multiply based on types of lists
    if broadcast_list1 and broadcast_list2: # (1,n1) x (m2,1)
        #out_list = [sum([list1[i]*list2[i] for i in range(n1)])]    
        out_list = [SuperMatAdd(*[SuperMatMul(list1[i],list2[i]).doit() for i in range(n1)]).doit()]
    elif broadcast_list1:  # (1,n1) x (m2,n2)
        out_list = [0 for _ in range(n2)]
        for i in range(n2):
            #out_list[i] = sum([list1[j]*list2[j][i] for j in range(m2)])
            out_list[i] = SuperMatAdd(*[SuperMatMul(list1[j],list2[j][i]).doit() for j in range(m2)]).doit()
    elif broadcast_list2:  # (m1,n1) x (m2,1)
        out_list = [0 for _ in range(m1)]
        for i in range(m1):
            #out_list[i] = sum([list1[i][j]*list2[j] for j in range(m2)])
            out_list[i] = SuperMatAdd(*[SuperMatMul(list1[i][j],list2[j]).doit() for j in range(m2)]).doit()
    else: # (m1,n1) x (m2,n2) 
        out_list = [[0 for _ in range(n2)] for _ in range(m1)]
        for i in range(m1):
            for j in range(n2):
                #out_list[i][j] = sum([list1[i][k]*list2[k][j] for k in range(n1)])
                out_list[i][j] = SuperMatAdd(*[SuperMatMul(list1[i][k],list2[k][j]).doit() for k in range(n1)]).doit()
    
    if debug:
        print("out_list: ",out_list)
           
    return out_list

def matadd(list1, list2, debug=False):
    """
        Adds two lists that must be the same shape. We reshape list of (m,) to (0,m).
    
        Returns a list of the same shape as the lists.
    """
    from symgp.superexpressions import SuperMatAdd
    
    if debug:
        print("list1: ", list1)
        print("list2: ", list2)
        
    # Check sizes
    if isinstance(list1[0],list):
        m1 = len(list1)
        n1 = len(list1[0])
    else:
        m1 = 0
        n1 = len(list1)
    
    if isinstance(list2[0],list):
        m2 = len(list2)
        n2 = len(list2[0])
    else:
        m2 = 0
        n2 = len(list2)

    if m1 != m2 or n1 != n2:
        raise Exception("Shapes don't match: %s, %s" % ((m1, n1), (m2, n2)))
    
    if m1 == 0:
        out_list = [SuperMatAdd(list1[i],list2[i]).doit() for i in range(n1)]
    else:
        out_list = [[SuperMatAdd(list1[i][j],list2[i][j]).doit() for j in range(n1)] for i in range(m1)]
    
    if debug:
        print("out_list: ", out_list)
        
    return out_list

def mattrans(mat, debug=False):
    """
        Returns the transpose of an mxn matrix (list of lists)
    
        Similar to numpy's transpose.
            - If mat's shape is (m,) we simply return mat
            - Otherwise we return the appropriate transpose for mat shape (m,n) 
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
    
def matinv(mat, debug=False):
    """
        Inverts nxn matrices. 
        
        If n > 2, we first partition then apply the algorithm again.
        If n == 1, we simply return the SuperMatInverse.
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
        
        #print("P: ",P, " Q: ",Q," R: ",R," S: ", S)
        P_bar = matinv(matadd(P,matmul(matmul(matmul(-1,Q),matinv(S)),R)))
        #print("Done P_bar")
        Q_bar = matmul(matmul(matmul(-1,P_bar),Q),matinv(S))
        #print("Done Q_bar")
        R_bar = matmul(matmul(matmul(-1,matinv(S)),R),P_bar)
        #print("Done R_bar")
        S_bar = matadd(matinv(S),matmul(matmul(matmul(matmul(matinv(S),R),P_bar),Q),matinv(S)))
        #print("Done S_bar")
        
        # Create new matrix by top bottom method i.e. create top of matrix then create bottom
        top = []
        for row1, row2 in zip(P_bar,Q_bar):
            top.append(row1+row2)
        
        bottom = []
        for row1, row2 in zip(R_bar,S_bar):
            bottom.append(row1+row2)
        
        return top+bottom

def partition_block(block, indices, debug=False):
    """
        Partitions a list into four or two sections based on the indices
    
        Args:
            block - The input block to be partitioned: 
                        - If block is 2-D, we partition it into [[P, Q], [R, S]]
                        - If block is 1-D, we partition it into [a, b] or [[a],[b]] if it is of shape (m,1)
            indices - The indices that form one partition:
                        - If block is 2-D, this can be 2-D or 1-D. If 1-D block needs to be square
                        - If block is 1-D, this should also be 1-D
                      Repeat indices are removed automatically
        
        Returns:
          Either
            P, Q, R, S - The four partitions for 2-D blocks
          Or
            a, b - The two partitions for 1-D blocks
    """
    
    if debug:
        print("debug: ", debug)
        print("indices: ", indices)
    
    new_block = []  #copy.deepcopy(block)
    if isinstance(block[0], list):
        for row in block:
            new_block.append(list(row))
    elif isinstance(block, list):
        new_block = list(block)
    else:
        raise Exception("The block to be partitioned must be a list")
    
    if debug:
        print("new_block: ", new_block)
    
    if isinstance(new_block[0],list) and len(new_block[0]) > 1:
        
        # Check for 1x1 case
        if len(new_block) == 1 and len(new_block[0]) == 1:
            raise Exception("Can't partition a 1x1 block. Minimum size is 2x2")
        
        # Check indices are 2-D. If not, convert to 2-D
        if not isinstance(indices[0],list):
            if all([not isinstance(i,list) for i in indices[1:]]):    # [a,b]
                indices = [indices, indices]
            else:                                  # [a,[b,c]]
                raise Exception("indices must be 2-D i.e. [[a,b],[c,d]] or [a,b]")
        else:
            if not isinstance(indices[1],list):    # [[a,b],c]
                raise Exception("indices must be 2-D i.e. [[a,b],[c,d]] or [a,b]")
        
        if debug:
            print("indices: ", indices)
            
        # Check that all indices are in appropriate range
        if any([(i < 0 or i > len(new_block)-1) for i in indices[0]]):
            raise Exception("Invalid indices. Must be in range: [%s,%s]" % (0,len(new_block)-1))
        
        if any([(i < 0 or i > len(new_block[0])-1) for i in indices[1]]):
            raise Exception("Invalid indices. Must be in range: [%s,%s]" % (0,len(new_block[0])-1))
            
        # Remove repeat set of indices
        indices[0] = list(set(indices[0]))
        indices[1] = list(set(indices[1]))
        
        if debug:
            print("indices: ", indices)
            
        # First push columns indicated by indices to end
        for idx, col in enumerate(sorted(indices[1])[::-1]):
            if col == len(new_block[0])-1:
                continue
            else:
                c = col
                while c < len(new_block[0])-(idx+1):
                    for row in range(len(new_block)):
                        temp = new_block[row][c]
                        new_block[row][c] = new_block[row][c+1]
                        new_block[row][c+1] = temp
                    c += 1    
        
        if debug:
            print("new_block: ", new_block)
            
        # Do same for rows
        for idx, row in enumerate(sorted(indices[0])[::-1]):
            if row == len(new_block)-1:
                continue
            else:
                r = row
                while r < len(new_block)-(idx+1):
                    for col in range(len(new_block[0])):
                        temp = new_block[r][col]
                        new_block[r][col] = new_block[r+1][col]
                        new_block[r+1][col] = temp
                    r += 1
    
        if debug:
            print("new_block: ", new_block)
            
        m = len(new_block) - len(indices[0])
        n = len(new_block[0]) - len(indices[1])
        
        if debug:
            print("m: %s, n: %s" % (m, n))
            
        if n < 1:
            P = new_block[:m]
            S = new_block[m:]
            
            if debug:
                print("P: ", P, "S: ", S)
                
            return P, S
        else:
            P = [new_block[i][:n] for i in range(m)]
            Q = [new_block[i][n:] for i in range(m)]
            R = [new_block[i][:n] for i in range(m, len(new_block))]
            S = [new_block[i][n:] for i in range(m, len(new_block))]
            
            if debug:
                print("P: ", P, "Q: ", Q, "R: ", R, "S: ", S)
                
            return P, Q, R, S
    else:
        # Check for 1x1 case
        if len(new_block) == 1:
            raise Exception("Can't partition a 1x1 block")
        
        # Check indices is correct shape
        if any([isinstance(i,list) for i in indices]):
            raise Exception("Incorrect form of indices. Must be 1-D")
        
        # Check that all indices are in appropriate range
        if any([(i < 0 or i > len(new_block)-1) for i in indices]):
            raise Exception("Invalid indices. Must be in range: [%s,%s]" % (0,len(new_block)-1))
            
        # Remove duplicates
        indices = list(set(indices))
        
        if debug:
            print("indices: ", indices)
        
        # Push elements corresponding to indices to end of list
        for idx, k in enumerate(sorted(indices)[::-1]):
            if k >= len(new_block)-1:
                continue
            else:
                i = k
                while i < len(new_block)-(idx+1):
                    temp = new_block[i]
                    new_block[i] = new_block[i+1]
                    new_block[i+1] = temp
                    i += 1
        
        if debug:
            print("new_block: ", new_block)
            
        m1 = len(new_block) - len(indices)
        a = new_block[:m1]
        b = new_block[m1:]
        
        if debug:
            print("m1: ", m1)
            print("a: ", a)
            print("b: ", b)
        
        return a, b


######## MVG helper functions ########
def get_Z(cov, debug=False):
    """
        Calculates normalising constant symbol using cov
    """
    return -cov.shape[0]/2*ln(2*pi) - Rational(1,2)*ln(Determinant(cov))


######### Search and replace functions ########
def replace_with_num(expr, d, debug=False):
    """
        Performs a DFS search through the  expression tree to replace MatrixSymbols with 
        numerical matrices.
    
        Args:
            - 'expr' - The current node in expression tree
            - 'd' - A dictionary mapping the matrix symbols to numerical matrices

        Returns:
            - Based on type of 'expr':
                - MatrixSymbol - Return corresponding numerical matrix
                - Number - Return Number
                - MatMul/MatAdd - Return (MatAdd/MatMul)({children})
        
    """
    #from symgp.superexpressions import SuperMatMul, SuperMatAdd, SuperMatInverse, SuperMatTranspose
    import numpy as np
    
    if isinstance(expr, MatrixSymbol) or isinstance(expr, Number):
        if isinstance(expr, MatrixSymbol):
            try:
                return d[expr.name]
            except KeyError as e:
                print("Error: No numerical matrix was specified for %s" % (e))
        else:
            return expr
        
    r = []
    if debug:
        print("expr: ",expr)
    for arg in expr.args:
        if debug:
            print("arg: ",arg)
        r.append(replace_with_num(arg, d, debug))
    
    # MatMul/MatAdd/Transpose/Inverse (MatrixSymbol is covered above)
    if isinstance(expr, MatrixExpr):  
        if debug:
            print("expr: ",expr)
            for i, ele in enumerate(r):
                print(i,": ",ele)
        if expr.is_MatMul:
            for e in r:
                if not isinstance(e,Number):
                    shape = e.shape[0]
                    break
                    
            out = np.eye(shape)
            for e in r:
                if isinstance(e,Number):
                    out *= np.float(e)
                elif not isinstance(e,np.ndarray):
                    out = np.dot(out,np.array(e.tolist(),dtype=np.float32))
                else:
                    out = np.dot(out,e)
            return out
            #return SuperMatMul(*r)
        elif expr.is_MatAdd:
            if len(r[0].shape) == 2:
                out = np.zeros((r[0].shape[0],r[0].shape[1]))
            else:
                out = np.zeros(r[0].shape[0])
            
            for e in r:
                if not isinstance(e,np.ndarray):
                    out += np.array(e.tolist(),dtype=np.float32).reshape(out.shape)
                else:
                    out += e.reshape(out.shape)
            return out
            #return SuperMatAdd(*r)
        elif expr.is_Inverse:
            out = np.linalg.inv(r[0])
            return out
            #return SuperMatInverse(*r)
        else: # expr.is_Transpose
            out = r[0].T 
            return out
            #return SuperMatTranspose(*r)
    else:
        raise Exception("Expression should be a MatrixExpr")

def evaluate_expr(expr, d, debug=False):
    """
        Evaluates a matrix expression with the given numerical matrices
    
        Args:
            - 'expr' - The symbolic matrix expression
            - 'd' - A dictionary mapping the matrix symbols to numerical matrices
    
        Returns:
            - 'r' - The result of all the matrix calculations
    """
    
    r = replace_with_num(expr, d, debug)
    
    return r
    
def replace_with_expanded(expr, done=True, debug=False):
    """
        Similar to 'replace_with_num' above except we replace SuperMatrixSymbols
        with their expanded forms if they exist
    
        Args:
            expr - The current MatrixExpr
        
        Returns:
            expr - The expanded MatrixExpr
            done - Boolean indicating whether no more expansions can be done
    """
    
    from symgp.superexpressions import (SuperMatSymbol, SuperMatTranspose, SuperMatInverse, SuperMatAdd, SuperMatMul, SuperDiagMat,
                                        SuperBlockDiagMat)
    
    #print("expr: ",expr)
    if (not isinstance(expr, MatMul) and not isinstance(expr, MatAdd) and 
        not isinstance(expr, Inverse) and not isinstance(expr, Transpose)):
        if isinstance(expr, SuperMatSymbol) and expr.expanded is not None:
            done = False
            #print("Returned %s" % (expr.expanded))
            return expr.expanded, done
        else:
            #print("Returned %s" % (expr))
            return expr, done
        
    r = []
    #print("expr: ",expr)
    for arg in expr.args:
        #print("expr: %s, arg: %s" % (expr, arg))
        expanded, done = replace_with_expanded(arg, done)
        #print("expr: %s, expanded: %s" % (expr, expanded))
        r.append(expanded)
       
    if isinstance(expr, MatrixExpr):
        if expr.is_MatMul:
            e = SuperMatMul(*r)
        elif expr.is_MatAdd:
            e = SuperMatAdd(*r)
        elif expr.is_Inverse:
            if isinstance(expr, SuperMatSymbol):
                e = SuperMatInverse(*r)
            else:
                e = SuperMatInverse(*r)
        elif expr.is_Transpose:
            if isinstance(expr, SuperMatSymbol):
                e = SuperMatTranspose(*r)
            else:
                e = SuperMatTranspose(*r)
        else:
            raise Exception("Unknown expression of type %s" % (type(expr)))
        
        #print("e: ",e)
        return e, done
    else:
        raise Exception("Expression should be a MatrixExpr")

def expand_to_fullexpr(expr, num_passes=-1, debug=False):
    """
        Expands a MatrixExpr composed of SuperMatSymbols by substituting any SuperMatSymbol
        with an 'expanded' or 'blockform'
    
        Args:
            expr - The expression to expand
            num_passes - The number of passes to make through the expression. -1 indicates that
                         we pass through expression until no more substitutions can be made.
    
        Return:
            expanded_expr - The expanded expression
    """
    
    # Keep on passing through expression until no more substitutions can be made
    if num_passes == -1:
        done = False
        e = expr
        while not done:
            done = True
            e, done = replace_with_expanded(e, done)
            #print("e (out): ",e)
        
        return e.doit().doit()
    else:
        e = expr
        for _ in range(num_passes):
            e, _ = replace_with_expanded(e)
        
        return e.doit().doit()
        
# TODO: Find a better substitution algorithm. Problem how to deal with symbols that aren't explicitly grouped
def replace(expr, rules, debug=False):
    """
        Replaces expressions in expr with the given rules.
    
        Args:
            expr - The expression for which we want to replace Matrix Expressions
            rules - A dictionary where we replace the key with the value.
                    N.B. For an expression of the form -1*A we must replace it with another expression
                         of the form -1*B and not A with B.
    
        Returns:
            The expression with the substituted for Matrix Symbols
    """
        
    from collections import deque
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat, SuperMatAdd, SuperMatSymbol
    
    # Get the full expression
    #full_expr = expand_to_fullexpr(expr)
    full_expr = expr
    
    # For each substitution rule, replace the corresponding sub-expression
    for k, v in rules.items():
        
        if debug:
            print("full_expr: ", full_expr)   
        
        m = len(k.args)   # Number of arguments. TODO: Check for cases where k is a single symbol
        
        if debug:
            print("m: ", m)
        
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
        tree_table = dict()
        
        queue = deque(((full_expr, 0, 0),))
        
        if debug:
            print("queue: ", queue)
        #tree_table[(full_expr,0)] = list(zip(list(full_expr.args),[1]*len(full_expr.args)))
        
        num_nodes = 0
        count = 1
        while len(queue) > 0:
            sub_expr, level, old_k = queue.pop()
            
            if debug:
                print("sub_expr: %s, level: %s, old_k: %s" % (sub_expr, level, old_k))
            
            if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
                isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
                isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            
                # Match current rule to expressions in this sub expression
                n = len(sub_expr.args)
                
                if debug:
                    print("n: ",n)
                
                i = 0
                while i < n:
                    j = 0
                    l = 0    # Used when we need to skip over symbols e.g. for addition where we may need to match over symbols.
                    while j < m and i + l + j < n:
                        if debug:
                            print("i: %s, j: %s, l: %s" % (i,j,l))
                            
                        if i + l + j >= n:
                            break
                        
                        if debug:
                            print("sub_expr.args[%s]: %s, k.args[%s]: %s" % (i+l+j,sub_expr.args[i+l+j], j, k.args[j]))
                            print("Match? ", sub_expr.args[i+l+j].doit() == k.args[j].doit())
                            
                        if (sub_expr.args[i+l+j].doit() != k.args[j].doit()):# or (sub_expr.args[i+l+j].match(k.args[j])):
                            #foundMatch = False
                            # As additions may be stored in any order, we need to skip symbols so that we can match
                            # the pattern
                            if isinstance(k, MatAdd) and isinstance(sub_expr, MatAdd):
                                l += 1
                            else:
                                break
                        else:
                            j += 1
                    
                          
                    if j == m:  # Match found: Replace match with pattern
                        if debug:
                            print("Match found!")
                            
                        if m == n:
                            new_level = level
                        else:
                            new_level = level + 1
            
                        queue.appendleft((v, new_level, count))
                        
                        if debug:
                            print("new_level: ", new_level)
                            if tree_table.get((sub_expr, level, old_k)):
                                print("(Before) tree_table[(sub_expr, level, old_k)]: ", tree_table[(sub_expr, level, old_k)])
                          
                        # We need to re-order sub_expr - mainly for matches in MatAdds with remainders e.g. matching A in A + B + C
                        if l > 0:
                            old_sub_expr = sub_expr
                            
                            rem = sub_expr
                            for c in k.args:
                                rem -= c
                            if debug:
                                print("rem: ", rem)
                            rem = [rem] if not isinstance(rem,MatAdd) else list(rem.args)
                            if debug:
                                print("rem: ", rem)
                            new_args = list(k.args) + rem
                            sub_expr = SuperMatAdd(*new_args)
                            if debug:
                                print("new sub_expr: ", sub_expr)
                            
                            # As we changed the sub_expr we have to reassign the elements of the old one
                            if tree_table.get((old_sub_expr, level, old_k)):
                                old_values = tree_table.pop((old_sub_expr, level, old_k), None)
                                tree_table[(sub_expr, level, old_k)] = old_values + [(v, new_level, count)]
                            else:
                                tree_table[(sub_expr, level, old_k)] = [(v, new_level, count)]
                            
                            count += 1
                                
                        else:    
                            # Check entry for sub_expr exists
                            if tree_table.get((sub_expr, level, old_k)):
                                tree_table[(sub_expr, level, old_k)].append((v, new_level, count))
                            else:
                                tree_table[(sub_expr, level, old_k)] = [(v, new_level, count)]
                            
                            count += 1
                        

                        if debug:    
                            print("(After) tree_table[(sub_expr, level, old_k)]: ", tree_table[(sub_expr, level, old_k)])    
                            
                        # Start after pattern     
                        i += m
                    else:
                        queue.appendleft((sub_expr.args[i], level+1, count))
                        
                        # Check entry for sub_expr exists
                        if tree_table.get((sub_expr, level, old_k)):
                            tree_table[(sub_expr, level, old_k)].append((sub_expr.args[i], level+1, count))
                        else:
                            tree_table[(sub_expr, level, old_k)] = [(sub_expr.args[i], level+1, count)]
                        
                        count += 1
                        
                        # Start at next symbol     
                        i += 1
                    
                    if debug:
                        print("queue: ",queue)
                        print("tree_table: ", tree_table) 
            else:
                # Add expression for this node
                tree_table[(sub_expr, level, old_k)] = sub_expr
                 
                if debug:
                    print("tree_table: ", tree_table)        
        
        # Create expression from table
        sorted_tree_table = sorted(tree_table.items(), key=lambda elem: elem[0][1], reverse=True)  # Sort based on level
        
        if debug:
            print("sorted_tree_table: ", sorted_tree_table)
        
        for p, c in sorted_tree_table:
            
            # Skip terminal nodes else update tree table for non-terminal nodes
            if p[0] == c:
                continue
            else:
                # Create MatrixExpr using the elements in the value c, which is a list, for the key p and
                # then update 'tree_table'
                tree_table[p] = type(p[0])(*[tree_table[e] for e in c])  
        
        # Rewrite full expression    
        full_expr = tree_table[sorted_tree_table[-1][0]]    
    
    if debug:   
        print("final_expr: ",full_expr)
                    
    return full_expr
            
def replace_with_SuperMat(expr, d, debug=False):
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
    #"expr: ",expr)
    for arg in expr.args:
        #print("arg: ",arg)
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
        return r'\mathbf{'+expr.name+'}'
        if (mat_type == 'mean' or mat_type == 'covar' or mat_type == 'invcovar' or
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
                return r'\mathbf{'+expr.name+'}'

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
                    mean_name += self._print(expand_to_fullexpr(expr.mean.blockform[i]).doit())+r'\\'
                mean_name += self._print(expand_to_fullexpr(expr.mean.blockform[-1]).doit())+r'\end{smallmatrix}\right]'
            
            # Covariance
            covar_short_name = r'\mathbf{\Sigma}_{'+vars_name_pre+r'}'
            
            if expr.covar.blockform is not None:
                covar_name = r'\left[\begin{smallmatrix}'
                for i in range(len(expr.covar.blockform)-1):
                    for j in range(len(expr.covar.blockform[i])-1):
                        covar_name += self._print(expand_to_fullexpr(expr.covar.blockform[i][j]).doit())+r'&'
                    covar_name += self._print(expand_to_fullexpr(expr.covar.blockform[i][-1]).doit())+r'\\'
            
                # Add last row
                for j in range(len(expr.covar.blockform[-1])-1):
                    covar_name += self._print(expand_to_fullexpr(expr.covar.blockform[-1][j]).doit())+r'&'
                covar_name += self._print(expand_to_fullexpr(expr.covar.blockform[-1][-1]).doit())+r'\end{smallmatrix}\right]'
            
            # Write shortened distribution expression
            latex_name += vars_name_N + r';' + mean_short_name + r',' + covar_short_name + r'\right)\\'+"\n"
            
            
        else:
            mean_short_name = r'\mathbf{m}_{'+vars_name_pre+r'}'
            mean_name = self._print(expand_to_fullexpr(expr.mean.expanded).doit()) if expr.mean.expanded is not None else ''
            covar_short_name = r'\mathbf{\Sigma}_{'+vars_name_pre+r'}'
            covar_name = self._print(expand_to_fullexpr(expr.covar.expanded).doit()) if expr.covar.expanded is not None else ''
            
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
        latex_name = r'\mathbf{'+expr.K_func.name+'}_{'+matLatex(expr.args[1])+','+matLatex(expr.args[2])+'}'
        return latex_name
             
def matLatex(expr, profile=None, debug=False, **kwargs):
    if profile is not None:
        profile.update(kwargs)
    else:
        profile = kwargs
    out_latex = matLatPrinter(profile).doprint(expr)
    
    #Clean up string
    out_latex = re.sub('(\+.\-1)','-',out_latex) # Change '+ -1' to '-'
    #out_latex = re.sub('','-{1}')
    
    return out_latex

def updateLatexDoc(filename, expr, debug=False):
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
def expand_mat_sums(sums, debug=False):
    """
        Helper method for 'expand_matmul' 
        Based on 'def _expandsums' in sympy.core.mul
    """
    from symgp.superexpressions.supermatadd import SuperMatAdd, SuperMatMul
    
    if debug:
        print("sums: ", sums)
             
    L = len(sums)
    
    if debug:
        print("L: ", L)
        
    if L == 1:
        return sums[0]
    terms = []
    left = expand_mat_sums(sums[:L//2], debug).args
    right = expand_mat_sums(sums[L//2:], debug).args
    
    if debug:
        print("left: ", left)
        print("right: ", right)
            
    terms = [a*b for a in left for b in right]
    added = SuperMatAdd(*terms)
    
    if debug:
        print("terms: ", terms)
        print("added: ", added)
        
    return added

def expand_matmul(expr, debug=False):
    """
        Expands MatMul objects e.g. C*(A+B) -> C*A + C*B
        Based on 'def _eval_expand_mul' in sympy.core.mul
    """
    from symgp.superexpressions import SuperMatAdd
    
    if debug:
        print("expr: ", expr)
         
    sums, rewrite = [], False
    for factor in expr.args:
        if debug:
            print("factor: ", factor)
    
        if isinstance(factor, MatrixExpr) and factor.is_MatAdd:
            sums.append(factor)
            rewrite = True
        else:
            sums.append(Basic(factor))
    
    if debug:
        print("sums: ", sums)
    
    if not rewrite:
        return expr
    else:
        if sums:
            terms = expand_mat_sums(sums, debug).args
            
            if debug:
                print("terms: ", terms)
                
            args = []
            for term in terms:
                t = term
                if isinstance(t,MatrixExpr) and t.is_MatMul and any(a.is_MatAdd if isinstance(a,MatrixExpr) else False for a in t.args):
                    t = expand_matmul(t, debug)
                    
                if debug:
                    print("t: ", t)
                    
                args.append(t)
            return SuperMatAdd(*args).doit()
        else:
            return expr
            
def expand_matexpr(expr, debug=False):
    """
        Expands matrix expressions (MatrixExpr)
    """
    from symgp.superexpressions import SuperMatAdd
    
    if debug:
        print("expr: ", expr)
        
    if expr.is_MatAdd:
        args = []
        args.extend([expand_matexpr(a, debug) if a.is_MatMul else a for a in expr.args])
        
        if debug:
            print("args: ", args)
        
        return SuperMatAdd(*args).doit()
    elif expr.is_MatMul:
        return expand_matmul(expr, debug).doit()
    else:
        return expr.doit()
                            
def collect(expr, syms, muls, evaluate=None, debug=False):
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
    
    if debug:
        print("evaluate: ",evaluate)
        
    if evaluate is None:
        evaluate = global_evaluate[0]
        if debug:
            print("evaluate: ",evaluate)
        

    def make_expression(terms):
        if debug:
            print("terms: ",terms)
        
        product = [term for term in terms]

        if debug:
            print("SuperMatMul(*product): ",SuperMatMul(*product))
            
        return SuperMatMul(*product)
    
    def parse_expression(terms, pattern, mul):
        """Parse terms searching for a pattern.
        terms is a list of MatrixExprs
        pattern is an expression treated as a product of factors
        
        Returns tuple of unmatched and matched terms.
        """
        if debug:
            print("terms: ", terms)
            
        if (not isinstance(pattern, MatrixSymbol) and 
            not isinstance(pattern, Transpose) and 
            not isinstance(pattern, Inverse) and
            not isinstance(pattern, MatAdd)):
            pattern = pattern.args
        else:
            pattern = (pattern,)
        
        if debug:    
            print("pattern: ", pattern)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            if not isinstance(pattern, MatAdd):      
                pattern = [elem for elem in pattern]
            
            if debug:
                print("pattern: ",pattern)

            terms = terms[:]  # need a copy
            
            if debug:
                print("terms: ",terms)
                
            elems = []

            for elem in pattern:
                
                if debug:
                    print("pattern: (%s)" % (elem))
                    
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
                    
                    if debug:
                        print("terms[%s]: %s" % (k,terms[k]))

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
            
            if debug:
                print("[_f for _f in terms if _f]: ",[_f for _f in terms if _f])

            return [_f for _f in terms if _f], elems

    if debug:
        print("expr: ",expr)

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
    
    if debug:
        print("syms: ", syms)

    expr = sympify(expr)
    
    if debug:
        print("expr: ",expr)

    # Get all expressions in summation
    
    # If syms[0] is a MatAdd, collect terms in summa that are equal to to the symbol
    if isinstance(syms[0], MatAdd) and isinstance(expr, MatAdd):
        matched, rejected = ZeroMatrix(expr.shape[0],expr.shape[1]), expr
        
        if debug:
            print("syms[0]: ",syms[0])
            print("matched: ", matched)
            print("rejected: ", rejected)
            
        for s in syms[0].args:
            
            if debug:
                print("s: ",s)
                
            for t in rejected.args:
                if debug:
                    print("t: ",t)
                if s == t:
                    if debug:
                        print("s==t")
                    matched += t
                    rejected -= t
                    break
                    
        if debug:
            print("matched: ",matched)
            print("rejected: ",rejected)
            
        summa = [matched]
        
        if debug:
            print("summa (MatAdd): ", summa)
            
        if matched != expr:
            if isinstance(rejected,MatAdd):
                summa += [i for i in rejected.args]
            else:
                summa += [rejected]
    else:
        summa = [i for i in expr.args]
    
    if debug:
        print("summa: ",summa)
        
    collected, disliked = defaultdict(list), ZeroMatrix(expr.shape[0],expr.shape[1])
    
    # For each product in the summation, match the first symbol and update collected/
    # disliked depending on whether a match was/wasn't made.
    for product in summa:
        if isinstance(product, MatMul):
            terms = [i for i in product.args]
        else:
            terms = [product]
        
        if debug:
            print("terms: ", terms)

        # Only look at first symbol
        symbol = syms[0]
        
        if debug:     
            print("symbol: ", symbol)
        result = parse_expression(terms, symbol, mul)
        
        if debug:
            print("result: ", result)

        # If symbol matched a pattern in terms, we collect the multiplicative terms for the 
        # symbol into a dictionary 'collected'
        if result is not None:
            terms, elems = result
                
            index = Identity(elems[0].shape[0])
            for elem in elems:
                
                if debug:
                    print("elem: ",elem)
                index *= elem
                
                if debug:
                    print("index: ",index)
    
            terms = make_expression(terms)
            if isinstance(terms, Number):
                if mul == 'left':
                    terms = SuperMatMul(Identity(index.shape[1]),terms)
                else:
                    terms = SuperMatMul(Identity(index.shape[0]),terms)
            
            if debug:
                print("terms: ",terms)
                print("index: ",index)
            collected[index].append(terms)
        else:
            # none of the patterns matched
            disliked += product
            if debug:
                print("disliked: ",disliked)
            
    # add terms now for each key
    if debug:
        print("collected: ",collected.items())
    collected = {k: SuperMatAdd(*v) for k, v in collected.items()}
    if isinstance(syms,list) and isinstance(muls,list):
        if debug:
            print("Second collect")
        second_mul = muls[1]
        first_sym, second_sym = syms 
        collected[first_sym] = collect(collected[first_sym],[second_sym],second_mul)
        if debug:
            print("Finished second collect")
    
    if debug:
        print("collected: ",collected)
    
    if not disliked.is_ZeroMatrix:
        if mul == 'left':
            collected[Identity(disliked.shape[0])] = disliked
        else:
            collected[Identity(disliked.shape[1])] = disliked
    if debug:
        print("collected: ",collected)
        print("mul: ",mul)

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
 
def accept_inv_lemma(e, start, end, debug=False):
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
        
    if debug:
        print("Matched E^{-1}")
            
    # Match E^{-1}F
    if not checkSym(arg_2):
        return False
        
    if debug:
        print("Matched E^{-1}F")
        
    # Match E^{-1}F({MatExpr})^{-1}    
    if not checkMatExpr(arg_3, Inverse):
        return False
        
    if debug:
        print("Matched E^{-1}F({MatExpr})^{-1}")
            
    # Match E^{-1}F({MatAdd})^{-1}    
    if not checkMatExpr(arg_3.arg, MatAdd):
        return False
        
    if debug:
        print("Matched E^{-1}F({MatAdd})^{-1}")
            
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
        
    if debug:
        print("Matched E^{-1}F(A+B)^{-1}")
        
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
        
    if debug:
        print("Matched E^{-1}F(A + GCD)^{-1} or E^{-1}F(A + (-1)*GCD)^{-1}")
        
    # Successful match
    return True
                            
def simplify(expr, debug=False):
    """
       A simplification algorithm
    """ 
    
    from symgp.superexpressions import SuperMatSymbol
    
    depth = get_max_depth(expand_to_fullexpr(expr))
    
    simps = []    # The simplified expressions we have obtained with the associated substitutions
    subs = {}     # Pairs substituted expressions with the substitutions made
    usedSubs = []   # The expressions we have substituted we have used so far
    
    # Get the expressions at every depth
    #exprs_by_depth = get_exprs_at_depth(expr, range(depth+1))
    
    usedNames = SuperMatSymbol.getUsedNames()
    
    if debug:
        print("depth: ", depth)
        print("usedNames: ", usedNames)
    
    min_expr = expr    
    for d in range(depth, -1, -1):
        
        if debug:
            print("d: ",d)
            print("min_expr: ",min_expr)
            
        # Get the exprs at each depth for the new shortest expressions
        exprs_by_depth = get_exprs_at_depth(min_expr, range(depth+1))
        
        if debug:
            print("exprs_by_depth: ", exprs_by_depth)
            
        sub_exprs = exprs_by_depth[d]
        
        if debug:
            print("sub_exprs: ", sub_exprs)
        
        min_syms = math.inf
        
        # For each sub expression at level d check for copies in other parts of expressions
        for s in sub_exprs:
            if debug:
                print("s: ", s)
            
            repetitions = 0
            
            # Find other similar expressions to s
            for k in exprs_by_depth.keys():
                if k == d:
                    continue
                
                if s in exprs_by_depth[k]:    
                    repetitions += exprs_by_depth[k].count(s)  
            
                
            if debug:
                print("repetitions: ", repetitions)
                
            # Make replacements if expression 's' appears more than twice throughout expression
            if (repetitions > 0 or accept_inv_lemma(s,0,len(s.args)-1)) and s not in usedSubs:
                
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
                    if debug:
                        print("c: ",c)
                    i = 0
                    r_name = c + '_{' + str(i) + '}'
                    while r_name in usedNames and i < 99:
                        if debug:
                            print("r_name: ", r_name)
                            print("i: ", i)
                        i += 1
                        r_name = c + '_{' + str(i) + '}'
                    
                    if debug:
                        print("r_name: ", r_name)
                                 
                    if not r_name in usedNames:
                        r = SuperMatSymbol(s.shape[0], s.shape[1], r_name, expanded=s)
                        
                        repl_dict = {s: r}         
                        simp_expr = replace(min_expr, repl_dict).doit()
                        
                        if debug:
                            print("repl_dict: ", repl_dict)
                            print("simp_expr: ", simp_expr)
                            print("min_syms: ", min_syms)
                    
                        if not subs.get(s):
                            subs[s] = r
                            
                        simps.append(simp_expr.doit())
                        
                        num_syms = get_num_symbols(simp_expr)
                        if num_syms < min_syms:#repetitions >= max_repetitions:
                            min_syms = num_syms
                            #max_repetitions = repetitions
                            min_expr = simp_expr.doit()
                         
                        
                        
                        if debug:
                            print("min_syms: ",min_syms)
                            print("simps: ", simps)
                            
                        # Check if we can collect any symbols on simp_expr
                        # If we can add to simps
                        if isinstance(simp_expr, MatAdd):
                            ends_of_expr_collection = get_ends(simp_expr,debug)
                            
                            if debug:
                                print("ends_of_expr_collection: ", ends_of_expr_collection)
                                
                            for ends_of_expr in ends_of_expr_collection:
                                if debug:
                                    print("ends_of_expr: ",ends_of_expr)
                                    
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
                            
                                if debug:
                                    print("ends_dict_left: ",ends_dict_left)
                                    print("ends_dict_right: ",ends_dict_right)
                                    print("ends_dict_both: ",ends_dict_both)
                                    print("simps(before): ",simps)
                                
                                # If there are two or more repetitions of a symbol, collect        
                                for key, val in ends_dict_left.items():
                                    if debug:
                                        print("simp_expr: ",simp_expr)
                                        print("key: ",key)
                                    simped = collect(simp_expr,key,'left').doit()
                                    if debug:
                                        print("simped (left): ",simped)
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)
                            
                                for key, val in ends_dict_right.items():
                                    simped = collect(simp_expr,key,'right').doit()
                                    if debug:
                                        print("simped (right): ",simped)
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)
                            
                                # For cases where both ends are repeated two or more times (e.g. A*P*A + A*Q*A + B), collect
                                for key, val in ends_dict_both.items():
                                    simped = collect(simp_expr,[key[0],key[1]],['left','right']).doit()
                                    if debug:
                                        print("simped (both): ",simped)
                                    if len(val) >= 2 and not simped in simps:
                                        simps.append(simped)
                            
                                if debug:
                                    print("simps (MatAdd): ",simps)          
                        break
                
                
                # Make replacement if r was created
                # if 'r' in locals():
                    
        
        #prev_subs += min_subs
                
                    
    return simps, subs            
                                              

######## Quick creation of variables/constants ########
def variables(var_names, var_shapes, debug=False):
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
        
    #var_shapes = [(shape,1) for shape in var_shapes if isinstance(shape, Symbol)]
    
    return (Variable(name, shape[0], shape[1]) for name, shape in zip(var_names, var_shapes))
        
def constants(const_names, const_shapes, debug=False):

    from symgp.superexpressions import Constant
    
    if isinstance(const_names, str):
        const_names = const_names.split(" ")
    
    # Lists must be of same length
    assert(len(const_names) == len(const_shapes))
    
    for i, shape in enumerate(const_shapes):
        if isinstance(shape, Symbol):
            const_shapes[i] = (shape,1)
        
    #var_shapes = [(shape,1) for shape in var_shapes if isinstance(shape, Symbol)]
    
    return (Constant(name, shape[0], shape[1]) for name, shape in zip(const_names, const_shapes))


######## Useful functions to get info about expressions ########
def get_exprs_at_depth(expr, depths, debug=False):
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
    
    if debug:
        print("depths: ",depths)
        
    exprs_at_depths = defaultdict(list)                        
    stack = [{expr: 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        
        if debug:
            print("sub_expr: %s, level: %s" % (sub_expr, level))
            
        if level in depths and (isinstance(sub_expr, MatAdd) or isinstance(sub_expr, MatMul)):
            
            if debug:
                print("sub_expr: ", sub_expr)
                
            if isinstance(sub_expr, MatAdd) and len(sub_expr.args) > 2:    # Substitute all permutations of 3 arg MatAdds
                sub_expr_perms = get_permutations(sub_expr)
                exprs_at_depths[level].extend(sub_expr_perms)
            elif isinstance(sub_expr, MatMul) and len(sub_expr.args) > 2:    # Substitute 
                l = len(sub_expr.args)
                start, end = 0, 2
                
                while end < l:
                    if (accept_inv_lemma(sub_expr,start,end, debug)):
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
            
            if debug:
                print("exprs_at_depth[%s]: %s" % (level, exprs_at_depths[level]))
                
        #print("sub_expr: ",srepr(sub_expr))
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            
            # Add all combinations of summations up to length len(sub_expr.args)
            #if isinstance(sub_expr, MatAdd) and len(sub_expr.args) > 2:
            
            if debug:    
                print("sub_expr.args: ", sub_expr.args)
                
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                #print(type(arg),arg)
                stack.append({arg: level+1})
                
    return exprs_at_depths

def get_ends(expr, debug=False):
    """
        Returns the left and right matrices of the args of the MatAdd expression, expr.
    
        For example for A*Q*B + 2*C + D*E, we return [(A,B), (C,), (D,E)]
                 or for (Q+A)*R*(Q+A) + Q + A we return [(Q+A,Q+A), (Q+A,)]
    """
    
    from symgp.superexpressions import SuperMatMul, SuperMatAdd
    
    if debug:
        print("expr: ",expr.doit())
        
    #assert(expr, MatMul)
    
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
    
    if debug:
        print("expr_args: ",expr_args)
    
    # Remove numbers in expr_args
    #for i in range(len(expr_args)):
    #    if isinstance(expr_args[i].args[0], Number):
    #        expr_args[i] = type(expr_args[i])(*expr_args[i].args[1:]).doit()
    
    """if debug:
        print("expr_args: ", expr_args)"""
    
    mmul_to_rem = {}   # Pairs a MatMul to the remainder keyed by the ends. We ignore expressions of form {Number}*A where
                       # A is a MatSym, MatTrans or MatInv
                       
    for a in expr_args:
        if debug:
            print("a.as_coeff_mmul: ", a.as_coeff_mmul())
        a_mmul = a.as_coeff_mmul()[1].doit()
        if isinstance(a, MatMul):
            #a = a.as_coeff_mmul()[1]
            ends.append((a_mmul.args[0],a_mmul.args[-1]))
            mmul_to_rem[(a_mmul.args[0],a_mmul.args[-1])] = (a,(expr - a).doit())
        else:
            #a = a.as_coeff_mmul()[1]
            ends.append((a_mmul,))
        
        if debug:
            print("a_mmul: ",a_mmul)
    
    ends_collection.append(ends)
    
    if debug:
        print("ends: ",ends)
        print("mmul_to_rem: ",mmul_to_rem)
    
    for ends_mmul, val in mmul_to_rem.items():
        if debug:
            print("ends_mmul, val: %s, %s"%(ends_mmul, val))
            
        for end in ends_mmul:
            if isinstance(end,MatAdd):
                rem = val[1]
                match = [elem for elem in get_permutations(val[1],debug) if elem==end]
                if debug:
                    print("end: ",end)
                    print("rem: ",rem)
                    print("match: ",match)
                    
                if len(match) > 1:
                    raise Exception("More than one match found: %s"%(match))
                
                if len(match) > 0:
                    new_ends = [ends_mmul]
                    new_ends.append((match[0],))
                    for arg in match[0].args:
                        rem = (rem - arg).doit()
                    
                    if debug:
                        print("rem: ",type(rem))
                        
                    # Get remaining elements
                    if isinstance(rem, MatMul):
                        for arg in rem.args:
                            if isinstance(arg, MatMul):
                                new_ends.append((arg.args[0],arg.args[-1]))
                            else:
                                new_ends.append((arg,))
                    else:
                        new_ends.append((rem,))
                    
                    if debug:
                        print("new_ends: ",new_ends)        
                    #new_ends.extend([e for e in ends if e not in new_ends])
                    if not new_ends in ends_collection:
                        ends_collection.append(new_ends)
        
        
    # Check and correct for case where ends are MatAdds
    """singles = [end for end in ends if len(end)==1]
    doubles = [end for end in ends if len(end)==2]
    
    if debug:
        print("singles: ",singles)
        print("doubles: ",doubles)
    
    # Loop through each double ('(A,B)') and for each element check if it is MatAdd and expression matches a permutation
    # of args of rest of expression. If so we add to ends_collection.     
    for d in doubles:
        rem = 
        for d_i in d:
            if debug:
                print("d_i: ",d_i)
                
            if isinstance(d_i, MatAdd):
                l = len(d_i.args)
                
                if debug:
                    print("l: ",l)
                    
                perms = [e.doit() for e in get_permutations(SuperMatAdd(*[s[0] for s in singles]).doit()) if len(e.args)==l]
                
                if debug:
                    print("get_permutations(SuperMatAdd(*[s[0] for s in singles]).doit()): ",get_permutations(SuperMatAdd(*[s[0] for s in singles]).doit()))
                    print("perms: ",perms)
                    
                for p in perms:
                    if p == d_i:
                        new_ends = [e for e in doubles]      # Add current doubles
                        new_ends.append((SuperMatMul(p).doit(),))      # Add the MatAdd we have matched
                        new_ends.extend([e for e in singles if e[0] not in p.args])    # Add the remaining singles
                        ends_collection.append(new_ends)"""
                            
    return ends_collection 

def get_num_symbols(expr, debug=False):
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
            
        #print("sub_expr: ",srepr(sub_expr))
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            #print(sub_expr.args)
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                #print(type(arg),arg)
                stack.append({arg: level+1})
    
    return numSyms

def display_expr_tree(expr, debug=False):
    """
        Visualizes the expression tree for the given expression
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
                              
    stack = [{expand_to_fullexpr(expr): 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        #print("sub_expr: ",srepr(sub_expr))
        
        print("-" + 4*level*"-",sub_expr)
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            #print(sub_expr.args)
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                #print(type(arg),arg)
                stack.append({arg: level+1})

def get_max_depth(expr, debug=False):
    """
        Get the maximum depth of the expression tree down to the lowest symbol 
    """
    
    from symgp.superexpressions import SuperDiagMat, SuperBlockDiagMat
    
    depth = 0                          
    stack = [{expr: 0}]
    
    while len(stack) > 0:
        sub_expr, level = list(stack.pop().items())[0]
        #print("sub_expr: ",srepr(sub_expr))
        
        if (isinstance(sub_expr, MatMul) or isinstance(sub_expr, MatAdd) or 
            isinstance(sub_expr, Inverse) or isinstance(sub_expr, Transpose) or
            isinstance(sub_expr, SuperDiagMat) or isinstance(sub_expr, SuperBlockDiagMat)):
            #print(sub_expr.args)
            for arg in reversed(sub_expr.args):   # TODO: Why do we need to reverse?
                #print(type(arg),arg)
                stack.append({arg: level+1})
            
            depth = level + 1 if level+1 > depth else depth
    
    return depth

def get_permutations(expr, debug=False):
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

def get_var_coeffs(expr, var, debug=False):
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
    
    if debug:
        print("expr: ", expr)
        print("var: ", var)
        
    coeffs = [ZeroMatrix(expr.shape[0],v.shape[0]) for v in var]
    
    if debug:
        print("coeffs: ", coeffs)
         
    # Search the expression tree for each variable in var then add coefficient to list
    if expr.is_MatAdd:
        for arg in expr.args:
            
            if debug:
                print("arg: ", arg)
                
            if arg in var:
                for i, v in enumerate(var):
                    if arg == v:
                        coeffs[i] = arg.as_coeff_mmul()[0]
                        if debug:
                            print("coeffs["+str(i)+"]: ", coeffs[i])
            else:
                for arg2 in arg.args:
                    if debug:
                        print("arg2: ",arg2)
                        
                    if arg2 in var:
                        for i, v in enumerate(var):
                            if arg2 == v:
                                coeffs[i] = SuperMatMul(*[c for c in arg.args if c != arg2]).doit()
                                if debug:
                                    print("coeffs["+str(i)+"]: ", coeffs[i])
        rem = SuperMatAdd(*[c for c in expr.args if c not in [c*v for c,v in zip(coeffs,var)]]).doit()
    elif expr.is_MatMul:
        rem = expr
        for arg in expr.args:
            if debug:
                print("arg: ", arg)
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
    
    if debug:
        print("rem: ",rem)    
    
    return coeffs, rem


######## GUI lexer ########
class diag_token:
    def __init__(self, t):
        self.value = t   # 'diag'|'blockdiag'|'blkdiag'

class operator_token:
    def __init__(self, op):
        self.value = op  # '+'|'-'|'*'

class paren_token:
    def __init__(self, t):
        self.value = t   # ')'|'('|']'|'['|'{'|'}'

class mat_identifier_token:
    def __init__(self, mat):
        self.name = mat   # The name of the matrix variable identifier

class vec_identifier_token:
    def __init__(self, vec):
        self.name = vec   # The name of the vector variable identifier

class kernel_token:
    def __init__(self, name, arg1, arg2):
        self.name = name
        self.arg1 = arg1
        self.arg2 = arg2

class inv_token:
    def __init__(self, mat):
        self.sym = mat

class trans_token:
    def __init__(self, t):
        self.sym = t

def get_tokens(expr, debug=False):
    
    expr = expr.replace(" ","")
    if debug:
        print("expr: ",expr)
      
    # Regex expressions
    digit = re.compile(r"[0-9_]")
    lower_char = re.compile(r"[a-z]")
    upper_char = re.compile(r"[A-Z]")
    operators = re.compile(r"\+|\-|\*")
    diag_op = re.compile(r"diag|blkdiag|blockdiag")
        
    mat_identifier = re.compile(r"{1}(?:{0}|{1}|{2})*".format(\
                            lower_char.pattern, upper_char.pattern, digit.pattern))
    vec_identifier = re.compile(r"{0}(?:{0}|{1}|{2})*".format(\
                            lower_char.pattern, upper_char.pattern, digit.pattern))
        
    kernel = re.compile(r"(?:{0}|{1})\((?:{2}|{3}),(?:{2}|{3})\)".format(\
                            lower_char.pattern, upper_char.pattern, vec_identifier.pattern, mat_identifier.pattern))
                                
    inv_op = re.compile(r"(?:%s|%s)(?:\.I|\^\-1|\^\{\-1\})"%(\
                            mat_identifier.pattern, kernel.pattern))
    inv_op_grouped = re.compile(r"(%s|%s)(?:\.I|\^\-1|\^\{\-1\})"%(\
                            mat_identifier.pattern, kernel.pattern))
    trans_op = re.compile(r"(?:%s|%s|%s)(?:\.T|\'|\^t|\^T|\^\{t\}|\^\{T\})"%(\
                            mat_identifier.pattern, vec_identifier.pattern, kernel.pattern))
    trans_op_grouped = re.compile(r"(%s|%s|%s)(?:\.T|\'|\^t|\^T|\^\{t\}|\^\{T\})"%(\
                            mat_identifier.pattern, vec_identifier.pattern, kernel.pattern))
        
    symbols = re.compile(r"{0}|{1}|{2}|{3}|{4}".format(\
                            mat_identifier.pattern, vec_identifier.pattern, kernel.pattern,\
                            trans_op.pattern, inv_op.pattern))
        
    expr_re = re.compile(r"^(?:({0})(\[))?(\()?({1})(\))?((?:(?:{2})\(?(?:{1})\)?)*)(\])?".format(\
                            diag_op.pattern, symbols.pattern, operators.pattern))
        
    def match2Symbol(s):
        """
            Determines whether expr matches to mat_identifier, vec_identifier, kernel
        """
            
        if mat_identifier.fullmatch(s):
            return mat_identifier_token(s)
        elif vec_identifier.fullmatch(s):
            return vec_identifier_token(s)
        elif kernel.fullmatch(s):
                
            # Break up 's' into the kernel name and the two arguments
            match = s.split("(")
            name = match[0]
                
            arg1, arg2 = match[1].strip(")").split(",")  
           
            return kernel_token(name, arg1, arg2)
        else:
            return None
            
        
    tokens = []
    expr_match = expr_re.fullmatch(expr)
    if expr_match:
        groups = expr_match.groups()
        if debug:
            print("groups: ", groups)
            
        if groups[0]: # diag_op
            tokens.append(diag_token(groups[0]))
        
        if groups[1]: # '['
            tokens.append(paren_token(groups[1]))
            
        if groups[2]: # '('
            tokens.append(paren_token(groups[2]))
            
        if groups[3]: # mat_identifier|vec_identifier|kernel|inv_op|trans_op
                
            token = match2Symbol(groups[3])
            
            # token must be inv_op or trans_op
            if not token:  
                if trans_op.fullmatch(groups[3]):
                    token = match2Symbol(trans_op_grouped.fullmatch(groups[3]).groups()[0])
                    token = trans_token(token)
                else: # inv_op.fullmatch(groups[2]):
                    token = match2Symbol(inv_op_grouped.fullmatch(groups[3]).groups()[0])
                    token = inv_token(token)
                
            tokens.append(token)
            
        if groups[4]: # ')'
            tokens.append(paren_token(groups[4]))
        
            
        right = groups[5]
            
        right_regex = re.compile(r"^({0})(\()?({1})(\))?((?:(?:{0})\(?(?:{1})\)?)*)\]?".format(\
                                    operators.pattern, symbols.pattern))
        while len(right) > 0:
            subgroups = right_regex.fullmatch(right).groups()
            if debug:
                print("right: ", right)
                print("subgroups: ", subgroups)
                
            if subgroups[0]:
                tokens.append(operator_token(subgroups[0]))
                
            if subgroups[1]:
                tokens.append(paren_token(subgroups[1]))
                
            if subgroups[2]:
                token = match2Symbol(subgroups[2])
                
                # token must be inv_op or trans_op
                if not token:
                    if trans_op.fullmatch(subgroups[2]):
                        token = match2Symbol(trans_op_grouped.fullmatch(subgroups[2]).groups()[0])
                        token = trans_token(token)
                    else: # inv_op.fullmatch(groups[2]):
                        token = match2Symbol(inv_op_grouped.fullmatch(subgroups[2]).groups()[0])
                        token = inv_token(token)
                
                tokens.append(token)
                
            if subgroups[3]:
                tokens.append(paren_token(subgroups[3]))
                
            right = subgroups[4]
        
        if groups[6]: # ']'
            tokens.append(paren_token(groups[6]))
        
        return tokens
                    
    else:
        raise Exception("Invalid input")
        
def tokens2string(t, debug=False):
    output = ""
    for token in t:
        if debug:
            print("type(token): ", type(token))
            
        if isinstance(token, diag_token) or isinstance(token, operator_token) or isinstance(token, paren_token):
            output += token.value
        elif isinstance(token, mat_identifier_token) or isinstance(token, vec_identifier_token):
            output += token.name
        elif isinstance(token, inv_token) or isinstance(token, trans_token):
            sym = token.sym
            if isinstance(sym, kernel_token):
                output += sym.name + "(" + sym.arg1 + "," + sym.arg2 + ")"
            else:
                output += sym.name  
            
            if isinstance(token, inv_token):
                output += ".I"
            else:
                output += ".T" 
                    
        else:
            output += token.name +"(" + token.arg1 + ","+ token.arg2 + ")"
            
        if debug:
            print("output: ",output) 
    return output    

        
        
    