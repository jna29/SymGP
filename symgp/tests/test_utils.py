import unittest
from sympy import symbols, Identity, Matrix
from symgp import *
import numpy as np

class UtilsTestCase(unittest.TestCase):
    
    def setUp(self):
        m, n, l = symbols('m n l')
        self.m, self.n, self.l = m, n, l
        
        self.A, self.B, self.C, self.D, self.E, self.F = \
                                utils.constants('A B C D E F', [(m,m), (m,n), (n,m), (n,n), (l,m), (l,n)])
        self.a, self.b, self.c = utils.constants('a b c', [m, n, l])
        
        
    ######## Matrix operations with lists as matrices ########
    def test_matmul(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        a, b, c = self.a, self.b, self.c
        
        # Case 1: Normal (2,2) x (2,3)
        Mat_1 = [[A, B], [C, D]]
        Mat_2 = [[A.T, C.T, E.T], [B.T, D.T, F.T]]
        true_result = [[A*A.T + B*B.T, A*C.T + B*D.T, A*E.T + B*F.T], 
                       [C*A.T + D*B.T, C*C.T + D*D.T, C*E.T + D*F.T]]
        
        self.assertEqual(utils.matmul(Mat_1,Mat_2), true_result)
        
        # Case 2: Exception for (2,3) x (2,2)
        with self.assertRaises(Exception):
            utils.matmul(Mat_2,Mat_1)
        
        # Case 3: (2,3) x (3,)
        vec_1 = [a, b, c]
        true_result = [A.T*a + C.T*b + E.T*c, B.T*a + D.T*b + F.T*c]
        
        self.assertEqual(utils.matmul(Mat_2,vec_1), true_result)
        
        # Case 4: (2,) x (2,3)
        vec_2 = [a.T, b.T]
        true_result = [a.T*A.T + b.T*B.T, a.T*C.T + b.T*D.T, a.T*E.T + b.T*F.T]
        
        self.assertEqual(utils.matmul(vec_2,Mat_2), true_result)
        
        # Case 5: (2,) x (2,)
        vec_2_T = [a, b]
        true_result = [a.T*a + b.T*b]
        
        self.assertEqual(utils.matmul(vec_2,vec_2_T), true_result)
        
        # Case 6: (3,) x (2,) - Exception
        with self.assertRaises(Exception):
            utils.matmul(vec_1,vec_2)
        
        # Case 8: 5*vec_1 and vec_1*5
        true_result = [5*a, 5*b, 5*c]
        
        self.assertEqual(utils.matmul(5,vec_1), true_result)
        self.assertEqual(utils.matmul(vec_1,5), true_result)
        
        # Case 9: 5*Mat_2 and Mat_2*5
        true_result = [[5*A.T, 5*C.T, 5*E.T], [5*B.T, 5*D.T, 5*F.T]]
        
        self.assertEqual(utils.matmul(5,Mat_2), true_result)
        self.assertEqual(utils.matmul(Mat_2,5), true_result)
    
    def test_matadd(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        a, b, c = self.a, self.b, self.c
        
        # Case 1: (2,2) + (2,2)
        Mat_1 = [[A, B], [C, D]]
        Mat_2 = [[A, B], [C, D]]
        true_result = [[A+A, B+B], [C+C, D+D]]
        
        self.assertEqual(true_result, utils.matadd(Mat_1, Mat_2))
        
        # Case 2: (3,) + (2,)
        vec_1 = [a, b, c]
        vec_2 = [a, b]
        
        with self.assertRaises(Exception):
            utils.matadd(vec_1, vec_2)
        
        # Case 3: (3,) + (3,)
        true_result = [a+a, b+b, c+c]
        
        self.assertEqual(true_result, utils.matadd(vec_1, vec_1))
        
        # Case 3: (3,) + (3,) - Error shapes of elements don't match
        vec_3 = [b, a, c]
        with self.assertRaises(Exception):
            utils.matadd(vec_1,vec_3)
        
    def test_mattrans(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        a, b = self.a, self.b
        
        # Case 1: (2,2).T
        true_result = [[A.T, C.T], [B.T, D.T]]
        
        self.assertEqual(true_result, utils.mattrans([[A, B],[C, D]]))
        
        # Case 2: (2,).T
        true_result = [a.T, b.T]
        
        self.assertEqual(true_result, utils.mattrans([a, b]))
        
        # Case 3: (3, 2)
        true_result = [[A.T, C.T, E.T], [B.T, D.T, F.T]]
        
        self.assertEqual(true_result, utils.mattrans([[A, B],[C, D],[E, F]]))
        
        # Case 4: (3,2) - Exception case ([[A, B], C, [D, E]])
        with self.assertRaises(Exception):
            utils.mattrans([[A, B], C, [D, E]])
         
    def test_matinv(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        
        # Case 1: (2,2)
        true_result = [[(A - B*D.I*C).I, -(A - B*D.I*C).I*B*D.I], [-D.I*C*(A - B*D.I*C).I, D.I + D.I*C*(A - B*D.I*C).I*B*D.I]]
        
        self.assertEqual(utils.matinv([[A, B], [C, D]]), true_result)
        
        # Case 2: (1,1)
        true_result = [[A.I]]
        
        self.assertEqual(utils.matinv([[A]]), true_result)
        
        # Case 3: (3,2) - Exception
        with self.assertRaises(Exception):
            utils.matinv([[A, B], [C, D], [F, E]])
        
        # Case 4: (1,1) (non-square) - Exception
        with self.assertRaises(Exception):
            utils.matinv([B.I])
        
        # Case 5: (3,2) - Exception case ([[A, B], C])
        with self.assertRaises(Exception):
            utils.matinv([[A, B], C])
    
    def test_partition_block(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        a, b, c = self.a, self.b, self.c
        
        G, H, I = utils.constants('G H I',[(self.m,self.l), (self.n,self.l), (self.l,self.l)])
        
        # Case 1: Normal 3x3 block, indices = [0,1]
        input_block = [[A,B,G],[C,D,H],[E,F,I]]
        indices = [0,1]
        P_t, Q_t = [[I]], [[E,F]]
        R_t, S_t = [[G],[H]], [[A,B],[C,D]]
        
        P, Q, R, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        P, Q, R, S = utils.partition_block(input_block, [indices, indices])
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        # Case 2: Invalid indices, [0,3]
        indices = [0,3]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 3: Invalid indices, [[0,3],[0,1]]
        indices = [[0,3],[0,1]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 4: Invalid indices, [[0,1],[0,3]]
        indices = [[0,1],[0,3]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 5: Normal 3x3 block, indices = [[0,1],[1]]
        input_block = [[A,B,G],[C,D,H],[E,F,I]]
        indices = [[0,1],[1]]
        P_t, Q_t = [[E,I]], [[F]]
        R_t, S_t = [[A,G],[C,H]], [[B],[D]]
                
        P, Q, R, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        # Case 6: Normal 1x1 matrix block, indices = [0]
        input_block = [[A]]
        indices = [0]
        
        with self.assertRaises(Exception):  # 1x1 case, indices = [0]
            utils.partition_block(input_block, indices)
        
        with self.assertRaises(Exception):  # 1x1 case, indices = [[0]]
            utils.partition_block(input_block, [indices])
        
        # Case 7: 1x1 vector block, indices = [0]
        input_block = [a]
        indices = [0]
        
        with self.assertRaises(Exception):   # 1x1 case
            utils.partition_block(input_block, indices)
            
        # Case 8: 3x1 block, indices = [0,1]
        input_block = [[a],[b],[c]]
        indices = [0,1]
        P_t, S_t = [[c]], [[a],[b]]
        
        P, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 9: Indices = [[0],[1]]
        indices = [[0],[1]]
        
        P, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 10: Invalid indices, indices = [[0,1]]
        indices = [[0,1]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 11: Invalid indices, indices = [[0,2],[1]]
        indices = [[0,2],[1]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 12: Invalid indices, indices = [0,3]
        indices = [0,3]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)   
        
        # Case 13: 1x3 block, indices = [1,2] and indices = [[1],[2]]
        input_block = [a.T, b.T, c.T]
        indices = [1,2]
        P_t, S_t = [a.T], [b.T, c.T]
        
        P, S = utils.partition_block(input_block,indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 14: 1x3 block, indices = [[1],[2]]
        indices = [[1],[2]]
        
        P, S = utils.partition_block(input_block,indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 14: 1x3 block, indices = [1,[2]]. Should raise exception
        indices = [1,[2]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
    
    
    ######## MVG helper functions ########
    def test_get_Z(self):
        pass
    
    
    ######### Search and replace functions ########
    def test_replace_with_num(self):
        
        A, B, C, D, E, F = self.A, self.B, self.C, self.D, self.E, self.F
        a, b, c = self.a, self.b, self.c
        
        # We try both numpy and SymPy Matrix inputs
        # Shapes we use: m, n, l = 2, 3, 4 
    
        # Case 1: A
        expr = A
        d_np = {'A': np.array([[30., 50.], [10., 20.]])}
        d_sympy = {'A': Matrix([[30., 50.], [10., 20.]])}
        true_result = d_np['A']
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        
        # Case 2: a
        expr = a
        d_np['a'] = np.array([1.,2.,3.])
        d_sympy['a'] = Matrix([1.,2.,3.])
        true_result_np = d_np['a']
        true_result_sympy = np.array(d_sympy['a'].tolist(),dtype=np.float32)
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result_np,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result_sympy,decimal=5)
        
        
        # Case 3a: B*b (b.shape = (3,1))
        expr = B*b
        d_np['B'] = np.array([[20., 80., 50.],[25., 100., 60.]])
        d_np['b'] = np.array([[15.],[25.],[35.]])
        d_sympy['B'] = Matrix([[20., 80., 50.],[25., 100., 60.]])
        d_sympy['b'] = Matrix([[15.],[25.],[35.]])
        true_result = np.dot(d_np['B'],d_np['b']) #np.array([[ 4050.], [ 4975.]])
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        # Case 3b: B*b (b.shape = (3,))
        d_np['b'] = np.array([15., 25., 35.])
        true_result = np.dot(d_np['B'],d_np['b']) #np.array([4050., 4975.])
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        
         
        # Case 4a: B.T*a (a = Matrix([[5.], [45.]]))
        expr = B.T*a
        d_np['a'] = np.array([[5.], [45.]])
        d_sympy['a'] = Matrix([[5.], [45.]])
        true_result = np.dot(d_np['B'].T,d_np['a'])#np.array([[1225.], [4900.], [2950.]])
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        # Case 4b: B.T*a (a = Matrix([5., 45.]))
        d_sympy['a'] = Matrix([5., 45.])
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)

        
        # Case 5: (A - 2*B*D.I*C).I
        expr = (A - 2*B*D.I*C).I
        d_np['C'] = np.array([[15., 80.], [45., 65.], [85., 90.]])
        d_np['D'] = np.array([[45., 75., 55.], [90., 35., 85.], [0., 30., 25.]])
        d_sympy['C'] = Matrix([[15., 80.], [45., 65.], [85., 90.]])
        d_sympy['D'] = Matrix([[45., 75., 55.], [90., 35., 85.], [0., 30., 25.]])
        true_result = np.linalg.inv(d_np['A'] - 2*np.dot(d_np['B'],np.dot(np.linalg.inv(d_np['D']),d_np['C'])))
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        
        # Case 6: (A - B*D.I*C).I*a
        expr = (A - 2*B*D.I*C).I*a
        true_result = np.dot(true_result,d_np['a'])
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        
        # Case 7: 5*B
        expr = 5*B
        true_result = 5*d_np['B']
        
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_np),true_result,decimal=5)
        np.testing.assert_almost_equal(utils.replace_with_num(expr,d_sympy),true_result,decimal=5)
        
        
        # Case 8: 1 - raise Exception("Expression should be a MatrixExpr")
        expr = 1
        with self.assertRaises(Exception):
            utils.replace_with_num(expr,d_np)
        
        
        # Cade 9: (A - B*D.I*C).I*E.T (No E matrix supplied) - Exception
        expr = (A - B*D.I*C).I*E.T
        with self.assertRaises(Exception):
            utils.replace_with_num(expr,d_np)
        
        with self.assertRaises(Exception):
            utils.replace_with_num(expr,d_sympy)
        
    def test_replace_with_expanded(self):
        pass
    
    def test_expand_to_fullexpr(self):
        
        m, n = symbols('m n')
        s_a, s_b = symbols('s_a s_b') 
        
        # Case 1: Simple sum of symbols expanded forms
        A = SuperMatSymbol(n, n, name='A', expanded=s_a**2*Identity(n))
        B = SuperMatSymbol(n, n, name='B', expanded=s_b**2*Identity(n))
        
        true_fullexpr = A.expanded + B.expanded
        
        self.assertEqual(utils.expand_to_fullexpr(A+B),true_fullexpr)
        self.assertEqual(utils.expand_to_fullexpr(A+B,1),true_fullexpr)
        
        # Case 2: Two layers of expanded forms
        A = SuperMatSymbol(n, n, name='A', expanded=s_a**2*Identity(n))
        B = SuperMatSymbol(n, n, name='B', expanded=s_b**2*Identity(n))
        C = SuperMatSymbol(n, n, name='C', expanded=A*B)
        D = SuperMatSymbol(n, n, name='D', expanded=B*A)
        
        true_fullexpr = A.expanded+B.expanded+A.expanded*B.expanded+B.expanded*A.expanded
        
        self.assertEqual(utils.expand_to_fullexpr(A+B+C+D),true_fullexpr)
        self.assertEqual(utils.expand_to_fullexpr(A+B+C+D,3),true_fullexpr)
        
    def test_replace(self):
        
        m, n = symbols('m n')
        
        A,B,C,D = utils.constants('A B C D',[(m,m),(m,m),(m,m),(m,m)])
        a,b,u = utils.variables('a b u',[m,m,n])
        K = Kernel()
        
        ######## Matrix replacements ########
        # Case 1: 'A*B' with R_1 in A*B*C
        expr = A*B*C
        R_1 = SuperMatSymbol(m,m,'R_1')
        repl = [(A*B, R_1)]
        correct_result = R_1*C
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 2: 'A*B' with R_1 in A*B*C*A*B
        expr = A*B*C*A*B
        correct_result = R_1*C*R_1
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 3: 'A*(B+D)' with R_2 in A*(B+D)*C
        expr = A*(B+D)*C
        R_2 = SuperMatSymbol(m,m,'R_2')
        repl = [(A*(B+D), R_2)]
        correct_result = R_2*C
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())  
        
        # Case 4: 'A*(B+D)' with R_2 in A*(B+D)*C*A*(B+D)
        expr = A*(B+D)*C*A*(B+D)
        correct_result = R_2*C*R_2
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 5: 'A+B' with R_3 in A+B+C
        expr = A + B + C
        R_3 = SuperMatSymbol(m,m,'R_3')
        repl = [(A+B, R_3)]
        correct_result = R_3+C
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 6: 'A+B' with R_3 in A+B+C*(A+B)
        expr = A + B + C*(A+B)
        repl = [(A+B, R_3)]
        correct_result = R_3+C*R_3
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 7: 'A+B*C*D' with R_4 in A+B*C*D+D
        expr = A + B*C*D + D
        R_4 = SuperMatSymbol(m,m,'R_4')
        repl = [(A+B*C*D, R_4)]
        correct_result = R_4+D
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 8: 'A+B*C*D' with R_4 in 2*A+B*C*D
        """expr = 2*A + B*C*D
        repl = [(A+B*C*D, R_4)]
        correct_result = A+R_4
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())"""
        
        # Case 9: 'A+B*C*D' with R_4 in (A+B*C*D)*A
        expr = (A+B*C*D)*A
        repl = [(A+B*C*D, R_4)]
        correct_result = R_4*A
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 10: 'A+B*C*D' with R_4 in (A+B*C*D)*A*(A+B*C*D)
        expr = (A+B*C*D)*A*(A+B*C*D)
        repl = [(A+B*C*D, R_4)]
        correct_result = R_4*A*R_4
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 11: 'A+B*C*D' with R_4 in (A+B*C*D)*A*(A+B*C*D).I
        expr = (A+B*C*D)*A*(A+B*C*D).I
        repl = [(A+B*C*D, R_4)]
        correct_result = R_4*A*R_4.I
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        ######## Kernel replacements ########
        # Case 12: Kernel 'K' with 'Q' in K(a,b)*K(b,a)
        """expr = K(a,b)*K(b,a)
        Q = Kernel(name='Q')
        repl = {K: Q}
        correct_result = Q(a,b)*Q(b,a)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 13: Kernel 'K' with 'Q' in K(a,b)*A*K(b,a)
        expr = K(a,b)*A*K(b,a)
        repl = {K: Q}
        correct_result = Q(a,b)*A*Q(b,a)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())"""
        
        # Case 14: Kernel 'K(*,u)*K(u,u)*K(u,*)' with 'Q(*,*)' in 
        #          A + K(a,u)*K(u,u)*K(u,b)
        expr = A + K(a,u)*K(u,u)*K(u,b)
        M_uu = SuperMatSymbol(n,n,name='M_{u,u}',dep_vars=[u],expanded=K(u,u))
        Q = Kernel(sub_kernels=[K,K], kernel_type='mul', mat=M_uu, name='Q')
        repl = [Q]
        correct_result = A + Q(a,b)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 15: Kernel 'K(*,u)*K(u,u)*K(u,*)' with 'Q(*,*)' in 
        #          A + K(a,u)*K(u,u)*K(u,b)*B*C.I*D*K(b,u)*K(u,u)*K(u,a)
        expr = A + K(a,u)*K(u,u)*K(u,b)*B*C.I*D*K(b,u)*K(u,u)*K(u,a)
        repl = [Q]
        correct_result = A + Q(a,b)*B*C.I*D*Q(b,a)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 16: Kernel 'K(*,u)*K(u,u)*K(u,*)' with 'Q(*,*)' in 
        #          A + K(a,u)*K(u,u)*K(u,b)*B*(C + K(a,u)*K(u,u)*K(u,a)).I*D*K(b,u)*K(u,u)*K(u,a)
        expr = A + K(a,u)*K(u,u)*K(u,b)*B*(C + K(a,u)*K(u,u)*K(u,a)).I*D*K(b,u)*K(u,u)*K(u,a)
        repl = [Q]
        correct_result = A + Q(a,b)*B*(C + Q(a,a)).I*D*Q(b,a)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
        
        # Case 17: Kernel 'K(*,u)*K(u,u)*K(u,*)' with 'Q(*,*)' in 
        #          A + K(a,u)*K(u,u)*K(u,b)*B*(C + K(a,u)*K(u,u)*K(u,a)).I*D*K(b,u)*K(u,u)*K(u,a) + 
        #          K(b,u)*K(u,u)*K(u,b)
        expr = A + K(a,u)*K(u,u)*K(u,b)*B*(C + K(a,u)*K(u,u)*K(u,a)).I*D*K(b,u)*K(u,u)*K(u,a) + K(b,u)*K(u,u)*K(u,b)
        repl = [Q]
        correct_result = A + Q(a,b)*B*(C + Q(a,a)).I*D*Q(b,a) + Q(b,b)
        
        self.assertEqual(utils.replace(expr,repl),correct_result.doit())
          
    def test_replace_with_SuperMat(self):
        pass
    
    
    ######## LaTeX printing ########
    def test_matLatex(self):
        pass
    
    def test_updateLatexDoc(self):
        pass
    
    
    ######## Expression conversion functions ########
    def test_expand_mat_sums(self):
        pass
    
    def test_expand_matmul(self):
        pass
    
    def test_expand_matexpr(self):
        pass
    
    def test_collect(self):
        # Case 1: A + B + C + (A+B)*(s_y^2*I + D)
        m, s_y = symbols('m \u03c3_y')
        A, B, C, D = utils.constants('A B C D',[(m,m)]*4)
        
        expr = A + B + C + (A+B)*(s_y**2*Identity(m) + D)
        true_expr = (A + B)*(s_y**2*Identity(m) + D + Identity(m)) + C 
        self.assertEqual(utils.collect(expr,[A+B],'left'),true_expr)
    
    def test_accept_inv_lemma(self):
        pass
    
    def test_simplify(self):
        pass
    
    
    ######## Quick creation of variables/constants ########
    def test_variables(self):
        pass
    
    def test_constants(self):
        pass
    
    
    ######## Useful functions to get info about expressions ########
    def test_get_exprs_at_depth(self):
        pass
    
    def test_get_ends(self):
        pass
    
    def test_get_num_symbols(self):
        pass
    
    def test_display_expr_tree(self):
        pass
    
    def test_get_max_depth(self):
        pass
    
    def test_get_permutations(self):
        pass
    
    def test_get_var_coeffs(self):
        pass
          
    

def mat_ops_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_matmul'))
    suite.addTest(UtilsTestCase('test_matadd'))
    suite.addTest(UtilsTestCase('test_mattrans'))
    suite.addTest(UtilsTestCase('test_matinv'))
    suite.addTest(UtilsTestCase('test_partition_block'))
    return suite

def mvg_help_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_get_Z'))
    return suite

def search_and_replace_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_replace_with_num'))  # Same as evaluate_expr
    suite.addTest(UtilsTestCase('test_replace_with_expanded'))
    suite.addTest(UtilsTestCase('test_expand_to_fullexpr'))
    suite.addTest(UtilsTestCase('test_replace'))
    #suite.addTest(UtilsTestCase('test_replace_with_SuperMat'))
    return suite

def latex_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_matLatex'))
    uite.addTest(UtilsTestCase('test_updateLatexDoc'))
    return suite

def expr_conversion_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_expand_mat_sums'))
    suite.addTest(UtilsTestCase('test_expand_matmul'))
    suite.addTest(UtilsTestCase('test_expand_matexpr'))
    suite.addTest(UtilsTestCase('test_collect'))
    suite.addTest(UtilsTestCase('test_accept_inv_lemma'))
    suite.addTest(UtilsTestCase('test_simplify'))
    return suite

def quick_creation_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_variables'))
    suite.addTest(UtilsTestCase('test_constants'))
    return suite

def expr_helpers_suite():
    suite = unittest.TestSuite()
    suite.addTest(UtilsTestCase('test_get_exprs_at_depth'))
    suite.addTest(UtilsTestCase('test_get_ends'))
    suite.addTest(UtilsTestCase('test_get_num_symbols'))
    suite.addTest(UtilsTestCase('test_display_expr_tree'))
    suite.addTest(UtilsTestCase('test_get_max_depth'))
    suite.addTest(UtilsTestCase('test_get_permutations'))
    suite.addTest(UtilsTestCase('test_get_var_coeffs'))
    return suite

# TODO: Find a better way to avoid repetition here
def print_test_results(result):
    print("Number of tests run: %d" % (test_result.testsRun))
    
    print("Errors: ")
    for test_case, error in test_result.errors:
        print("Test case: %s, Error: %s" % (test_case.id(), error))
    
    print("Failures: ")
    for test_case, failure in test_result.failures:
        print("Test case: %s, Failure: %s" % (test_case.id(), failure))
    
    print("Skipped: ")
    for test_case, skipped in test_result.skipped:
        print("Test case: %s, Skipped: %s" % (test_case.id(), skipped))
    
    print("Expected Failures: ")
    for test_case, expected_failure in test_result.expectedFailures:
        print("Test case: %s, Expected failure: %s" % (test_case.id(), expected_failure))
    
    print("Unexpected Successes: ")
    for test_case, unexpected_success in test_result.unexpectedSuccesses:
        print("Test case: %s, Unexpected success: %s" % (test_case.id(), unexpected_success))
    
if __name__ == '__main__':
    suite = search_and_replace_suite()
    test_result = unittest.TestResult()
    test = suite.run(test_result)
    if test_result.wasSuccessful():
        print("Success!!!")
        print_test_results(test_result)
    else:
        print("Tests failed!!!")
        print_test_results(test_result)
    
    
    
    