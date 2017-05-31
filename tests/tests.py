import unittest
from sympy import symbols, Identity
from symgp import *

"""class TestSuperMatSymbol(unittest.TestCase):

    def test_partition(self):
        
        # Test case 1: Normal 2x2
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)"""

class TestUtils(unittest.TestCase):
    
    @unittest.skip("Not tested now")
    def test_partition_block(self):
        
        # Case 1: Normal 4x4 block, indices = [1,3]
        input_block = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
        indices = [1,3]
        P_t, Q_t = [[1,3],[9,11]], [[2,4],[10,12]]
        R_t, S_t = [[5,7],[13,15]], [[6,8],[14,16]]
        
        P, Q, R, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        # Case 2: Normal 3x4 block, indices = [[0,2],[1]]
        input_block = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        indices = [[1],[0,2]]
        P_t, Q_t = [[2,4],[10,12]], [[1,3],[9,11]]
        R_t, S_t = [[6,8]], [[5,7]]
        
        P, Q, R, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        # Case 3: Normal 5x5 block, indices = [[0,2],[0,2]]
        input_block = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
        indices = [0,2]
        P_t, Q_t = [[7,9,10],[17,19,20],[22,24,25]], [[6,8],[16,18],[21,23]]
        R_t, S_t = [[2,4,5],[12,14,15]], [[1,3],[11,13]]
        
        P, Q, R, S = utils.partition_block(input_block, [indices,indices])
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        P, Q, R, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(Q,Q_t)
        self.assertEqual(R,R_t)
        self.assertEqual(S,S_t)
        
        # Case 4: Normal 1x1 block, indices = [0]
        input_block = [1]
        indices = [0]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        with self.assertRaises(Exception):  # [[0]]
            utils.partition_block(input_block, [indices])
            
        # Case 5: Normal 1x1 block, indices = [0]
        input_block = [[1]]
        indices = [0]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        with self.assertRaises(Exception):  # [[0]]
            utils.partition_block(input_block, [indices])
            
        # Case 6: 5x1 block, indices = [1,3]
        input_block = [[1],[2],[3],[4],[5]]
        indices = [1,3]
        P_t, S_t = [[1],[3],[5]], [[2],[4]]
        
        P, S = utils.partition_block(input_block, indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 7: 5x1 block, indices = [[1,3],[1]]. Should raise Exception
        input_block = [[1],[2],[3],[4],[5]]
        indices = [[1,3],[1]]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
        
        # Case 8: 1x5 block, indices = [1,3]
        input_block = [1,2,3,4,5]
        indices = [1,3]
        P_t, S_t = [1,3,5], [2,4]
        
        P, S = utils.partition_block(input_block,indices)
        self.assertEqual(P,P_t)
        self.assertEqual(S,S_t)
        
        # Case 9: 1x5 block, indices = [1,[3],5]. Should raise exception
        input_block = [1,2,3,4,5]
        indices = [1,[3],5]
        
        with self.assertRaises(Exception):
            utils.partition_block(input_block, indices)
    
    @unittest.skip("Not tested now")
    def test_matmul(self):
        
        # Case 1: Normal (4,5) x (5,3)
        a = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
        b = [[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35]]
        true_result = [[ 435,  450,  465], [1110, 1150, 1190], [1785, 1850, 1915], [2460, 2550, 2640]]
        
        matmul_result = utils.matmul(a,b)
        self.assertEqual(matmul_result, true_result)
        
        # Case 2: Exception for (5,3) x (4,5)
        with self.assertRaises(Exception):
            utils.matmul(b,a)
        
        # Case 3: (4,5) x (5,)
        c = [1, 2, 3, 4, 5]
        true_result = [ 55, 130, 205, 280]
        
        matmul_result = utils.matmul(a,c)
        self.assertEqual(matmul_result, true_result)
        
        # Case 4: (5,) x (5,3)
        true_result = [435, 450, 465]
        
        matmul_result = utils.matmul(c,b)
        self.assertEqual(matmul_result, true_result)
        
        # Case 5: (5,) x (5,)
        true_result = [55]
        
        matmul_result = utils.matmul(c,c)
        self.assertEqual(matmul_result, true_result)
        
        # Case 6: (1,) x (1,)
        d = [4]
        true_result = [16]
        
        matmul_result = utils.matmul(d,d)
        self.assertEqual(matmul_result, true_result)
        
        # Case 7: (1,) x (5,)
        with self.assertRaises(Exception):
            utils.matmul(d,c)
        
        # Case 8: 5*c and c*5
        true_result = [5, 10, 15, 20, 25]
        
        self.assertEqual(utils.matmul(5,c), true_result)
        self.assertEqual(utils.matmul(c,5), true_result)
        
        # Case 9: 5*a and a*5
        true_result = [[  5,  10,  15,  20,  25], [ 30,  35,  40,  45,  50], [ 55,  60,  65,  70,  75], [ 80,  85,  90,  95, 100]]
        
        self.assertEqual(utils.matmul(5,a), true_result)
        self.assertEqual(utils.matmul(a,5), true_result)
    
    @unittest.skip("Not tested now")    
    def test_expand_to_fullexpr(self):
        
        m, n = symbols('m n')
        s_a, s_b = symbols('s_a s_b') 
        
        # Case 1: Simple sum of symbols expanded forms
        A = SuperMatSymbol(m, m, expanded=s_a**2*Identity(m))
        B = SuperMatSymbol(n, n, expanded=s_b**2*Identity(n))
        
        true_fullexpr = A.expanded + B.expanded
        
        self.assertEqual(utils.expand_to_fullexpr(A+B),true_fullexpr)
        self.assertEqual(utils.expand_to_fullexpr(A+B,1),true_fullexpr)
        
        # Case 2: Two layers of expanded forms
        A = SuperMatSymbol(m, m, expanded=s_a**2*Identity(m))
        B = SuperMatSymbol(n, n, expanded=s_b**2*Identity(n))
        C = SuperMatSymbol(m, m, expanded=A*B)
        D = SuperMatSymbol(n, n, expanded=B*A)
        
        true_fullexpr = A.expanded*B.expanded*B.expanded*A.expanded
        
        self.assertEqual(utils.expand_to_fullexpr(A+B),true_fullexpr)
        self.assertEqual(utils.expand_to_fullexpr(A+B,2),true_fullexpr)
        
    def test_replace(self):
        
        # Case 1: A + B + C + (A+B)*(s_y^2*I + D)
        m, s_y = symbols('m \u03c3_y')
        A, B, C, D = utils.constants('A B C D',[(m,m)]*4)
        
        expr = A + B + C + (A+B)*(s_y**2*Identity(m) + D)
        true_expr = (A + B)*(s_y**2*Identity(m) + D + Identity(m)) + C 
        self.assertEqual(utils.collect(expr,[A+B],'left'),true_expr)
#class TestMVG(unittest.TestCase):
#    pass
    

if __name__ == '__main__':
    unittest.main()