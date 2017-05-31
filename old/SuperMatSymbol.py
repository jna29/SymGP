from sympy import MatrixSymbol, BlockMatrix, Symbol, Inverse, Transpose
    
class SuperMatSymbol(MatrixSymbol):
    
    def __new__(cls, name, m, n, expanded=None, blockform=None):
        #print("SuperMatSymbol.__new__ called")
        return super(SuperMatSymbol, cls).__new__(cls, name, m, n)
    
    def __init__(self, name, m, n, expanded=None, blockform=None):
        #print("SuperMatSymbol.__init__ called")
        self.expanded = expanded
        self.blockform = blockform
        self.blockmat = None
        if self.blockform is not None:
            self.blockmat = BlockMatrix(self.blockform)
 
    def inverse(self):
        return SuperMatInverse(self)
    
    I = property(inverse, None, None, 'Matrix inversion')
    
    def transpose(self):
        return SuperMatTranspose(self)
    
    T = property(transpose, None, None, 'Matrix transposition')  
            

class SuperMatInverse(Inverse):
    
    def __new__(cls, mat, expanded=None, blockform=None):
        return super(SuperMatInverse, cls).__new__(cls, mat)
        
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = None
        self.blockform = None
        self.blockmat = None
        if mat.expanded is not None:
            self.expanded = self.expanded.I
        if mat.blockform is not None:
            self.blockform = [[0,0], [0,0]]
            P, Q = mat.blockform[0][0], mat.blockform[0][1]
            R, S = mat.blockform[1][0], mat.blockform[1][1]
            self.blockform[0][0] = (P-Q*S.I*R).I
            self.blockform[0][1] = -self.blockform[0][0]*Q*S.I
            self.blockform[1][0] = -S.I*R*self.blockform[0][0]
            self.blockform[1][1] = S.I + S.I*R*self.blockform[0][0]*Q*S.I 
            self.blockmat = BlockMatrix(self.blockform)
    
    
class SuperMatTranspose(Transpose):
    
    def __new__(cls, mat, expanded=None, blockform=None):
        return super(SuperMatTranspose, cls).__new__(cls, mat)
    
    def __init__(self, mat, expanded=None, blockform=None):
        self.expanded = None
        self.blockform = None
        self.blockmat = None
        if mat.expanded is not None:
            self.expanded = self.expanded.T
        if mat.blockform is not None:
            self.blockform = [[0,0], [0,0]]
            self.blockform[0][0] = mat.blockform[0][0].T
            self.blockform[0][1] = mat.blockform[1][0].T
            self.blockform[1][0] = mat.blockform[0][1].T
            self.blockform[1][1] = mat.blockform[1][1].T
            self.blockmat = BlockMatrix(self.blockform)
    
    
        