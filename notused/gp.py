from sympy import Identity, ZeroMatrix
from sympy.core.decorators import call_highest_priority

import utils
from kernels.kernel import Kernel, KernelMatrix
import SuperMatExpr as SME
import MVG


class GP(object):
    
    _op_priority = 20.0
    
    def __init__(self, d, K, m=None, X=None, y=None, sd=0):
        """
            Creates a GP (Gaussian Process) object.
        
            Args:
                d - The number of dimensions for the data
                K - The GP Kernel. This is a Kernel object
                m - The GP mean function.
              (Optional)
                X - Input locations. Must be a (N,D) SuperMatSymbol where each col is an input location
                y - Observations at locations given by x. Must be an (N,) or (N,1) SuperMatSymbol
                sd - Standard deviation. Must be a Symbol.  
        """
        
        self.d = d
        self.K = K
        self.m = m
        self.X = X
        self.y = y
        #self.sd = sd
        #if X is not None:
        #    self.K_XX = K(X, X) + sd**2*Identity(X.shape[0]) if X is not None else None
    
    def project(self, X, y):
        """
            Project the GP onto a finite set of samples.
        
            Args:
                X - The input locations over which to form a multivariate Gaussian distribution. Must be shape
                     (n, d) where d is the dimension of each vector and n is the number of vectors.
                y - The values corresponding to each of the points in X. Must be a (n, 1) SuperMatSymbol.
        
            Returns:
                An MVG object over y
        """
        
        n = X.shape[0]
        
        m_y = SME.SuperMatSymbol(n,1,'m('+X.name+')',mat_type='other',expanded=ZeroMatrix(n, 1))
        cov_yy = SME.SuperMatSymbol(n,n,'K('+','.join([X.name, X.name])+')',mat_type='other',expanded=self.K(X, X))
        
        return MVG.MVG([y], moments=[m_y, cov_yy, utils.getDet(cov_yy)])
    
    """def predict(self, Xs):
        
            Create the predictive distribution over ys
        
            Args:
                Xs - The input locations for which to create the predictive distribution. Must be shape
                     (d, n) where d is the dimension of each vector and n is the number of vectors.
        
            Returns:
                The MVG object of the predictive distribution over ys
        
        
        n = Xs.shape[1]
        
        K_ysy = self.K(xs, X) # K(xs.T,X)
        
        ys = SME.SuperMatSymbol('y_(*)',n,1,'var')
        m_ys = K_ysy.T*self.K_yy.I*y  # Predictive mean
        K_ysys = self.K(xs,xs) - K_ysy*self.K_yy.I*K_ysy.T  # Predictive covariance
        
        return MVG.MVG([ys], moments=[m_ys, K_ysys, utils.getDet(K_ysys)])"""
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """
            Multiplication of GP with a GP or an MVG
        """
        
        # 2 cases: GP*MVG, GP*GP
        if isinstance(other, MVG.MVG):
            y = other.variables  # Variables in the MVG
            shapes = [v.shape[0] for v in y]    # Shapes of each variable in y
            locs = [[loc] for loc in other.conditioned_vars if ((loc.shape[0] in shapes) and (loc.shape[1] == self.d))]   # locations corresponding to y
            
            if len(locs) == 0:
                raise Error("No data matrices in MVG. MVG should be conditioned on the data matrices")
            
            n = sum(shapes)
            
            # Concatenate variables into one SuperMatSymbol if there are more than 1  
            if len(y) > 1:
                X = SME.SuperMatSymbol(n,self.d,'X_('+','.join([l[0].name for l in locs])+')',mat_type='var',blockform=locs)
                y = SME.SuperMatSymbol(n,1,'y('+''.join([v.name for v in y])+')',mat_type='var',blockform=y)
            else:
                X = locs[0][0]
                y = y[0]
            
            other_covar = other.covar if other.covar.expanded is None else other.covar.expanded    
            K_yy = SME.SuperMatSymbol(n, n,'K_('+','.join([X.name, X.name])+')', mat_type='other',dependent_vars=[X],expanded=(self.K(X,X) + other_covar).I)
            new_K = self.K - Kernel([self.K, self.K], 'mul', K_yy)
                
            return GP(self.d, new_K, X=X, y=y)
        elif isinstance(other, GP):
            raise NotImplementedError   
        else:
            raise NotImplementedError
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        print(other)
        return self.__mul__(other)
    
    
        
        
         
         
    