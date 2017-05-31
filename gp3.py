from sympy import Identity, ZeroMatrix
from sympy.core.decorators import call_highest_priority

import utils
from kernels.kernel import Kernel, KernelMatrix
import SuperMatExpr as SME
import MVG


class GP3(object):
    
    _op_priority = 20.0
    
    def __init__(self, K, m, inducing_prior=None, train_cond=None, likelihood=None):
        """
            Creates a GP (Gaussian Process) object.
        
            Args:
                variables - The variables over which to place the GP.
                d - The number of dimensions for the data
                K - The GP Kernel. This is a Kernel object
                m - The GP mean function.
              (Optional)
                X - Input locations. Must be a (N,D) SuperMatSymbol where each col is an input location
                y - Observations at locations given by x. Must be an (N,) or (N,1) SuperMatSymbol
                sd - Standard deviation. Must be a Symbol.  
        """
        
        self.variables = variables
        self.d = d
        self.K = K
        self.m = m
        self.X = X
        self.name = 'p('+','.join([v.name for v in self.variables])
        if len(X) > 0:
            self.name += '|'+','.join([v.name for v in self.X])
        self.name += ')'
        #self.y = y
        #self.sd = sd
        #if X is not None:
        #    self.K_XX = K(X, X) + sd**2*Identity(X.shape[0]) if X is not None else None
    
    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name
    
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
        
        m_y = SME.SuperMatSymbol(n,1,name='m('+X.name+')',mat_type='other',expanded=ZeroMatrix(n, 1))
        cov_yy = SME.SuperMatSymbol(n,n,name='K('+','.join([X.name, X.name])+')',mat_type='other',expanded=self.K(X, X))
        
        return MVG.MVG([y], moments=[m_y, cov_yy, utils.getZ(cov_yy)])
    
    def condition(self, L):
        """
            Condition the GP on the given data.
        
            Args:
                X, y - The design matrix symbol and corresponding observations. These are lists of SuperMatSymbols where each element
                       is conditioned on a variable in self.variables. They should correspond to the training set latent variables in 
                       self.variables. Each element is of shape (n,d) and (n,1), respectively, where n is number of data points and d 
                       is the dimension. If there is only one element, a list doesn't have to be used.
                   S - Covariance matrix of p(y|f). This is a list of lists where each element corresponds to the covariance between 
                       the respective variables in y. If there is only one element in y a list doesn't have to be used.
                   m - Symbol representing shape of test set.
        
            Returns:
                An MVG over f and f_c where f are the latent function values corresponding to y and f_c are all the other
                latent values that aren't f.
                
        """
        y = L.variables
        #X = L.conditioned_vars
        # Find which variables in self.variables, y is conditioned on.
        if isinstance(y,list):
            #X = [x for x in self.X if x in L.conditioned_vars]
            #prior_vars, prior_shapes, prior_cond_vars = [], [], []
            """for o in y:
                o_cond_var = [v for v in o.cond_vars if any([l==v for l in self.variables])]
                if len(o_cond_var) > 1:
                    raise Exception("The observations should have only one latent vector")
                else:
                    o_cond_var = o_cond_var[0]
                
                prior_vars.append(o_cond_var)
                prior_shapes.append(o_cond_var.shape[0])
                prior_cond_vars += o_cond_var.cond_vars"""
            
            # Add non training set latent variables
            #test_vars = [v for v in self.variables if not v in prior_vars]
            prior_vars = self.variables
            
            prior_shapes = [v.shape[0] for v in prior_vars]
            """if len(test_vars) > 0:
                prior_shapes += [v.shape[0] for v in test_vars]
                test_cond_vars = []
                prior_cond_vars = prior_cond_vars + test_cond_vars
            prior_cond_vars = list(set(prior_cond_vars))"""
            prior_cond_vars = []
            for v in prior_vars:
                prior_cond_vars += v.cond_vars
            print(prior_cond_vars)
            #print(X)
            # Create MVG prior
            m_prior = SME.SuperMatSymbol(sum(prior_shapes),1,mat_type='mean',dependent_vars=prior_vars,cond_vars=prior_cond_vars,
                                            blockform=[ZeroMatrix(n,1) for n in prior_shapes]) # TO-DO: Replace with mean function here
            cov_prior = SME.SuperMatSymbol(sum(prior_shapes),sum(prior_shapes),mat_type='covar',dependent_vars=prior_vars,cond_vars=prior_cond_vars,
                                            blockform=[[self.K(x1,x2) for x2 in self.X] for x1 in self.X])
            prior = MVG.MVG(prior_vars, moments=[m_prior, cov_prior, utils.getZ(cov_prior)],conditioned_vars=self.X)
            
            # Create joint then condition
            joint = L*prior
            posterior = joint.condition(y)
            
            return posterior#, joint
        else:
            raise Exception("Likelihood variables should be in list")
            """y_dep_var = y.dependent_vars[]
            for v in self.variables:
            
        
        n, d = X.shape 
        
        # Variables
        f = SME.SuperMatSymbol(n,1,'f('+X.name+')',mat_type='var',cond_vars=[X])
        f_c = SME.SuperMatSymbol(m,1,f.name+'_c',mat_type='var',cond_vars=[X])
    
        # Moments
        m_f = self.m()
        
        # MVG"""
        