from sympy import MatAdd, MatMul, Determinant, Rational, Identity, ln, pi, Add, ZeroMatrix, srepr
from sympy.core.decorators import call_highest_priority

import copy

from symgp.utils import utils
from symgp.superexpressions import SuperMatAdd, SuperMatMul, SuperMatSymbol, SuperMatInverse, SuperMatTranspose


class MVG(object):
    
    _op_priority = 15.0 # Used for multiplication operations between this object and a general GP object
    DEFAULT_PREFIX = 'p' # This is the class default prefix for the MVG name. Can be changed in the constructor.
    
    def __init__(self, variables, mean=None, cov=None, logZ=None, n0=None, n1=None, n2=None, cond_vars=[], prefix='p'):
        """
            Creates an MVG (MultiVariate Gaussian) object.
        
            It is responsible for setting up the moments and natural parameters as appropriate
        
            Args:
                - 'variables' - The random variables of the MVG.
                - 'mean', 'cov', 'logZ' - The expanded/blockform expressions for the mean, covariance and log-normalising constant expressions.
                                            If logZ isn't specified, it is automatically created.
                - 'n0', 'n1', 'n2' - The parameters of the MVG in natural parameter form.
                - 'cond_vars' - The variables this MVG is conditioned on.
                - 'prefix' - Changes the prefix of a distribution. Default is 'p'. Must be in LaTeX suitable form.
        """
        
        self.variables = list(variables)
        self.conditioned = (len(cond_vars) > 0)
        self.cond_vars = cond_vars
        self.shape = sum([v.shape[0] for v in self.variables])
        
        # Create distribution name
        self.prefix = prefix
        self.name = prefix+'('+','.join([v.name for v in self.variables])
        if len(cond_vars) > 0:
            self.name += '|'+','.join([v.name for v in self.cond_vars])
        self.name += ')'
        
        # Either or both moments and natural_params can be set.
        if ((mean is not None and cov is not None) and
            (n0 is None and n1 is None and n2 is None)):
            self.mean = mean
            if isinstance(cov, list):
                assert(isinstance(mean, list))
                self.mean = SuperMatSymbol(self.shape, 1, mat_type='mean', dep_vars=self.variables, cond_vars=self.cond_vars, blockform=mean)
                self.covar = SuperMatSymbol(self.shape, self.shape, mat_type='covar', dep_vars=self.variables, cond_vars=self.cond_vars, blockform=cov)
            elif cov is not None:
                assert(mean is not None)
                self.mean = SuperMatSymbol(self.shape, 1, mat_type='mean', dep_vars=self.variables, cond_vars=self.cond_vars, expanded=mean)
                self.covar = SuperMatSymbol(self.shape, self.shape, mat_type='covar', dep_vars=self.variables, cond_vars=self.cond_vars, expanded=cov)
            else:
                self.mean = SuperMatSymbol(self.shape, 1, mat_type='mean', dep_vars=self.variables, cond_vars=self.cond_vars)
                self.covar = SuperMatSymbol(self.shape, self.shape, mat_type='covar', dep_vars=self.variables, cond_vars=self.cond_vars)
                
            self.logZ = utils.get_Z(self.covar) if logZ is None else logZ
            
            self.n_0 = None
            #self.n_0 = SuperMatAdd(SuperMatMul(Rational(1,2),self.mean.T,self.covar.I,self.mean), SuperMatMul(Rational(1,2),ln(Determinant(self.covar)),Identity(1)))
            #if not isinstance(self.n_0, SuperMatSymbol):
            #    self.n_0 = SuperMatSymbol(1, 1, self.name+'_n_0','other', expanded=self.n_0)
             
            self.n_1 = None   
            #self.n_1 = self.covar.I*self.mean
            #if not isinstance(self.n_1, SuperMatSymbol):
            #    self.n_1 = SuperMatSymbol(self.covar.shape[0], 1, self.name+'_n_1','other', expanded=self.n_1)
            
            self.n_2 = None    
            #self.n_2 = SuperMatMul(-Rational(1,2),self.covar.I)
            #if not isinstance(self.n_2, SuperMatSymbol):
            #    self.n_2 = SuperMatSymbol(self.covar.shape[0], self.covar.shape[1], self.name+'_n_2','other', expanded=self.n_2)
                
        elif ((n0 is not None and n1 is not None and n2 is not None) and
              (mean is None and cov is None)): # TODO: Change these expressions as we have done above
            self.n_0 = n0
            self.n_1 = n1
            self.n_2 = n2
            
            self.mean = SuperMatMul(-Rational(1,2),self.n_2.I,self.n_1)
            if not isinstance(self.n_0, SuperMatSymbol):
                self.mean = SuperMatSymbol(self.mean.shape[0],  self.mean.shape[1], mat_type='mean', dep_vars=self.variables, 
                                            cond_vars=self.cond_vars, expanded=self.mean)
                
            self.covar = SuperMatMul(-Rational(1,2),Identity(self.n_2.shape[0]),self.n_2.I)
            if not isinstance(self.n_0, SuperMatSymbol):
                self.covar = SuperMatSymbol(self.covar.shape[0],  self.covar.shape[1], mat_type='covar', dep_vars=self.variables,
                                              cond_vars=self.cond_vars, expanded=self.covar)
                
            self.logZ = SuperMatMul(-self.n_1.shape[0]/2,ln(2*pi)) - SuperMatMul(Rational(1,2),ln(Determinant(self.covar)))
            if not isinstance(self.n_0, SuperMatSymbol):
                self.logZ = SuperMatSymbol(1, 1, self.name+'_logZ','other', expanded=self.logZ)
                
        elif ((n0 is not None and n1 is not None and n2 is not None) and
              (mean is not None and cov is not None)):
            raise Exception("Either specify moments or natural params")
        else:
            raise Exception("No parameters were specified")
    
    def __repr__(self):
        return self.name
        
    def __str__(self):
        return self.name
    
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """
            Operation for multiplying MVG objects.
        
            Split into 3 cases:
                - Both MVGs share same set of variables: p(x|.)*p(x|.)
                - MVGs overlap in variables: p(x,y|.)*p(y,z|.)  
                - Conditional case: p(x|y,.)*p(y,.|z,.)  p(x|y)p(y,w|z) = p(x|y)p(y|w,z)p(w|z) = p(x,y,w|z)
        
            Returns an MVG.
        """
        
        variables1 = self.variables
        variables2 = other.variables
        svariables1 = set(variables1)
        svariables2 = set(variables2)
        
        
        # The type of multiplications we need to do depends on the variables of both MVGs
        # Case 1: Variables are same in both MVGs
        if svariables1 & svariables2 == svariables1:  
            new_conditioned_vars = list(set(self.cond_vars).union(set(other.cond_vars)))
            m = self.covar.shape[0]
            
            if len(variables1) == 1:
                if self.covar.expanded is not None and other.covar.expanded is not None:
                    self_covar = self.covar#.expanded
                    other_covar = other.covar#.expanded
                else:
                    self_covar = self.covar
                    other_covar = other.covar
                
                if self.mean.expanded is not None and other.mean.expanded is not None:
                    self_mean = self.mean.expanded
                    other_mean = other.mean.expanded
                else:
                    self_mean = self.mean
                    other_mean = other.mean
                    
                new_covar = (self_covar.I + other_covar.I).I
                new_mean = new_covar*(self_covar.I*self_mean + other_covar.I*other_mean)
                    
            else:
                # Check blockforms for both self and other exist:
                if (self.covar.blockform is not None or other.covar.blockform is not None or
                    self.mean.blockform is not None or other.mean.blockform is not None):
                    raise Exception("The two MVGs must have blockforms as they have 2 or more variables")
                
                # TO-DO: Make sure variables, means and covars are aligned
                new_variables = self.variables
                """if new_variables != other.variables:
                    new_variables_keys = dict(zip(new_variables,range(len(new_variables))))
                    other.variables = sorted(other.variables, key=lambda m: new_variables_keys[m])
                    other.mean.blockform = other.mean.rearrange(new_variables_keys)
                    other.covar.blockform = other.covar.rearrange(new_variables_keys)"""
                
                # Matrix multiply blockforms of means and covariances  
                new_covar_blockform = utils.matinv(utils.matadd(utils.matinv(self.covar.blockform),utils.matinv(other.covar.blockform)))
                
                # Turn block matrices into symbols
                for i in range(len(new_covar_blockform)):
                    for j in range(len(new_covar_blockform[0])):
                        var_i, var_j = new_variables[i], new_variables[j]
                        #print("new_")
                        new_covar_blockform[i][j] = SuperMatSymbol(var_i.shape[0], var_j.shape[0], mat_type='covar',dep_vars=[var_i, var_j], 
                                                                        expanded=new_covar_blockform[i][j].doit())
                            
                new_mean_blockform = utils.matmul(new_covar_blockform,utils.matadd(utils.matmul(utils.matinv(self.covar.blockform),self.mean.blockform),
                                                                                   utils.matmul(utils.matinv(other.covar.blockform),other.mean.blockform)))
                                                                                                       
                new_covar = new_covar_blockform
                new_mean = new_mean_blockform
                                                                                                      
            
            # Choose new class prefix
            new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
            
            return MVG(variables1, mean=new_mean, cov=new_covar, cond_vars=new_conditioned_vars, prefix=new_prefix)
        else:
            variables_keys = dict(zip(variables1,range(len(variables1))))   # Used to order the variables
            # Add the distinct variables from other.variables
            for v in variables2:
                if v not in variables_keys:
                    variables_keys[v] = len(variables_keys)+1  
                         
            # If A={self.variables}, B={other.variables}, comp1 = B\A, comp2 = A\B, joint12 = A intersection B
            # We sort the lists so as to assure the mean and covar blockforms match                 
            comp1 = sorted(list(svariables1 - (svariables1 & svariables2)), key=lambda m: variables_keys[m])
            comp2 = sorted(list(svariables2 - (svariables1 & svariables2)), key=lambda m: variables_keys[m])
            joint12 = sorted(list(svariables1 & svariables2), key=lambda m: variables_keys[m])
            
            smeans1 = set(self.mean.blockform) if self.mean.blockform is not None else set([self.mean])
            smeans2 = set(other.mean.blockform) if other.mean.blockform is not None else set([other.mean])
            
            
            # Case 2: Overlap of variables - p(x,y|.)p(y,z|.)
            #           - We first condition on the non-overlapping vars: p(x,y|.) = p(y|x,.)p(x|.), p(y,z|.) = p(y|z,.)p(z|.)
            #           - We then multiply the conditionals together (p(y|x,z,.) = p(y|x,.)p(y|z,.)) and do
            #             the same for the marginals (p(x,z|.) = p(x|.)p(z|.))
            #           - Lastly, we combine the multiplied conditionals and the marginals to give p(x,y,z|.):
            #                                   p(x,y,z|.) = p(y|x,z,.)p(x,z|.)
            
            # First check that this isn't the conditioned case by checking that both the unique vars in
            # each MVG aren't conditioned on in the other MVG.
            if (len(set(self.cond_vars) & svariables2) == 0) and (len(set(other.cond_vars) & svariables1) == 0):
                
                # Form dictionary mapping variables to index in precision mat
                new_variables = comp1+joint12+comp2
                #variables_keys = dict(zip(new_variables,range(len(new_variables))))
                
                # Create conditioned variables as union of other and self
                #new_conditioned_vars = list(set(self.cond_vars).union(set(other.cond_vars)))
                #print(new_variables)
                if len(joint12) > 0:
                    # Form conditionals and marginals for both MVGs
                    self_MVG_cond = self.condition(list(comp1))
                    #"self_MVG_cond - mean: %s=%s, covar: %s=%s" % (self_MVG_cond.mean,self_MVG_cond.mean.expanded,
                    #                                                    self_MVG_cond.covar,self_MVG_cond.covar.expanded))
                    self_MVG_marg = self.marginalise(list(joint12))
                    #print("self_MVG_marg - mean: %s=%s, covar: %s=%s" % (self_MVG_marg.mean,self_MVG_marg.mean.expanded,
                    #                                                    self_MVG_marg.covar,self_MVG_marg.covar.expanded))
                    other_MVG_cond = other.condition(list(comp2))
                    #print("other_MVG_cond - mean: %s=%s, covar: %s=%s" % (other_MVG_cond.mean,other_MVG_cond.mean.expanded,
                    #                                                    other_MVG_cond.covar,other_MVG_cond.covar.expanded))
                    other_MVG_marg = other.marginalise(list(joint12))
                    #print("other_MVG_marg - mean: %s=%s, covar: %s=%s" % (other_MVG_marg.mean,other_MVG_marg.mean.expanded,
                    #                                                    other_MVG_marg.covar,other_MVG_marg.covar.expanded))
                
                    # Multiply conditionals and marginals separately
                    MVG_cond12 = self_MVG_cond*other_MVG_cond
                    #print("MVG_cond12 - mean: %s=%s, covar: %s=%s" % (MVG_cond12.mean,MVG_cond12.mean.expanded,
                    #                                                  MVG_cond12.covar,MVG_cond12.covar.expanded))
                    MVG_marg12 = self_MVG_marg*other_MVG_marg
                    #print("MVG_marg12 - mean: %s=%s, covar: %s=%s" % (MVG_marg12.mean,MVG_marg12.mean.blockform,
                    #                                                  MVG_marg12.covar,MVG_marg12.covar.blockform))
                
                    # Multiply all together
                    return MVG_cond12*MVG_marg12
                else:
                    #new_covar_blockform = [[ZeroMatrix(i.shape[0],j.shape[0]) for j in new_variables] for i in new_variables]
                    
                    S_11 = self.covar.blockform
                    if S_11 is None:
                        if not self.covar.expanded is None:
                            S_11 = [[self.covar.expanded]]
                        else:
                            S_11 = [[self.covar]]
                        S_11 = [[self.covar]]

                    S_12 = [[ZeroMatrix(i.shape[0],j.shape[0]) for j in comp2] for i in comp1]
                    S_21 = utils.mattrans(S_12)
                    S_22 = other.covar.blockform
                    if S_22 is None:
                        if not other.covar.expanded is None:
                            S_22 = [[other.covar.expanded]]
                        else:
                            S_22 = [[other.covar]]
                        S_22 = [[other.covar]]
             
                        
                    # Create new matrix by top bottom method i.e. create top half of matrix then create bottom
                    top = []
                    for row1, row2 in zip(S_11,S_12):
                        top.append(row1+row2)
        
                    bottom = []
                    for row1, row2 in zip(S_21,S_22):
                        bottom.append(row1+row2)
                
                    new_covar_blockform = top+bottom
                    
    
                    mu_1 = self.mean.blockform
                    if mu_1 is None:
                        if not self.mean.expanded is None:
                            mu_1 = [self.mean.expanded]
                        else:
                            mu_1 = [self.mean]
                    #mu_1 = [self.mean]
                            
                    mu_2 = other.mean.blockform if not other.mean.blockform is None else [other.mean.expanded]
                    if mu_2 is None:
                        if not other.mean.expanded is None:
                            mu_2 = [other.mean.expanded]
                        else:
                            mu_2 = [other.mean]
                    #mu_2 = [other.mean]
                    
                    new_mean_blockform = mu_1 + mu_2
                    
                    
                    new_conditioned_vars = list(set(self.cond_vars).union(set(other.cond_vars)))
                    
                    # Choose new class prefix
                    new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
                    
                    return MVG(new_variables, mean=new_mean_blockform, cov=new_covar_blockform, cond_vars=new_conditioned_vars, prefix=new_prefix)
                    
                
                """# Form list of new mean vector
                new_mean = list(smeans1 - (smeans1 & smeans2))+list(smeans1 & smeans2)+list(smeans2 - (smeans1 & smeans2))
                new_mean = sorted(new_mean, key=lambda m: variables_keys[list(m.variables_dim1.keys())[0]])  #Sort based on the variables corresponding to mean
                
                # TO-DO: Create a SuperMatSymbol for ZeroMatrix
                inv_new_covar_blockform = [[ZeroMatrix(i.shape[0],j.shape[0]) for j in new_variables] for i in new_variables]
                
                # Add entries of first inverse covar matrix then apply approp shift to start second
                self_covar_I = self.covar.I
                for i in range(len(variables1)):
                    for j in range(len(variables1)):
                        var_i, var_j = variables1[i], variables1[j]
                        if self.covar.blockform is not None:
                            inv_new_covar_blockform[variables_keys[var_i]][variables_keys[var_j]] += self_covar_I.blockform[i][j] 
                        else: # for case len(variables1) == 1
                            inv_new_covar_blockform[variables_keys[var_i]][variables_keys[var_j]] += self_covar_I  

                other_covar_I = other.covar.I
                for i in range(len(variables2)):
                    for j in range(len(variables2)):
                        var_i, var_j = variables2[i], variables2[j]
                        if other.covar.blockform is not None:
                            inv_new_covar_blockform[variables_keys[var_i]][variables_keys[var_j]] += other_covar_I.blockform[i][j]
                        else: # for case len(variables2) == 1
                            inv_new_covar_blockform[variables_keys[var_i]][variables_keys[var_j]] += other_covar_I # for case len(variables2) == 1
                
                
                # Create new moments
                new_shape = sum([var.shape[0] for var in new_mean])
                inv_new_covar = SuperMatSymbol(new_shape,  new_shape, mat_type='invcovar', dep_vars=new_variables, cond_vars=new_conditioned_vars, blockform=inv_new_covar_blockform)
                new_covar_blockform = inv_new_covar.I.blockform
                new_covar = SuperMatSymbol(new_shape,  new_shape, mat_type='covar', dep_vars=new_variables, cond_vars=new_conditioned_vars, blockform=new_covar_blockform)
                
                new_mean = SuperMatSymbol(new_shape,  1, mat_type='mean', dep_vars=new_variables, cond_vars=new_conditioned_vars, blockform=new_mean)
                
                new_logZ = -(new_shape/2)*ln(2*pi) - Rational(1,2)*ln(Determinant(new_covar))
                
                new_moments.extend([new_mean, new_covar, new_logZ])
                
                return MVG(new_variables, moments=new_moments, cond_vars=new_conditioned_vars)"""
            else: # Case 3: Conditional case
            
                # Determine the conditional and the marginal
                if (len(set(self.cond_vars) & svariables2) != 0) and (len(set(other.cond_vars) & svariables1) == 0):
                    conditional = self
                    marginal = other
                elif (len(set(other.cond_vars) & svariables1) != 0) and (len(set(self.cond_vars) & svariables2) == 0):
                    conditional = other
                    marginal = self
                else:
                    raise NotImplementedError("This conditional case isn't supported")
                
                # Get total shapes of the conditional's variables       
                conditional_vars_shape = sum([v.shape[0] for v in conditional.variables])
                
                # Marginal and conditional conditioned-on variables
                conditional_cond_vars = set(conditional.cond_vars)
                marginal_vars = set(marginal.variables)
                
                # The conditioned variables
                cond_vars = conditional_cond_vars & marginal_vars
                
                # This situation solves the following case:
                #
                #                  p(a|b)*p(b,c)
                #
                # where a, b and c are sets of variables. We solve it using two steps:
                #
                #           1. Condition: p(b,c) = p(c|b)p(b) 
                #           2. Multiply: p(a,c|b) = p(a|b)p(c|b) then p(a,c,b) = p(a,c|b)p(b)
                #              
                
                if len(cond_vars) < len(marginal_vars):
                    
                    other_marginal_vars = marginal_vars - cond_vars  # c
                    
                    cond_vars = sorted(list(cond_vars), key=lambda m: marginal.variables.index(m))
                    other_marginal_vars = sorted(list(other_marginal_vars), key=lambda m: marginal.variables.index(m))
                    
                    p_other_g_cond = marginal.condition(cond_vars)  # p(c|b)
                    p_cond = marginal.marginalise(other_marginal_vars)  # p(b)
                    
                    p_conditional_j_other = conditional*p_other_g_cond  # p(a,c|b)
                    #print("p_conditional_j_other.mean: (%s)-(%s)" % (p_conditional_j_other.mean, p_conditional_j_other.mean.blockform))
                    #print("p_conditional_j_other.covar: (%s)-(%s)" % (p_conditional_j_other.covar, p_conditional_j_other.covar.blockform))
                    
                    return p_conditional_j_other*p_cond  # p(a,c|b)p(b)
                    #raise NotImplementedError("This operation isn't implemented. Conditioned-on variables must be the same variables of marginal")
                
                cond_vars_shape = sum([v.shape[0] for v in cond_vars])
                
                # New variables shape
                new_shape = conditional_vars_shape + cond_vars_shape 
                  
                # Check for other conditioned variables
                new_conditioned = False
                new_conditioned_vars = []
                if len(cond_vars) <= len(conditional_cond_vars):
                    new_conditioned_vars = list((conditional_cond_vars - cond_vars).union(set(marginal.cond_vars)))
                    new_conditioned = True
                
                # Convert cond_vars to list and sort so as to match vars in marginal
                cond_vars = list(cond_vars)
                cond_vars = sorted(cond_vars, key=lambda m: marginal.variables.index(m))
                #print("cond_vars: ",cond_vars)
            
                # Create list of new variables and create a dict for ordering variables
                new_variables = conditional.variables+cond_vars
                new_variables_keys = dict(zip(new_variables, range(len(new_variables))))
                
                #### Get new mean and covariance matrix algorithm ####
                # As we can express the conditional mean, mu_(x|y) as:
                #             
                #              mu_(x|y) = Lambda + Omega*y
                #
                # We can find all our unknowns: S_xx, S_xy (and S_yx) and mu_x
                # from our knowns S_(x|y), Lambda, Omega and mu_(x|y) as follows:
                #
                #                S_xy*S_yy^-1*y = Omega*y -> S_xy = Omega*S_yy
                #
                #        S_(x|y) = S_xx - S_xy*S_yy^-1*S_yx -> S_xx = S_(x|y)+Omega*S_yy*Omega'
                #
                #           Lambda = mu_x - S_xy*S_yy^-1*mu_y -> mu_x = Lambda + Omega*mu_y
                
                if len(conditional.variables) == 1:
                
                    if conditional.mean.expanded is not None:
                        expanded_cond_mean = utils.expand_matexpr(conditional.mean.expanded)
                    else: # This case should never really happen
                        raise Exception("The conditional mean should have an expanded form")
                    
                    Omega, Lambda = utils.get_var_coeffs(expanded_cond_mean, cond_vars)
                    #print("Omega: ",Omega)
                    #print("Lambda: ",Lambda)
                    Omega = [Omega]  # 2-D
                    Lambda = [Lambda]  # 1-D
                else:
                    
                    if conditional.mean.blockform is not None:
                        # For each conditional variable extract the coefficients of cond_vars (Omega) and concatenate 
                        # them all into a list of lists to form the appropriate blockform for 'comp_marg_vars'
                        # The remainder is collected in Lambda
                        Omega, Lambda = [], []                                             
                        for i in range(len(conditional.variables)):
                            #print(conditional.variables[i])
                            if (not (isinstance(conditional.mean.blockform[i], SuperMatSymbol) or 
                                isinstance(conditional.mean.blockform[i], SuperMatInverse) or
                                isinstance(conditional.mean.blockform[i], SuperMatTranspose)) or 
                                (conditional.mean.blockform[i].expanded is None)):
                                expanded_cond_mean = conditional.mean.blockform[i]  
                            else: 
                                expanded_cond_mean = conditional.mean.blockform[i].expanded
                            expanded_cond_mean = utils.expand_matexpr(expanded_cond_mean)
                            omega_i, lambda_i = utils.get_var_coeffs(expanded_cond_mean, cond_vars)
                            Omega.append(omega_i)
                            Lambda.append(lambda_i)
                    else:
                        raise Exception("The conditional mean should have a blockform")
                 
                                                
                # Create new covariance matrix and new mean
                if marginal.mean.blockform is None:
                    if marginal.mean.expanded is not None:
                        marginal_mean = [marginal.mean.expanded]
                    else:
                        marginal_mean = [marginal.mean]
                else:
                    marginal_mean = marginal.mean.blockform
                
                #print("Lambda: ", Lambda)
                #print("Omega: ",Omega)
                #print("marginal_mean: ",marginal_mean)
                #print("utils.matmul(Omega,marginal_mean): ",utils.matmul(Omega,marginal_mean))
                new_mean_blockform = utils.matadd(Lambda,utils.matmul(Omega,marginal_mean))+marginal_mean
                new_mean = [m.doit() for m in new_mean_blockform]
                #new_mean_blockform = [m[0] for m in new_mean_blockform]
                
                
                # Create new covar blockform
                if conditional.covar.blockform is None:
                    if conditional.covar.expanded is not None:
                        conditional_covar = [[conditional.covar.expanded]]
                    else:
                        raise Exception("Conditional should have an expanded or blockform")
                else:
                    conditional_covar = conditional.covar.blockform
                
                if marginal.covar.blockform is None:
                    if marginal.covar.expanded is not None:
                        marginal_covar = [[marginal.covar.expanded]]
                    else:
                        marginal_covar = [[marginal.covar]]
                else:
                    marginal_covar = marginal.covar.blockform
                
                """print("utils.matmul(Omega,marginal_covar): ",utils.matmul(Omega,marginal_covar))
                print("utils.matmul(utils.matmul(Omega,marginal_covar),utils.mattrans(Omega)): ",utils.matmul(utils.matmul(Omega,marginal_covar),utils.mattrans(Omega)))
                print("conditional_covar: ",conditional_covar)
                print("marginal_covar: ",marginal_covar)"""
                #print("Omega: ",srepr(Omega[0][0]))
                S_11 = utils.matadd(conditional_covar,utils.matmul(utils.matmul(Omega,marginal_covar),utils.mattrans(Omega)))
                #print("S_11: ",S_11)
                S_12 = utils.matmul(Omega,marginal_covar)
                #print("S_12: ",S_12)
                S_21 = utils.mattrans(S_12)
                #print("S_21: ",S_21)
                S_22 = marginal_covar
                #print("S_22: ",S_22)
                
                # Create new matrix by top bottom method i.e. create top half of matrix then create bottom
                top = []
                for row1, row2 in zip(S_11,S_12):
                    top.append(row1+row2)
        
                bottom = []
                for row1, row2 in zip(S_21,S_22):
                    bottom.append(row1+row2)
                
                new_covar_blockform = top+bottom
                
                # Turn block matrices into symbols
                for i in range(len(new_covar_blockform)):
                    for j in range(len(new_covar_blockform[0])):
                        var_i, var_j = new_variables[i], new_variables[j]
                        #print("new_")
                        new_covar_blockform[i][j] = SuperMatSymbol(var_i.shape[0], var_j.shape[0], mat_type='covar',dep_vars=[var_i, var_j], 
                                                                        expanded=new_covar_blockform[i][j].doit())
                        #print("new_covar_blockform[%s][%s]: %s" % (i,j,new_covar_blockform[i][j].expanded))
                                                            
                new_covar = new_covar_blockform
                                        
                
                #print("Cond multiplication done")
                
                # Choose new class prefix
                new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
                
                return MVG(new_variables, mean=new_mean, cov=new_covar, cond_vars=new_conditioned_vars, prefix=new_prefix)        
    
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.__mul__(other)        
    
    def condition(self, x, use_blockform=True):
        """
            Condition operation.

            We condition on the variables given in x.
        """
        
        if len(self.variables) == 1:
            raise Exception("Can't condition as this MVG has only one variable")
        else:
            
            # Convert to list
            x = list(x)
            
            # Assign indices to variables so as to obtain and preserve orderings
            variables_dict = {}
            for i in range(len(self.variables)):
                variables_dict[self.variables[i]] = i
            
            # Get partitioned covariance matrix
            cond_vars_indices = [variables_dict[v] for v in x]
            P, Q, R, S = self.covar.partition([cond_vars_indices]*2)
            
            if use_blockform:
                P, Q = P.blockform, Q.blockform
                R, S = R.blockform, S.blockform
            else:
                P, Q = [[P]], [[Q]]
                R, S = [[R]], [[S]]
            
            # Marginal (Conditioned-on) variables
            marg_vars = [v for v in x]
            marg_shape = sum([v.shape[0] for v in x])
                        
            # Conditional variables
            cond_vars = [v for v in self.variables if v not in x]
            cond_shape = sum([v.shape[0] for v in cond_vars])
            
            # Create conditioned MVG parameters
            if use_blockform:
                v_marg_vars = marg_vars
                old_mean = [self.mean.blockform[i] for i in range(len(self.variables)) if self.variables[i] not in x]
                marg_mean = [self.mean.blockform[i] for i in range(len(self.variables)) if self.variables[i] in x]
            else:
                v_marg_vars = SuperMatSymbol(marg_shape,1,'v_('+','.join([v.name for v in marg_vars])+')',mat_type='var',dep_vars=marg_vars,blockform=marg_vars)
                v_marg_vars = [v_marg_vars]
                old_mean, marg_mean = self.mean.partition(cond_vars_indices)
                old_mean = [old_mean]
                marg_mean = [marg_mean]
             
            new_mean_blockform = utils.matadd(old_mean,utils.matmul(Q,utils.matmul(utils.matinv(S),
                                            utils.matadd(v_marg_vars,utils.matmul(-1,marg_mean)))))
            new_mean_blockform = [m.doit() for m in new_mean_blockform]
            
            new_covar_blockform = utils.matadd(P,utils.matmul(utils.matmul(utils.matmul(-1,Q),utils.matinv(S)),R))
            new_covar_blockform = [[c.doit() for c in r] for r in new_covar_blockform]
                
                
            new_conditioned_vars = self.cond_vars+v_marg_vars
            
            if len(new_mean_blockform) == 1:
                new_cond_mean = new_mean_blockform[0].doit()
                new_cond_covar =new_covar_blockform[0][0].doit()
            else:
                new_cond_mean = new_mean_blockform
                new_cond_covar = new_covar_blockform
            
            
            return MVG(cond_vars, mean=new_cond_mean, cov=new_cond_covar, cond_vars=new_conditioned_vars, prefix=self.prefix)
    
    def marginalise(self, x, debug=False):
        
        # Check that this MVG has 2 or more vars
        if len(self.variables) < 2:
            raise Exception("Can only marginalise when MVG is a distribution over 2 or more variables.")
        
        x = list(x)
        
        x_loc = [-1]*len(x)
        for j in range(len(x)):
            for i in range(len(self.variables)):
                if self.variables[i] == x[j]:
                    x_loc[j] = i
        
        x_loc = sorted(x_loc, reverse=True)
        
        if debug:
            print("x: ",x)
            print("x_loc: ",x_loc)
        
        if any([i==-1 for i in x_loc]):
            raise Exception("%s is not a variable of this MVG" % x)
                
        new_blockform = []
        for row in self.covar.blockform:
            new_blockform.append(list(row)) 

        if debug:
            print("new_blockform: ",new_blockform)
            
        # Delete down the rows then across the cols
        num_rows, num_cols = len(new_blockform), len(new_blockform[0])
        
        # Delete cols
        new_blockform = [[r[idx] for idx in range(num_cols) if idx not in x_loc] for r in new_blockform]
        
        if debug:
            print("new_blockform: ", new_blockform)
            
        # Delete rows
        new_blockform = [row for i, row in enumerate(new_blockform) if i not in x_loc]
        
        #for i in range(len(new_blockform)):
        #    for j in range(len(new_blockform[0])):
        #        new_blockform[i][j] = new_blockform[i][j].expanded
        
        # New variables
        new_variables = [v for v in self.variables if v not in x]
        shape = self.covar.shape[0] - sum([var.shape[0] for var in x])
        
        if debug:
            print("new_variables: ",new_variables)
            print("shape: ",shape)
            
        # Create new moments
        if len(new_variables) > 1:
            new_covar = new_blockform
            new_mean = [self.mean.blockform[i].doit() for i in range(len(self.variables)) if self.variables[i] not in x]
        else:
            new_covar = new_blockform[0][0]
            new_mean = [self.mean.blockform[i].doit() for i in range(len(self.variables)) if self.variables[i] not in x][0]
            
       
        if debug:
            print("new_covar: ",new_covar)
            print("new_mean: ",new_mean)

        return MVG(new_variables, mean=new_mean, cov=new_covar, cond_vars=self.cond_vars, prefix=self.prefix)
    
    def changeName(self, name):
        """
            Change the name of the MVG
        """
        
        self.name = name
    
    def Elogp(self, p, to_full_expr=False):
        """
            Calculates the expectation of log p w.r.t to this distribution where p is also
            a distribution
        
            Args:
                p - An MVG object
                to_full_expr - Set to True to indicate that you want to use the full expression
                               for the mean and covariance of this MVG
        
            Returns:
                res - The calculation of this expression
        """
        a, A = self.mean, self.covar
        b, B = p.mean, p.covar
        
        if to_full_expr:
            a = utils.expand_to_fullexpr(a)
            A = utils.expand_to_fullexpr(A)
            b = utils.expand_to_fullexpr(b)
            B = utils.expand_to_fullexpr(B)
                
        return -Rational(1,2)*(ln(Determinant(2*pi*B)) + Trace(B.I*A) + (a-b).T*B.I*(a-b))
        
        
                 
        
            
                
    
        
        
                
                
                
                
            
        
        
                 
        
        