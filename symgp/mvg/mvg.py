from typing import List, Union

from sympy import MatAdd, MatMul, Determinant, Rational, \
    Identity, ln, pi, Add, ZeroMatrix, srepr, MatrixExpr
from sympy.core.decorators import call_highest_priority

from symgp.superexpressions import Variable, CompositeVariable, Mean, Covariance
from symgp.utils import utils
from symgp.superexpressions import SuperMatAdd, SuperMatMul, SuperMatSymbol, \
    SuperMatInverse, SuperMatTranspose

Vector_t = List[MatrixExpr]
Matrix_t = List[List[MatrixExpr]]

class MVG(object):

    # TODO: Maybe remove this?
    # Used for multiplication operations between this object and a general GP object
    _op_priority = 15.0

    # This is the class default prefix for the MVG name. Can be changed in the constructor.
    DEFAULT_PREFIX = 'p'
    
    def __init__(self, variables: List[Union[Variable, CompositeVariable]],
                 mean: Union[MatrixExpr, Vector_t]=None, cov: Union[MatrixExpr, Matrix_t]=None,
                 cond_vars: List[Union[Variable, CompositeVariable]]=None, prefix: str='p'):
        """
        Creates an MVG (MultiVariate Gaussian) object.

        This represents a multivariate Gaussian distribution that we can perform operations on to
        obtain new MVGs. (See ``condition``, ``__mul__``/``multiply``, ``marginalise`` below.)
        
        It is responsible for setting up the moments (0th - 2nd) of the distribution as appropriate.
        
        :param variables: The random variables of the MVG.
        :param mean: The optional expanded or blockform expressions for the mean. If this is a
        blockform, its length has to match that of ``variables``. If left as default, we create a
        default parameter based on ``variables`` and ``cond_vars`` with no blockform/expanded.
        :param cov: The optional expanded or blockform expressions for the covariance. If this is a
        blockform, it must be square with dimensions matching the length of ``variables``. If left
        as default, we create a default parameter based on ``variables`` and ``cond_vars`` with no
        blockform/expanded.
        :param cond_vars: The variables this MVG is conditioned on.
        :param prefix: Changes the prefix of a distribution. Default is 'p'. Must be in LaTeX
        format if any special symbols are to be used.
        """
        
        self.variables = list(variables)
        self.shape = sum([v.shape[0] for v in self.variables])
        
        if isinstance(mean, Vector_t):
            if cond_vars:
                self.cond_vars = cond_vars
            else:
                self.cond_vars = []
                for e in mean:
                    self.cond_vars.extend(utils.get_variables(e))
                self.cond_vars = list(set(self.cond_vars))
        elif isinstance(mean, MatrixExpr):
            self.cond_vars = cond_vars if not mean is None else utils.get_variables(mean)
        else:
            raise KeyError("Invalid type for ``mean``. Must be: {}. Provided: {}".
                           format(Vector_t, type(mean)))

        # Create distribution name
        self.prefix = prefix
        self.name = prefix+'('+ utils.create_distr_name(self.variables, self.cond_vars) + ')'
        self.cond_vars = [] if self.cond_vars is None else self.cond_vars
        
        # Must set both moments
        if (mean is not None and cov is not None):
            self.mean = Mean(self.variables, cond_vars=cond_vars, full_expr=mean)
            self.covar = Covariance(self.variables, cond_vars=cond_vars,
                                    full_expr=cov)
        else:
            print("Creating default parameters without blockform/expanded")
            self.mean = Mean(self.variables, cond_vars=cond_vars)
            self.covar = Covariance(self.variables, cond_vars=cond_vars)
        self.logZ = utils.get_logZ(self.covar)
    
    def __repr__(self):
        return self.name
        
    def __str__(self):
        output = self.name + '\n\n'
        output += '    ' + self._create_mean_name() + '\n'
        output += '    ' + self._create_cov_name()
        return output

    def _create_mean_name(self) -> str:
        mean_name = self.mean.name
        if self.mean.blockform is not None:
            mean_name += " = " + str(self.mean.blockform)
        elif self.mean.expanded is not None:
            mean_name += " = " + str(self.mean.expanded)
        else:
            mean_name += " = " + self.mean.name
            #raise Exception("self.mean should have an expanded or blockform")

        return mean_name

    def _create_cov_name(self) -> str:
        cov_name = self.covar.name
        if self.covar.blockform is not None:
            cov_name += " = " + str(self.covar.blockform)
        elif self.covar.expanded is not None:
            cov_name += " = " + str(self.covar.expanded)
        else:
            cov_name += " = " + self.covar.name
            #raise Exception("self.mean should have an expanded or blockform")

        return cov_name

    def _mul_same_vars(self, other):
        """
        The multiplication operation for case: p(x|.)*q(x|.)
        
        Returns an MVG object
        """
        
        def get_params(self, other, param):
            if param == 'mean':
                this_param, other_param = self.mean, other.mean
            elif param == 'covar':
                this_param, other_param = self.covar, other.covar
            else:
                raise Exception("Invalid entry for param")
            
            if this_param.expanded is not None and other_param.expanded is not None:
                return this_param.expanded, other_param.expanded
            else:
                return this_param, other_param
        
        self_variables = self.variables
        
        new_conditioned_vars = list(set(self.cond_vars).union(set(other.cond_vars)))
        
        if len(self_variables) == 1:
            self_covar, other_covar = get_params(self, other, 'covar')
            self_mean, other_mean = get_params(self, other, 'mean')
                
            new_covar = (self_covar.I + other_covar.I).I
            new_mean = new_covar*(self_covar.I*self_mean + other_covar.I*other_mean)       
        else:
            # Check blockforms for both self and other exist:
            if (self.covar.blockform is not None or other.covar.blockform is not None or
                self.mean.blockform is not None or other.mean.blockform is not None):
                raise Exception("The two MVGs must have blockforms for the mean and covariance "
                                "parameters as they have 2 or more variables")
            
            # Calculate covariance
            new_covar = utils.matinv(utils.matadd(utils.matinv(self.covar.blockform),
                                                  utils.matinv(other.covar.blockform)))
            
            # Turn block matrices into symbols
            for i in range(len(new_covar)):
                for j in range(len(new_covar[0])):
                    var_i, var_j = self_variables[i], self_variables[j]
                    new_covar[i][j] = Covariance(v1=var_i, v2=var_j, cond_vars=new_conditioned_vars,
                                                 full_expr=new_covar[i][j].doit())

            # Calculate mean
            new_mean = utils.matmul(new_covar, utils.matadd(
                utils.matmul(utils.matinv(self.covar.blockform), self.mean.blockform),
                utils.matmul(utils.matinv(other.covar.blockform), other.mean.blockform)))

                                                                                                  
        # Choose new class prefix
        new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
        
        return MVG(self_variables, mean=new_mean, cov=new_covar, cond_vars=new_conditioned_vars,
                   prefix=new_prefix)
    
    def _mul_overlap_case(self, other, sets):
        """
            The multiplication operation for case: p(x,y|.)*p(y,z|.)
        
            Returns an MVG object
        """
        
        comp1, joint12, comp2 = sets
        
        # Form conditionals and marginals for both MVGs
        self_MVG_cond = self.condition(list(comp1))
        self_MVG_marg = self.marginalise(list(joint12))
        other_MVG_cond = other.condition(list(comp2))
        other_MVG_marg = other.marginalise(list(joint12))
    
        # Multiply conditionals and marginals separately
        MVG_cond12 = self_MVG_cond*other_MVG_cond
        MVG_marg12 = self_MVG_marg*other_MVG_marg
    
        # Multiply all together
        return MVG_cond12*MVG_marg12
    
    def _mul_disjoint_case(self, other, sets):
        """
            The multiplication operation for case: p(x|.)*p(y|.)
        
            Returns an MVG object
        """
        
        comp1, comp2 = sets
        
        def _get_blocks(param):
            if param == 'covar':
                S_11 = self.covar.blockform
                if S_11 is None:
                    S_11 = [[self.covar]]
                S_12 = [[ZeroMatrix(i.shape[0],j.shape[0]) for j in comp2] for i in comp1]
                S_21 = utils.mattrans(S_12)
                S_22 = other.covar.blockform
                if S_22 is None:
                    S_22 = [[other.covar]]
                
                return S_11, S_12, S_21, S_22
            elif param == 'mean':
                mu_1 = self.mean.blockform
                if mu_1 is None:
                    if not self.mean.expanded is None:
                        mu_1 = [self.mean.expanded]
                    else:
                        mu_1 = [self.mean]
                
                mu_2 = other.mean.blockform
                if mu_2 is None:
                    if not other.mean.expanded is None:
                        mu_2 = [other.mean.expanded]
                    else:
                        mu_2 = [other.mean]
                
                return mu_1, mu_2
            else:
                raise Exception("Invalid entry for param")
         
        # Covariance    
        P, Q, R, S = _get_blocks('covar')
        new_covar_blockform = utils.create_blockform(P,Q,R,S)
        
        # Mean
        mu_1, mu_2 = _get_blocks('mean')
        new_mean_blockform = mu_1 + mu_2
        
        new_conditioned_vars = list(set(self.cond_vars).union(set(other.cond_vars)))
        new_variables = comp1 + comp2
        
        # Choose new class prefix
        new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
        
        return MVG(new_variables, mean=new_mean_blockform, cov=new_covar_blockform,
                   cond_vars=new_conditioned_vars, prefix=new_prefix)
    
    def _mul_conditional_case(self, other, vars_12):
        """
        The multiplication operation for case: p(x|y,.)*p(y,.|z,.) or
        p(x|y)p(y,w|z) = p(x|y)p(y|w,z)p(w|z) = p(x,y,w|z)
        :param other: The other MVG object
        :param vars_12: The unordered sets of random variables for both MVGs
        :return: An MVG object
        """
        
        def _get_conditional_and_marginal(self, other, svars1, svars2):
            """
            Returns the conditional and marginal distributions based on the random variables and
            conditioned-on variables for both distributions.
            :param self: The MVG object calling the encapsulating function.
            :param other: The other MVG object
            :param svars1: The random variables for the self MVG.
            :param svars2: The random variables for the other MVG.
            :return: A tuple of (conditional, marginal) MVG.
            """
            if (len(set(self.cond_vars) & svars2) != 0) and (len(set(other.cond_vars) & svars1) == 0):
                return self, other
            elif (len(set(other.cond_vars) & svars1) != 0) and (len(set(self.cond_vars) & svars2) == 0):
                return other, self
            else:
                raise NotImplementedError("This conditional case isn't supported")

        svariables1, svariables2 = vars_12    # Sets of variables
        
        conditional, marginal = _get_conditional_and_marginal(self, other, svariables1, svariables2)

        # Marginal and conditional conditioned-on variables
        conditional_cond_vars = set(conditional.cond_vars)
        marginal_vars = set(marginal.variables)
        
        # The conditioned-on variables that are also in the marginal
        cond_vars = conditional_cond_vars & marginal_vars
        

        if len(cond_vars) < len(marginal_vars):
            # This situation solves the following case:
            #
            #                  p(a|b)*p(b,c)
            #
            # where a, b and c are sets of variables. We solve it using two steps:
            #
            #           1. Condition: p(b,c) = p(c|b)p(b)
            #           2. Multiply: p(a,c|b) = p(a|b)p(c|b) then p(a,c,b) = p(a,c|b)p(b)
            #
            
            # c
            other_marginal_vars = marginal_vars - cond_vars  
            
            cond_vars = sorted(list(cond_vars), key=lambda m: marginal.variables.index(m))
            other_marginal_vars = sorted(list(other_marginal_vars),
                                         key=lambda m: marginal.variables.index(m))
            
            # p(c|b)
            p_other_g_cond = marginal.condition(cond_vars)
              
            # p(b)
            p_cond = marginal.marginalise(other_marginal_vars)  
            
            # p(a,c|b)
            p_conditional_j_other = conditional*p_other_g_cond  
            
            return p_conditional_j_other*p_cond  # p(a,c|b)p(b)
        else:
            # Check for other conditioned variables
            new_conditioned_vars = []

            if len(cond_vars) <= len(conditional_cond_vars):
                new_conditioned_vars = list((conditional_cond_vars - cond_vars).union(set(marginal.cond_vars)))

            # Convert cond_vars to list and sort so as to match vars in marginal
            cond_vars = sorted(list(cond_vars), key=lambda m: marginal.variables.index(m))
    
            # Create list of new variables and create a dict for ordering variables
            new_variables = conditional.variables + cond_vars
        
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
        
                if not conditional.mean.expanded is None:
                    expanded_cond_mean = conditional.mean.to_full_expr()
                else: # This case should never really happen
                    raise Exception("The conditional mean should have an expanded form")
                    
                Omega, Lambda = utils.get_var_coeffs(expanded_cond_mean, cond_vars)
                Omega = [Omega]  # 2-D
                Lambda = [Lambda]  # 1-D
            else:
            
                if not conditional.mean.blockform is None:
                    # For each conditional variable extract the coefficients of cond_vars (Omega) and concatenate 
                    # them all into a list of lists to form the appropriate blockform for 'comp_marg_vars'
                    # The remainder is collected in Lambda
                    Omega, Lambda = [], []                                             
                    for i in range(len(conditional.variables)):
                        if (not (isinstance(conditional.mean.blockform[i], SuperMatSymbol) or 
                            isinstance(conditional.mean.blockform[i], SuperMatInverse) or
                            isinstance(conditional.mean.blockform[i], SuperMatTranspose)) or 
                            (conditional.mean.blockform[i].expanded is None)):
                            expanded_cond_mean = conditional.mean.blockform[i]  
                        else: 
                            expanded_cond_mean = conditional.mean.to_full_expr()
                        
                        omega_i, lambda_i = utils.get_var_coeffs(expanded_cond_mean, cond_vars)
                        Omega.append(omega_i)
                        Lambda.append(lambda_i)
                else:
                    raise Exception("The conditional mean should have a blockform")
         
                                        
            # Create new mean blockform
            if marginal.mean.blockform is None:
                if not marginal.mean.expanded is None:
                    marginal_mean = [marginal.mean.expanded]
                else:
                    marginal_mean = [marginal.mean]
            else:
                marginal_mean = marginal.mean.blockform
            
            new_mean_blockform = utils.matadd(Lambda,utils.matmul(Omega,marginal_mean)) + \
                                 marginal_mean
            new_mean = [m.doit() for m in new_mean_blockform]
        
        
            # Create new covar blockform
            if conditional.covar.blockform is None:
                if not conditional.covar.expanded is None:
                    conditional_covar = [[conditional.covar.expanded]]
                else:
                    raise Exception("Conditional should have an expanded or blockform")
            else:
                conditional_covar = conditional.covar.blockform
        
            if marginal.covar.blockform is None:
                #if not marginal.covar.expanded is None:
                #    marginal_covar = [[marginal.covar.expanded]]
                #else:
                marginal_covar = [[marginal.covar]]
            else:
                marginal_covar = marginal.covar.blockform
        
            S_11 = utils.matadd(conditional_covar, utils.matmul(utils.matmul(Omega, marginal_covar),
                                                                utils.mattrans(Omega)))
            S_12 = utils.matmul(Omega, marginal_covar)
            S_21 = utils.mattrans(S_12)
            S_22 = marginal_covar
        
            new_covar_blockform = utils.create_blockform(S_11,S_12,S_21,S_22)

            # Turn block matrices into symbols
            for i in range(len(new_covar_blockform)):
                for j in range(len(new_covar_blockform[0])):
                    var_i, var_j = new_variables[i], new_variables[j]
                    new_covar_blockform[i][j] = Covariance(v1=var_i, v2=var_j,
                                                           cond_vars=new_conditioned_vars,
                                                           full_expr=new_covar_blockform[i][j].doit())
                                                    
            new_covar = new_covar_blockform
        
            # Choose new class prefix
            new_prefix = self.prefix if self.prefix != MVG.DEFAULT_PREFIX else other.prefix
        
            return MVG(new_variables, mean=new_mean, cov=new_covar, cond_vars=new_conditioned_vars,
                       prefix=new_prefix)
         
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """
        Operation for multiplying MVG objects.
        
        Split into 3 cases:
            - Both MVGs share same set of variables: p(x|.)*q(x|.)
            - MVGs overlap in variables: p(x,y|.)*p(y,z|.)
            - Conditional case: p(x|y,.)*p(y,.|z,.) or p(x|y)p(y,w|z) = p(x|y)p(y|w,z)p(w|z) = p(x,y,w|z)
        
        Returns an MVG object.
        """
        
        def _get_ordered_vars(vars1, vars2):
            """
            Returns a dictionary where the keys are Variables and the values are integers that
            specify an order on the set of variables.

            We obtain the variables from two lists ``vars1`` and ``vars2`` which come from
            different MVGs.
            """
            
            v_keys = dict(zip(vars1,range(len(vars1))))
            
            # Add the distinct variables from other.variables
            for v in vars2:
                if v not in v_keys:
                    v_keys[v] = len(v_keys)+1
            
            return v_keys
        
        def _get_ordered_sets_of_vars(svars1, svars2, v_keys):
            """
            Returns the sorted relative complements and intersection of two sets of variables,
            ``svars1`` and ``svars2`` using the values in the dictionary ``v_keys`` to sort
            elements i.e.

            If A=svars1 and B=svars2, we return

                comp1 = B\A, comp2 = A\B, joint12 = A intersection B

            where the lists are sorted based on values of dict ``v_keys``.
            """

            intersection = svars1 & svars2

            comp1 = sorted(list(svars1 - intersection), key=lambda m: v_keys[m])
            comp2 = sorted(list(svars2 - intersection), key=lambda m: v_keys[m])
            joint12 = sorted(list(intersection), key=lambda m: v_keys[m])
            
            return comp1, joint12, comp2
            
            
        self_variables = self.variables
        other_variables = other.variables
        svariables_self = set(self_variables)
        svariables_other = set(other_variables)
        
        
        # The type of multiplications we need to do depends on the variables of both MVGs
        # Case 1: Variables are same in both MVGs
        if svariables_self & svariables_other == svariables_self:
            product = self._mul_same_vars(other)
            return product
        else:
            variables_keys = _get_ordered_vars(self_variables, other_variables)  # Used to preserve orderings
            
            comp_self, joint, comp_other = _get_ordered_sets_of_vars(svariables_self,
                                                                     svariables_other,
                                                                     variables_keys)
            
            # First check that this isn't the conditioned case by checking that the random
            # variables in each MVG aren't conditioned on in the other MVG.
            conditional_case = (len(set(self.cond_vars) & svariables_other) != 0) or \
                               (len(set(other.cond_vars) & svariables_self) != 0)
            
            if not conditional_case:                
                if len(joint) > 0:  # Non-zero overlap
                    sets = (comp_self, joint, comp_other)
                    return self._mul_overlap_case(other, sets)
                else:
                    sets = (comp_self, comp_other)
                    return self._mul_disjoint_case(other, sets)
            else: # Case 3: Conditional case
                variables_self_other = (svariables_self, svariables_other)
                return self._mul_conditional_case(other, variables_self_other)
                     
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self.__mul__(other)        

    def multiply(self, other):
        """
        Multiplies two MVG objects
        :param other: The second MVG object
        :return: An MVG object that is the result of the operation.
        """
        return self.__mul__(other)

    def condition(self, x, use_full_expr=True):
        """
        Condition operation. MVG must have more than one Variable/CompositeVariable
        :param x: The list of Variables to condition on.
        :param use_full_expr: Indi
        :return: The conditional MVG i.e. p(x,y) -> p(y|x)
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

            new_mats = []
            for mat in (P, Q, R, S):
                if use_full_expr and mat.blockform is not None:
                    new_mats.append(mat.blockform)
                elif use_full_expr and mat.expanded is not None:
                    new_mats.append([[mat.expanded]])
                else:
                    new_mats.append([[mat]])

            P, Q, R, S = new_mats

            #if use_full_expr and all([mat.blockform is not None for mat in (P, Q, R, S)]):
            #    P, Q = P.blockform, Q.blockform
            #    R, S = R.blockform, S.blockform
            #else:
            #    P, Q = [[P]], [[Q]]
            #    R, S = [[R]], [[S]]
            
            # Marginal (Conditioned-on) variables
            marg_vars = [v for v in x]
                        
            # Conditional variables
            cond_vars = [v for v in self.variables if v not in x]
            
            # Create conditioned MVG parameters
            if use_full_expr:
                v_marg_vars = marg_vars
                old_mean = [self.mean.blockform[i] for i in range(len(self.variables))
                            if self.variables[i] not in x]
                marg_mean = [self.mean.blockform[i] for i in range(len(self.variables))
                             if self.variables[i] in x]
            else:
                name = 'v_('+','.join([v.name for v in marg_vars])+')'
                v_marg_vars = [CompositeVariable(name=name, variables=marg_vars)]
                old_mean, marg_mean = self.mean.partition(cond_vars_indices)
                old_mean = [old_mean]
                marg_mean = [marg_mean]
             
            new_mean_blockform = utils.matadd(old_mean,utils.matmul(Q,utils.matmul(utils.matinv(S),
                                            utils.matadd(v_marg_vars,utils.matmul(-1,marg_mean)))))
            new_mean_blockform = [m.doit() for m in new_mean_blockform]
            
            new_covar_blockform = utils.matadd(P,utils.matmul(utils.matmul(utils.matmul(-1, Q),
                                                                           utils.matinv(S)), R))
            new_covar_blockform = [[c.doit() for c in r] for r in new_covar_blockform]
                
                
            new_conditioned_vars = self.cond_vars+v_marg_vars
            
            if len(new_mean_blockform) == 1:
                new_cond_mean = new_mean_blockform[0].doit()
                new_cond_covar = new_covar_blockform[0][0].doit()
            else:
                new_cond_mean = new_mean_blockform
                new_cond_covar = new_covar_blockform
            
            
            return MVG(cond_vars, mean=new_cond_mean, cov=new_cond_covar,
                       cond_vars=new_conditioned_vars, prefix=self.prefix)
    
    def marginalise(self, x):
        
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
        
        if any([i==-1 for i in x_loc]):
            raise Exception("%s is not a variable of this MVG" % x)
                
        new_blockform = []
        for row in self.covar.blockform:
            new_blockform.append(list(row)) 
            
        # Delete down the rows then across the cols
        num_rows, num_cols = len(new_blockform), len(new_blockform[0])
        
        # Delete cols
        new_blockform = [[r[idx] for idx in range(num_cols) if idx not in x_loc]
                         for r in new_blockform]
            
        # Delete rows
        new_blockform = [row for i, row in enumerate(new_blockform) if i not in x_loc]
        
        # New variables
        new_variables = [v for v in self.variables if v not in x]
            
        # Create new moments
        if len(new_variables) > 1:
            new_covar = new_blockform
            new_mean = [self.mean.blockform[i].doit() for i in range(len(self.variables)) if
                        self.variables[i] not in x]
        else:
            new_covar = new_blockform[0][0]
            new_mean = [self.mean.blockform[i].doit() for i in range(len(self.variables)) if
                        self.variables[i] not in x][0]
            

        return MVG(new_variables, mean=new_mean, cov=new_covar, cond_vars=self.cond_vars,
                   prefix=self.prefix)
    
    def change_name(self, name):
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
        
        
                 
        
            
                
    
        
        
                
                
                
                
            
        
        
                 
        
        