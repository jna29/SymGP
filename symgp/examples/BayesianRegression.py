import sys
# Add the symgp folder path to the sys.path list
module_path = r'/Users/jaduol/Documents/Uni (original)/Part II/IIB/MEng Project/'
if module_path not in sys.path:
    sys.path.append(module_path)

from symgp import SuperMatSymbol, utils, MVG, Variable
from sympy import symbols, ZeroMatrix, Identity


# Define some symbols
D, N, Ns = symbols('D N Ns')
sig_y = symbols('\u03c3_y')

# Prior
w = Variable('w',D,1)
p_w = MVG([w],mean=ZeroMatrix(D,1),cov=Identity(D))

print("p_w:\n ", p_w)

# Likelihood of w given X
X, y = utils.variables('X y',[(D,N), N])
p_y = MVG([y], mean=X.T*w,
               cov=sig_y**2*Identity(N),
               cond_vars=[w,X])

print("p_y:\n ", p_y)

# Joint of w and y
p_w_y = p_w*p_y

print("p_w_y:\n ", p_w_y)

# Inference: posterior over w
p_w_post = p_w_y.condition([y])

print("p_w_post:\n ", p_w_post)

#Prediction

# Likelihood of w given Xs
Xs, ys = utils.variables('X_{*} y_{*}',[(D,Ns), Ns])
p_ys = MVG([ys], mean=Xs.T*w,
                 cov=sig_y**2*Identity(Ns),
                 cond_vars=[w,Xs])

print("p_ys:\n ", p_ys)

# Joint of w and ys
p_w_ys = p_w_post*p_ys

print("p_w_ys:\n ", p_w_ys)

# Predictive distribution of ys
p_ys_post = p_w_ys.marginalise([w])

print("p_ys_post:\n ", p_ys_post)