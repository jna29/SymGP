import MVG as MVG
import SuperMatExpr as SME
from sympy import *
import utils as utils


# Define some symbols
D, N, Ns = symbols('D N Ns')
sig_y = symbols('\u03c3_y')

# Prior
w = SME.SuperMatSymbol(D,1,name='w',mat_type='var')
mu_w = SME.SuperMatSymbol(D,1,mat_type='mean',dep_vars=[w],expanded=ZeroMatrix(D,1))
S_ww = SME.SuperMatSymbol(D,D,mat_type='covar',dep_vars=[w],expanded=Identity(D))
p_w = MVG.MVG([w],moments=[mu_w,S_ww,utils.getZ(S_ww)])

# Likelihood of w given X
X = SME.SuperMatSymbol('X',D,N,name='X',mat_type='var')
y = SME.SuperMatSymbol('y',N,1,name='X',mat_type='var')
mu_y = SME.SuperMatSymbol('',N,1,mat_type'mean',dep_vars=[y],cond_vars=[w,X],expanded=X.T*w)
S_yy = SME.SuperMatSymbol('',N,N,'covar',dep_vars=[y],cond_vars=[w,X],expanded=sig_y**2*Identity(N))
p_y = MVG.MVG([y],moments=[mu_y,S_yy,utils.getZ(S_yy)],conditioned_vars=[w,X])

# Joint of w and y
p_w_y = p_w*p_y

# Inference: posterior over w
p_w_post = p_w_y.condition([y])

#Prediction

# Likelihood of w given Xs
Xs = SME.SuperMatSymbol('X_s',D,Ns,'var')
ys = SME.SuperMatSymbol('y_s',Ns,1,'var')
mu_ys = SME.SuperMatSymbol('',Ns,1,'mean',dep_vars=[ys],cond_vars=[w,Xs],expanded=Xs.T*w)
S_ysys = SME.SuperMatSymbol('',Ns,Ns,'covar',dep_vars=[ys],cond_vars=[w,Xs],expanded=sig_y**2*Identity(Ns))
p_ys = MVG.MVG([ys],moments=[mu_ys,S_ysys,utils.getZ(S_ysys)],conditioned_vars=[w,Xs])

# Joint of w and ys
p_w_ys = p_w_post*p_ys

# Predictive distribution of ys
p_ys_post = p_w_ys.marginalize([w])


