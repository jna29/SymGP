"""
    Kalman Filtering
"""

from sympy import *
import utils as utils
import SuperMatExpr as SME
import MVG as MVG

# Shapes
m, n = symbols('m n')

# Variables                               
x_prev = SME.SuperMatSymbol(m,1,name='x_(t-1)',mat_type='var')    # x_(t-1)
x_t = SME.SuperMatSymbol(m,1,name='x_t',,mat_type='var')           # x_t
y_prev = SME.SuperMatSymbol(n,1,name='y_(1:t-1)',mat_type='var')  # y_(1:t-1) 
y_t = SME.SuperMatSymbol(n,1,name='y_t',mat_type='var')           # y_t
b_t = SME.SuperMatSymbol(m,1,name='b_t',mat_type='var')           # b_t
d_t = SME.SuperMatSymbol(n,1,name='d_t',mat_type='var')           # d_t
w_t = SME.SuperMatSymbol(m,1,name='w_t',mat_type='var')           # w_t
e_t = SME.SuperMatSymbol(n,1,name='e_t',mat_type='var')           # e_t

# Constant parameters
A = SME.SuperMatSymbol(m,m,name='A',mat_type='other')                                     
Q = SME.SuperMatSymbol(m,m,name='Q',mat_type='covar',dependent_vars=[w_t])
R = SME.SuperMatSymbol(n,n,name='R',mat_type='covar',dependent_vars=[e_t])
C = SME.SuperMatSymbol(n,m,name='B'mat_type='other')

# p(x_(t-1)|y_(1:t-1))
f_prev = SME.SuperMatSymbol(m,1,name='m_(t|t)',mat_type='mean',dependent_vars=[x_prev])   # \hat{x}_(t|t) (mean)
F_prev = SME.SuperMatSymbol(m,m,name='P_(t|t)'mat_type='covar',dependent_vars=[x_prev])  # P_(t|t) (covariance)
p_xprev = MVG.MVG([x_prev], moments=[f_prev, F_prev, utils.getZ(F_prev)], conditioned_vars=[y_prev])

# p(x_t|x_(t-1))
m_xt = SME.SuperMatSymbol(m,1,mat_type='mean',dependent_vars=[x_t],cond_vars=[x_prev],expanded=A*x_prev + b_t) 
cov_xt = SME.SuperMatSymbol(m,m,mat_type='covar',dependent_vars=[x_t],cond_vars=[x_prev],expanded=Q)
p_xt = MVG.MVG([x_t],moments=[m_xt, cov_xt, utils.getZ(cov_xt)], conditioned_vars=[x_prev])

# p(x_t,x_(t-1)|y_(1:t-1))
p_xt_xprev = p_xt*p_xprev

# p(x_t|y_(1:t-1))
p_xt_predict = p_xt_xprev.marginalize([x_prev])

# p(y_t|x_t)
m_yt = SME.SuperMatSymbol(n,1,mat_type='mean',dependent_vars=[y_t],cond_vars=[x_t],expanded=C*x_t + d_t)
cov_yt = SME.SuperMatSymbol(n,n,mat_type='covar',dependent_vars=[y_t],cond_vars=[x_t],expanded=R)
p_yt = MVG.MVG([y_t], moments=[m_yt,cov_yt,utils.getZ(cov_yt)],conditioned_vars=[x_t])

# p(y_t,x_t|y_(1:t-1))
p_yt_xt = p_yt*p_xt_predict

# p(x_t|y_(1:t))
p_xt_update = p_yt_xt.condition([y_t])

print(utils.matLatex(utils.expand_to_fullexpr(p_xt_update.mean.expanded)))
print(utils.matLatex(utils.expand_to_fullexpr(p_xt_update.covar.expanded)))