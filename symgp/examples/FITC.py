import sys
# Add the symgp folder path to the sys.path list
module_path = r'/Users/jaduol/Documents/Uni (original)/Part II/IIB/MEng Project/'
if module_path not in sys.path:
    sys.path.append(module_path)

from symgp import SuperMatSymbol, utils, MVG, Variable, SuperDiagMat, Kernel
from sympy import symbols, ZeroMatrix, Identity

m, n, l = symbols('m n l')
s_y = symbols('\u03c3_y')
K = Kernel()

u = Variable('u',m,1)
p_u = MVG([u],mean=ZeroMatrix(m,1),cov=K(u,u))

print("p_u:\n ", p_u)

f, fs, y = utils.variables("f f_{*} y",[n, l, n])

q_fgu = MVG([f], mean=K(f,u)*K(u,u).I*u,
                 cov=SuperDiagMat(K(f,f)-K(f,u)*K(u,u).I*K(u,f)),
                 cond_vars=[u],
                 prefix='q_{FITC}')

print("q_fgu:\n")

q_fsgu = MVG([fs], mean=K(fs,u)*K(u,u).I*u,
                   cov=K(fs,fs)-K(fs,u)*K(u,u).I*K(u,fs),
                   cond_vars=[u],
                   prefix='q_{FITC}')

print("q_fgu:\n ", q_fsgu)

# q(f,fs|u)
q_f_fs_g_u = q_fgu*q_fsgu

print("q_f_fs_g_u:\n ", q_f_fs_g_u)

# q(f,fs,u)
q_f_fs_u = q_f_fs_g_u*p_u

print("q_f_fs_u:\n ", q_f_fs_u)

# Effective prior: q(f,fs)
q_f_fs = q_f_fs_u.marginalise([u])

print("q_f_fs:\n ", q_f_fs)

p_ygf = MVG([y],mean=f,cov=s_y**2*Identity(n),cond_vars=[f])

print("p_ygf:\n ", p_ygf)

# q(f,fs,y)
q_f_fs_y = p_ygf*q_f_fs

print("q_f_fs_y:\n ", q_f_fs_y)

# q(f,fs|y)
q_f_fs_g_y = q_f_fs_y.condition([y])

print("q_f_fs_g_y:\n ", q_f_fs_g_y)

# q(fs|y)
q_fs_g_y = q_f_fs_g_y.marginalise([f])

print("q_fs_g_y:\n ", q_fs_g_y)