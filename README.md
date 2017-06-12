# SymGP
A symbolic algebra library for multivariate Gaussian and Gaussian process probabilistic models

## Installation

Install from PyPi using the following command:

`sudo pip install symgp`

## Example of usage

See examples in folder notebooks

## Things to improve

- Allow `utils.replace` to replace expressions of the form `D+A` in `D+2*A` with `C` to give `A+C`.
- Provide functionality for `utils.simplify` to automatically replace expressions composed of `KernelMatrix` matrices e.g.  
   for an expression `K(f,u)*K(u,u).I*K(u,g)`, we would want to replace the whole expression with a new `Kernel`  
   `Q = Kernel(sub_kernels=[K,K],kernel_type='mul',mat=M,name='Q')` (see `kernel/kernel.py`) where  
   `M = Constant('M_{u,u}',n,n,full_expr=K(u,u).I)`, `n = Symbol('n')` (represents a variable shape)  
   `K = Kernel()` (represents a generic kernel function). This would then give the expression `Q(f,g)`  
   
   This can be done using `utils.replace` by calling `utils.replace(K(f,u)*K(u,u).I*K(u,g),[Q])` but we need to find  
   a way to detect these types of expressions automatically in `simplify`.
- Add more functionality to the `MVG` class to allow for example, addition of `MVG` objects.
- Improve code generation to convert symbolic expressions to code files in languages such as Python, Matlab, Julia, etc.   
   
