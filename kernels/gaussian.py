from kernels.kernel import *

class GaussianKernel(Kernel):
    
    def __init__(self, params, sub_kernels=[], kernel_type='nul', mat=None):
        """
            Create a Gaussian Kernel with given hyperparam symbols
        
            Args:
                mat, sub_kernels, kernel_type - See kernel.py
                params (in this order):
                    s_f - signal standard deviation
                    s_y - noise standard deviation
                    l - length-scale
        """
        super(GaussianKernel, self).__init__(sub_kernels, kernel_type, mat)
        self.s_f = params[0]
        self.s_y = params[1]
        self.l = params[2]
                    
    
    def eval_K(self, d):
        """
            Evaluates the kernel for the supplied inputs.
        
            Args:
                d - A dictionary mapping the symbols for the inputs (xi, xj) and parameters (s_f, s_y, l) to
                    numerical values
        
            Returns:
                The 
        """
        pass
        
        
        