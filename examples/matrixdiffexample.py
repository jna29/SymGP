# Declaration
 
# -*- coding: utf-8 -*-
 
#----------------------------------------------------------------------
#
# FUNCTIONS FOR THE AUTOMATIC DIFFERENTIATION  OF MATRICES WITH SYMPY
# 
#----------------------------------------------------------------------
 
from sympy import *
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
 
 
 
#####  M  E  T  H  O  D  S
 
 
 
def matrices(names):
    ''' Call with  A,B,C = matrix('A B C') '''
    return symbols(names,commutative=False)
 
 
# Transformations
 
d = Function("d",commutative=False)
inv = Function("inv",commutative=False)
 
class t(Function):
    ''' The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) '''
    is_commutative = False
    def __new__(cls,arg):
        if arg.is_Add:
            return Add(*[t(A) for A in arg.args])
        elif arg.is_Mul:
            L = len(arg.args)
            return Mul(*[t(arg.args[L-i-1]) for i in range(L)])
        else:
            return Function.__new__(cls,arg)
 
 
# Differentiation
 
MATRIX_DIFF_RULES = { 
         
        # e =expression, s = a list of symbols respsect to which
        # we want to differentiate
         
        Symbol : lambda e,s : d(e) if (e in s) else 0,
        Add :  lambda e,s : Add(*[matDiff(arg,s) for arg in e.args]),
        Mul :  lambda e,s : Mul(matDiff(e.args[0],s),Mul(*e.args[1:]))
                      +  Mul(e.args[0],matDiff(Mul(*e.args[1:]),s)) ,
        t :   lambda e,s : t( matDiff(e.args[0],s) ),
        inv : lambda e,s : - e * matDiff(e.args[0],s) * e
}
 
def matDiff(expr,symbols):
    if expr.__class__ in MATRIX_DIFF_RULES.keys():
        return  MATRIX_DIFF_RULES[expr.__class__](expr,symbols)
    else:
        return 0
 
 
 
#####  C  O  S  M  E  T  I  C  S
 
 
# Console mode
 
class matStrPrinter(StrPrinter):
    ''' Nice printing for console mode : X¯¹, X', ∂X '''
     
    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return  self._print(expr.args[0]) +'¯¹'
        else:
            return '(' +  self._print(expr.args[0]) + ')¯¹'
     
    def _print_t(self, expr):
        return  self._print(expr.args[0]) +"'"
     
    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return '∂'+  self._print(expr.args[0])
        else:
            return '∂('+  self._print(expr.args[0]) +')'   
 
def matPrint(m):
    mem = Basic.__str__ 
    Basic.__str__ = lambda self: matStrPrinter().doprint(self)
    print(str(m).replace('*',''))
    Basic.__str__ = mem
 
 
# Latex mode
 
class matLatPrinter(LatexPrinter):
    ''' Printing instructions for latex : X^{-1},  X^T, \partial X '''
     
    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) +'^{-1}'
        else:
            return '(' + self._print(expr.args[0]) + ')^{-1}'
    def _print_t(self, expr):
        return  self._print(expr.args[0]) +'^T'
     
    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return '\partial '+ self._print(expr.args[0])
        else:
            return '\partial ('+ self._print(expr.args[0]) +')'
 
def matLatex(expr, profile=None, **kargs):
    if profile is not None:
        profile.update(kargs)
    else:
        profile = kargs
    return matLatPrinter(profile).doprint(expr)
 
 
 
#####    T  E  S  T  S
 
 
X,S = matrices("X S")
H= X*inv(t(X)*inv(S)*X)*t(X)*inv(S)

matPrint(inv(X)*X) 
#matPrint(  expand( expand( matDiff(H,[X]) ) ) )
 
#print(matLatex( matDiff(H,[X]) ))