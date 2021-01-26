import sympy as sym
from sympy.abc import x, y, z

fu = 9*x**2 + 5 * y**3 - 3*z

print(sym.diff(sym.poly(fu), x))
print(sym.diff(sym.poly(fu), y))
print(sym.diff(sym.poly(fu), z))