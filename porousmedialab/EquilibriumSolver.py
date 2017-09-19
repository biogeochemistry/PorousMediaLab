# from sympy.core.symbol import symbols
# from sympy.solvers.solveset import nonlinsolve
# from numpy import vectorize
# from sympy.utilities.lambdify import lambdify
# import numpy as np

# Gas, Aq = symbols('Gas, Aq')
# TOT, Hcc = symbols('TOT, Hcc')


# def henry_law_algebraic_solution(TOT, Hcc):
#     return nonlinsolve([Aq / Gas / Hcc - 1, Gas + Aq - TOT], [Gas, Aq])


# henry_law_solution = henry_law_algebraic_solution(TOT, Hcc)


# def henry_law_substitute_single(totC, HenryC):
#     return next(iter(henry_law_solution.subs([(TOT, totC), (Hcc, HenryC)])))


# def solve_henry_law(totC_vec, HenryC):
#     func = lambdify('TOT', henry_law_substitute_single('TOT', HenryC), 'numpy')
#     return func(totC_vec)


def solve_henry_law(totC_vec, HenryC):
    g = totC_vec / (1 + HenryC)
    a = totC_vec - g
    return g, a
