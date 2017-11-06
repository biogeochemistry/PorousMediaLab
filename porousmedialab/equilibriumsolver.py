def solve_henry_law(totC_vec, HenryC):
    g = totC_vec / (1 + HenryC)
    a = totC_vec - g
    return g, a
