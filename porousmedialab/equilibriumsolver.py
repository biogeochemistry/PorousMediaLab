def solve_henry_law(totC_vec, HenryC):
    """Solves Henry's law for gas-aqueous equilibrium.

    Args:
        totC_vec: total concentration vector
        HenryC: Henry's constant

    Returns:
        tuple: (g, a) where g is gas phase and a is aqueous phase concentration

    Raises:
        ValueError: if HenryC equals -1 (causes division by zero)
    """
    if HenryC == -1:
        raise ValueError("Henry's constant cannot be -1 (causes division by zero)")
    g = totC_vec / (1 + HenryC)
    a = totC_vec - g
    return g, a
