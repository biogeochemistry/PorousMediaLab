def solve_henry_law(totC_vec, HenryC):
    """Solves Henry's law for gas-aqueous equilibrium.

    Args:
        totC_vec: total concentration vector
        HenryC: Henry's constant

    Returns:
        tuple: (g, a) where g is gas phase and a is aqueous phase concentration

    Raises:
        ValueError: if HenryC is not positive (a dimensionless Henry's constant
            must be > 0; non-positive values are unphysical and HenryC == -1
            also causes division by zero)
    """
    if HenryC <= 0:
        raise ValueError(
            f"Henry's constant must be positive, got {HenryC}")
    g = totC_vec / (1 + HenryC)
    a = totC_vec - g
    return g, a
