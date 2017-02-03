from PorousMediaLab import PorousMediaLab
import numpy as np

D = 368
w = 0  # adjusted from 0.2
t = 182.5/365 * 2
dx = 0.2
L = 40
phi = 0.8
dt = 1e-4
is_solute = True

ftc_column = PorousMediaLab(L, dx, t, dt, phi)
ftc_column.add_temperature(D=281000, init_temperature=5)

ftc_column.add_species(is_solute=True, element='O2', D=368, init_C=0, bc_top=0.231, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
ftc_column.add_species(is_solute=True, element='Fe2', D=127, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
ftc_column.add_solid_species('OM', 5, 15, 0)
ftc_column.add_solid_species('FeOH3', 5, 75, 0)

ftc_column.constants['Q10'] = 2
ftc_column.constants['k_OM'] = 16
ftc_column.constants['Km_O2'] = 20e-3
ftc_column.constants['Km_FeOH3'] = 100


ftc_column.rates['R1'] = 'Q10**((Temperature-278)/10) * k_OM * OM * O2 / (Km_O2 + O2)'
ftc_column.rates['R2'] = 'Q10**((Temperature-278)/10) * k_OM * OM * FeOH3 / (Km_FeOH3 + FeOH3) * Km_O2 / (Km_O2 + O2)'


for i in range(1, len(ftc_column.time)):

    temp = 10 + 20 * np.sin(np.pi * 2 * ftc_column.time[i])
    ftc_column.Temperature.bc_top  = temp
    if temp < 0:
        ftc_column.O2.bc_top = 0
    else:
        ftc_column.O2.bc_top = 0.231

    # if ftc_column.time[i] > 182.5*1 / 365:
    #     ftc_column.Temperature.bc_top = -10
    # if ftc_column.time[i] > 182.5*2 / 365:
    #     ftc_column.Temperature.bc_top = 10
    # if ftc_column.time[i] > 182.5*3 / 365:
    #     ftc_column.Temperature.bc_top = -10
    ftc_column.integrate_one_timestep(i)
