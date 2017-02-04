from PorousMediaLab import PorousMediaLab
import numpy as np


t = 182.5 / 365 * 2
dx = 0.2
L = 40
phi = 0.8
dt = 0.5e-4


x = np.linspace(0, L, L / dx + 1)
Fe3_init = np.zeros(x.size)
Fe3_init[x > 5] = 75
Fe3_init[x > 15] = 0
Fe3_init[x > 25] = 75
Fe3_init[x > 35] = 0


ftc_column = PorousMediaLab(L, dx, t, dt, phi)
ftc_column.add_temperature(D=281000, init_temperature=5)

ftc_column.add_species(is_solute=True, element='O2', D=368, init_C=0, bc_top=0.231, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
ftc_column.add_species(is_solute=True, element='Fe2', D=127, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
ftc_column.add_species(is_solute=False, element='OM', D=5, init_C=15, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
ftc_column.add_solid_species('FeOH3', 5, Fe3_init, 0)

ftc_column.constants['Q10'] = 2
ftc_column.constants['k_OM'] = 100
ftc_column.constants['Km_O2'] = 20e-3
ftc_column.constants['Km_FeOH3'] = 10
ftc_column.constants['k8'] = 1.4e+5

# Q10**((Temperature-278)/10) *
# Q10**((Temperature-278)/10) *

ftc_column.rates['R1'] = 'k_OM * OM * O2 / (Km_O2 + O2)'
ftc_column.rates['R2'] = 'k_OM * OM * FeOH3 / (Km_FeOH3 + FeOH3) * Km_O2 / (Km_O2 + O2)'
ftc_column.rates['R8'] = 'k8 * O2 * Fe2'

ftc_column.dcdt['OM'] = '-R1-R2'
ftc_column.dcdt['O2'] = '-R1-R8'
ftc_column.dcdt['FeOH3'] = '-4*R2+R8'
ftc_column.dcdt['Fe2'] = '-R8+4*R2'



for i in range(1, len(ftc_column.time)):
    temp = 10 + 20 * np.sin(np.pi * 2 * ftc_column.time[i])
    ftc_column.Temperature.bc_top = temp
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
