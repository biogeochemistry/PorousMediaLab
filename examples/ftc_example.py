from PorousMediaLab import PorousMediaLab
import numpy as np

# Setting up the properties of time and space domain:
t = 28 / 365
dx = 0.2
L = 40
phi = 0.8
dt = 1e-5
x = np.linspace(0, L, L / dx + 1)

# Not simple initial concentration of solid iron:
Fe3_init = np.zeros(x.size)
Fe3_init[x > 5] = 75
Fe3_init[x > 15] = 0
Fe3_init[x > 25] = 75
Fe3_init[x > 35] = 0

# Create the domain:
ftc = PorousMediaLab(L, dx, t, dt, phi)

# SImulate Temperature with thermal diffusivity coefficient 281000 and init temperature 5C:
ftc.add_temperature(D=281000, init_temperature=5)

# Adding species with names, diffusion coefficients, initial concentrations and boundary top and bottom conditions:
ftc.add_species(is_solute=True, element='O2', D=368, init_C=0, bc_top=0.231, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
ftc.add_species(is_solute=True, element='Fe2', D=127, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
ftc.add_species(is_solute=False, element='OM', D=5, init_C=15, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
ftc.add_solid_species('FeOH3', 5, Fe3_init, 0)

# Specify the constants used in the rates:
ftc.constants['Q10'] = 4
ftc.constants['k_OM'] = 1
ftc.constants['Km_O2'] = 20e-3
ftc.constants['Km_FeOH3'] = 10
ftc.constants['k8'] = 1.4e+5

# Specify the equations for rates:
ftc.rates['R1'] = 'Q10**((Temperature-5)/10) * k_OM * OM * O2 / (Km_O2 + O2)'
ftc.rates['R2'] = 'Q10**((Temperature-5)/10) * k_OM * OM * FeOH3 / (Km_FeOH3 + FeOH3) * Km_O2 / (Km_O2 + O2)'
ftc.rates['R8'] = 'k8 * O2 * Fe2'

# ODEs for specific species:
ftc.dcdt['OM'] = '-R1-R2'
ftc.dcdt['O2'] = '-R1-R8'
ftc.dcdt['FeOH3'] = '-4*R2+R8'
ftc.dcdt['Fe2'] = '-R8+4*R2'

# Because we are changing the boundary conditions for temperature and Oxygen (when T < 0 => no oxygen at the top), then we
# need to have a time loop:
for i in range(1, len(ftc.time)):
    if ftc.time[i] > 0:
        # Thawing period: 5C + sin function changes during 1 day
        ftc.Temperature.bc_top = 5 + 10 * np.sin(np.pi * 2 * ftc.time[i]) + 5 * np.sin(np.pi * 2 * ftc.time[i] * 365)
    if ftc.time[i] > 7 / 365:
        # Freezing period: -10C + sin function changes during 1 day
        ftc.Temperature.bc_top = -10 + 10 * np.sin(np.pi * 2 * ftc.time[i]) + 5 * np.sin(np.pi * 2 * ftc.time[i] * 365)
    if ftc.time[i] > 14 / 365:
        # Thawing period: 5C + sin function changes during 1 day
        ftc.Temperature.bc_top = 5 + 10 * np.sin(np.pi * 2 * ftc.time[i]) + 5 * np.sin(np.pi * 2 * ftc.time[i] * 365)
    if ftc.time[i] > 21 / 365:
        # Freezing period: -10C + sin function changes during 1 day
        ftc.Temperature.bc_top = -10 + 10 * np.sin(np.pi * 2 * ftc.time[i]) + 5 * np.sin(np.pi * 2 * ftc.time[i] * 365)

    # when T < 0 => no oxygen at the top:
    if ftc.Temperature.bc_top < 0:
        ftc.O2.bc_top = 0
    else:
        ftc.O2.bc_top = 0.231

    # Integrate one timestep:
    ftc.integrate_one_timestep(i)
