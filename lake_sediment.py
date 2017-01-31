import numpy as np
from sediment_class import Sediment

D = 368
w = 1.5
t = 30
dx = 0.1
L = 25
phi = 0.9
dt = 0.0001
rho = 2
Init_C = 0.231
bc = 0.231

time = np.linspace(0, t, t / dt + 1)

sediment = Sediment(L, dx, t, dt, phi, w)
sediment.add_solute_species('O2', D, Init_C, bc)
sediment.add_solid_species('OM1', 5, 0, 1)
sediment.add_solid_species('OM2', 5, 0, 1)
sediment.add_solute_species('NO3', 359, 1.5e-3, 0)
sediment.add_solid_species('FeOH3', 5, 0, 75)
sediment.add_solute_species('SO4', 189, 28, 0)
sediment.add_solute_species('NH4', 363, 22e-3, 0)
sediment.add_solute_species('Fe2', 127, 0, 0)
sediment.add_solid_species('FeOOH', 5, 0, 0)
sediment.add_solute_species('H2S', 284, 0, 0)
sediment.add_solute_species('HS', 284, 0, 0)
sediment.add_solid_species('FeS', 5, 0, 0)
sediment.add_solute_species('S0', 100, 0, 0)
sediment.add_solute_species('PO4', 104, 0, 0)
sediment.add_solid_species('S8', 5, 0, 0)
sediment.add_solid_species('FeS2', 5, 0, 0)
sediment.add_solid_species('AlOH3', 5, 0, 0)
sediment.add_solid_species('PO4adsa', 5, 0, 0)
sediment.add_solid_species('PO4adsb', 5, 0, 0)
sediment.add_solute_species('Ca2', 141, 0, 0)
sediment.add_solid_species('Ca3PO42', 5, 0, 0)
sediment.add_solid_species('OMS', 5, 0, 0)

sediment.constants = {'k_OM1': 1, 'k_OM2': 0.1, 'Km_O2': 0.02, 'Km_NO3': 0.005, 'Km_FeOH3': 50, 'Km_FeOOH': 50, 'Km_SO4': 1.6, 'Km_oxao': 0.001, 'Km_amao': 0.1, 'Kin_O2': 0.3292, 'Kin_NO3': 0.1, 'Kin_FeOH3': 0.1, 'Kin_FeOOH': 0.1, 'k_amox': 2000, 'k_Feox': 8.7e1, 'k_Sdis': 0.1, 'k_Spre': 2500, 'k_FeS2pre': 3.17, 'k_alum': 0.1,
                      'k_pdesorb_a': 1.35, 'k_pdesorb_b': 1.35, 'k_rhom': 6500, 'k_tS_Fe': 0.1, 'Ks_FeS': 2510, 'k_Fe_dis': 0.001, 'k_Fe_pre': 21.3, 'k_apa': 0.37, 'kapa': 3e-6, 'k_oms': 0.3134, 'k_tsox': 1000, 'k_FeSpre': 0.001, 'accel': 30, 'f_pfe': 1e-6, 'k_pdesorb_c': 1.35, 'Cx1': 112, 'Ny1': 10, 'Pz1': 1, 'Cx2': 200, 'Ny2': 20, 'Pz2': 1}

sediment.rates['R1a'] = 'accel * k_OM1*OM1 * O2 /  (Km_O2 + O2)'
sediment.rates['R1b'] = 'accel * k_OM2*OM2 * O2 /  (Km_O2 + O2)'
sediment.rates['R2a'] = 'k_OM1*OM1 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R2b'] = 'k_OM2*OM2 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R3a'] = 'k_OM1*OM1 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R3b'] = 'k_OM2 *OM2 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R4a'] = 'k_OM1*OM1 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R4b'] = 'k_OM2*OM2 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R5a'] = 'k_OM1*OM1 * SO4 / (Km_SO4 + SO4 ) * Kin_FeOOH / (Kin_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R5b'] = 'k_OM2*OM2 * SO4 / (Km_SO4 + SO4 ) * Kin_FeOOH / (Kin_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)'
sediment.rates['R6'] = 'k_tsox * O2 * (HS+H2S)'
sediment.rates['R7'] = 'k_tS_Fe * FeOH3 *  (HS+H2S)'
sediment.rates['R8'] = 'k_Feox * Fe2 * O2'
sediment.rates['R9'] = 'k_amox * O2 / (Km_oxao + O2) * (NH4 / (Km_amao + NH4))'
sediment.rates['R10'] = 'k_oms * (HS+H2S) * (OM1 + OM2)'
sediment.rates['R11'] = 'k_FeSpre * FeS * S0'
sediment.rates['R12'] = 'k_rhom * O2 * FeS'
sediment.rates['R13'] = 'k_FeS2pre * FeS * (HS+H2S)'
sediment.rates['R14a'] = 'k_Fe_pre * ( Fe2 * (HS+H2S) / (1e-3**2 * Ks_FeS) - 1)'
sediment.rates['R14b'] = 'k_Fe_dis * FeS * ( 1 - Fe2 * (HS+H2S) / (1e-3**2 * Ks_FeS))'
sediment.rates['R15a'] = 'k_Spre * S0'
sediment.rates['R15b'] = 'k_Sdis * S8'
sediment.rates['R16a'] = 'k_pdesorb_a * FeOH3 * PO4'
sediment.rates[
    'R16b'] = 'f_pfe * (4 * (k_OM1*OM1 * FeOH3 /  (Km_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)+k_OM2*OM2 * NO3 /  (Km_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)) + 2 * k_tS_Fe * FeOH3 *  (HS+H2S))'
sediment.rates['R17a'] = 'k_pdesorb_b * FeOOH * PO4'
sediment.rates['R17b'] = 'f_pfe * (4 * (k_OM1*OM1 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)+k_OM2*OM2 * FeOOH /  (Km_FeOOH + FeOOH) * Kin_FeOH3 / (Kin_FeOH3 + FeOH3) * Kin_NO3 / (Kin_NO3 + NO3) * Kin_O2 / (Kin_O2 + O2)))'
sediment.rates['R18a'] = 'k_pdesorb_c * PO4 * AlOH3'
sediment.rates['R19'] = 'k_apa * (PO4 - kapa)'

sediment.dcdt['O2'] = '-0.25 * R8  - 2 * R9  - (R1a+R1b) - 3 * R12'
sediment.dcdt['OM1'] = '-1/Cx1*(R1a+R2a+R3a+R4a+R5a) - R10 '
sediment.dcdt['OM2'] = '-1/Cx2*(R1b+R2b+R3b+R4b+R5b) - R10'
sediment.dcdt['NO3'] = '- 0.8*(R2a+R2b)+ R9'
sediment.dcdt['FeOH3'] = '-4 * (R3a+R3b) - R16a - 2*R7 + R8'
sediment.dcdt['SO4'] = '- 0.5*(R5a+R5b) + R6'
sediment.dcdt['NH4'] = '(Ny1/Cx1 * (R1a+R2a+R3a+R4a+R5a) + Ny2/Cx2 * (R1b+R2b+R3b+R4b+R5b)) - R9'
sediment.dcdt['Fe2'] = '4*(R3a+R3b) + 4*(R4a+R4b) + 2*R7 - R8 + R14b - R14a'
sediment.dcdt['FeOOH'] = '-4*(R4a+R4b) - R17a + R12'
sediment.dcdt['H2S'] = '0'
sediment.dcdt['HS'] = '0.5*(R5a+R5b) - R6 - R7 + R14b - R14a - R10 -R13'
sediment.dcdt['FeS'] = '- R14b - R11 - 4*R12 -R13 + R14a'
sediment.dcdt['S0'] = '- R11 - R15a + R7 + R15b'
sediment.dcdt['PO4'] = '(Pz1/Cx1 * (R1a+R2a+R3a+R4a+R5a) + Pz2/Cx2 * (R1b+R2b+R3b+R4b+R5b)) + R16b + R17b - 2 * R19 - R18a - R16a - R17a'
sediment.dcdt['S8'] = '4*R12 - R15b + R15a'
sediment.dcdt['FeS2'] = '+ R11 + R13'
sediment.dcdt['AlOH3'] = '-R18a'
sediment.dcdt['PO4adsa'] = 'R16a - R16b'
sediment.dcdt['PO4adsb'] = 'R17a - R17b'
sediment.dcdt['Ca2'] = '-3*R19'
sediment.dcdt['Ca3PO42'] = 'R19'
sediment.dcdt['OMS'] = 'R10'

for i in np.arange(1, len(time)):
    bc = 0.231 + 0.2 * np.sin(time[i] * 2 * 3.14)
    sediment.new_boundary_condition('O2', bc)
    sediment.integrate_one_timestep(i)

