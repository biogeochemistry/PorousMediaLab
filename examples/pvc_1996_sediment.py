from PorousMediaLab import PorousMediaLab

w = 0.2
t = 10
dx = 0.2
L = 25
phi = 0.9
dt = 1e-1

sediment = PorousMediaLab(L, dx, t, dt, phi, w)

sediment.add_species(is_solute=True, element='O2', D=368, init_C=0, bc_top=0.231, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='NO3', D=359, init_C=0, bc_top=1.5e-3, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='Mn2', D=220, init_C=2e-3, bc_top=2e-3, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='Fe2', D=127, init_C=0, bc_top=0, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='SO4', D=189, init_C=15, bc_top=28, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='NH4', D=363, init_C=22e-3, bc_top=22e-3, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='CH4', D=220, init_C=0, bc_top=0, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='TIC', D=220, init_C=0, bc_top=0, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='TIC', D=220, init_C=0, bc_top=0, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=True, element='TRS', D=284, init_C=0, bc_top=0, bc_top_type='dirichlet', bc_bot=0, bc_bot_type='flux')


sediment.add_species(is_solute=False, element='OM1', D=20, init_C=15, bc_top=180, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='OM2', D=20, init_C=5, bc_top=40, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='MnO2', D=20, init_C=0, bc_top=40, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='FeOH3', D=20, init_C=0, bc_top=75, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='MnCO3', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='FeCO3', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='adsFe', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='adsMn', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='adsNH4', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')
sediment.add_species(is_solute=False, element='FeS', D=20, init_C=0, bc_top=0, bc_top_type='flux', bc_bot=0, bc_bot_type='flux')


sediment.constants['x1'] = 112
sediment.constants['y1'] = 16
sediment.constants['x2'] = 200
sediment.constants['y2'] = 20
sediment.constants['k_OM1'] = 1
sediment.constants['k_OM2'] = 0.1
sediment.constants['H'] = 10**(-7.5 + 3)
sediment.constants['T'] = 5
sediment.constants['F'] = 0.6
sediment.constants['Km_O2'] = 20e-3
sediment.constants['Km_NO3'] = 5e-3
sediment.constants['Km_SO4'] = 1.6
sediment.constants['Km_MnO2'] = 16
sediment.constants['Km_FeOH3'] = 100
sediment.constants['xMn'] = 1
sediment.constants['xFe'] = 1
sediment.constants['k7'] = 5e+6 * 1e-3
sediment.constants['k8'] = 1.4e+5 * 1e-3
sediment.constants['k9'] = 5e+7 * 1e-3
sediment.constants['k10'] = 3e+6 * 1e-3
sediment.constants['k11'] = 5e+6 * 1e-3
sediment.constants['k12'] = 1.6e+5 * 1e-3
sediment.constants['k13'] = 2e+4 * 1e-3
sediment.constants['k14'] = 8e+3 * 1e-3
sediment.constants['k15'] = 3e+5 * 1e-3
sediment.constants['k16'] = 1e+10 * 1e-3
sediment.constants['k17'] = 1e+4 * 1e-3
sediment.constants['k21'] = 1e-4 * 1e+3
sediment.constants['k21m'] = 2.5e-1
sediment.constants['k22'] = 4.5e-4 * 1e+3
sediment.constants['k22m'] = 2.5e-1
sediment.constants['k23'] = 1.5e-5 * 1e+3
sediment.constants['k23m'] = 1e-3
sediment.constants['K_MnCO3'] = 10**(-8.5 + 6)
sediment.constants['K_FeCO3'] = 10**(-8.4 + 6)
sediment.constants['K_FeS'] = 10**(-2.2 + 3)

sediment.rates['R1a'] = 'k_OM1 * OM1 * O2 /  (Km_O2 + O2)'
sediment.rates['R1b'] = 'k_OM2 * OM2 * O2 /  (Km_O2 + O2)'
sediment.rates['R2a'] = 'k_OM1 * OM1 * NO3 /  (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R2b'] = 'k_OM2 * OM2 * NO3 /  (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R3a'] = 'k_OM1 * OM1 * MnO2 /  (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R3b'] = 'k_OM2 * OM2 * MnO2 /  (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R4a'] = 'k_OM1 * OM1 * FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R4b'] = 'k_OM2 * OM2 * FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R5a'] = 'k_OM1 * OM1 * SO4 / (Km_SO4 + SO4 ) * Km_FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R5b'] = 'k_OM2 * OM2 * SO4 / (Km_SO4 + SO4 ) * Km_FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R6a'] = 'k_OM1 * OM1 * Km_SO4 / (Km_SO4 + SO4 ) * Km_FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R6b'] = 'k_OM2 * OM2 * Km_SO4 / (Km_SO4 + SO4 ) * Km_FeOH3 / (Km_FeOH3 + FeOH3) * Km_MnO2 / (Km_MnO2 + MnO2) * Km_NO3 / (Km_NO3 + NO3) * Km_O2 / (Km_O2 + O2)'
sediment.rates['R7'] = 'k7 * O2 * adsMn'
sediment.rates['R8'] = 'k8 * O2 * Fe2'
sediment.rates['R9'] = 'k9 * O2 * adsFe'
sediment.rates['R10'] = 'k10 * MnO2 * Fe2'
sediment.rates['R11'] = 'k11 * NH4 * O2'
sediment.rates['R12'] = 'k12 * TRS * O2'
sediment.rates['R13'] = 'k13 * TRS * MnO2'
sediment.rates['R14'] = 'k14 * TRS * FeOH3'
sediment.rates['R15'] = 'k15 * FeS * O2'
sediment.rates['R16'] = 'k16 * CH4 * O2'
sediment.rates['R17'] = 'k17 * CH4 * SO4'
sediment.rates['R21'] = 'k21 * (Mn2 * TIC / xMn / K_MnCO3  - 1)'
sediment.rates['R21m'] = 'k21m * MnCO3 * (1 - Mn2 * TIC / xMn / K_MnCO3)'
sediment.rates['R22'] = 'k22 * (Fe2 * TIC / xFe / K_FeCO3 - 1)'
sediment.rates['R22m'] = 'k22m * FeCO3 * (1 - Fe2 * TIC / xFe / K_FeCO3)'
sediment.rates['R23'] = 'k23 * (Fe2 * TRS / H / K_FeS - 1)'
sediment.rates['R23m'] = 'k23m * FeS * (1 - Fe2 * TRS / H / K_FeS)'

sediment.dcdt['O2'] = 'F * (  - (x1 + 2 * y1) / x1 * R1a - (x2 + 2 * y2) / x2 * R1b - R7 / 2 - R9 / 4 - 2 * R15) - ( R8 / 4 + 2 * R11 + 2 * R12 + 2 * R16)'
sediment.dcdt['NO3'] = 'F * ( y1/x1 * R1a + y2/x2 * R1b -  (4 * x1 + 3 * y1) / 5 / x1 * R2a + (4 * x2 + 3 * y2) / 5 / x2 * R2b) + R11'
sediment.dcdt['Mn2'] = 'F * ( 2 * (R3a + R3b) + R10 + R13  - xMn * R21 + xMn * R21m)'
sediment.dcdt['Fe2'] = 'F * ( 4 * (R4a + R4b)- 2 * R10 + 2 * R14 + R15 - xFe * R22 + xFe * R22m) - R8'
sediment.dcdt['SO4'] = 'F * ( - (R5a + R5b)/2 + R15 ) + R12 - R17'
sediment.dcdt['NH4'] = 'F * y1/x1 * ( R1a + R2a + R3a + R4a + R5a + R6a ) + F * y2/x2 * (R1b + R2b + R3b + R4b + R5b + R6b) - R11'
sediment.dcdt['CH4'] = 'F / 2 * (R6a + R6b)- R16 - R17'
sediment.dcdt['adsMn'] = '-R7'
sediment.dcdt['adsFe'] = '-R9'
sediment.dcdt['adsNH4'] = '0'
sediment.dcdt['OM1'] = '- (R1a + R2a + R3a + R4a + R5a + R6a)'
sediment.dcdt['OM2'] = '- (R1b + R2b + R3b + R4b + R5b + R6b)'
sediment.dcdt['MnO2'] = '-2 * (R3a + R3b) + R7 - R10 - R13'
sediment.dcdt['FeOH3'] = '-4 * (R4a + R4b)+ R8 / F + R9 + 2 * R10 - 2 * R14'
sediment.dcdt['MnCO3'] = 'xMn * ( R21 - R21m )'
sediment.dcdt['FeCO3'] = 'xFe * ( R22 - R22m )'
sediment.dcdt['FeS'] = '- R15 + R23 - R23m'
sediment.dcdt['TIC'] = 'F *  (R1a + R2a + R3a + R4a + R5a + R6a/2 + R1b + R2b + R3b + R4b + R5b + R6b + R6b/2 - R21 + R21m - R22 + R22m) + R16 + R17'
sediment.dcdt['TRS'] = 'F * ( (R5a + R5b)/2 - R13 - R14 - R23 + R23m) - R12 - R17'

sediment.solve()
