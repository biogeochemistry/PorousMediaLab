import numpy as np

dt = 0.1

conc = {'OM': np.array([0, 1, 2, 3]), 'O2': np.array([0, 1, 2, 3])}
coef = {'k': 1}
rates = {'R1':'k*OM*O2'}
dcdt = {'O2':'-4 * R1', 'OM': '-R1'}

rates_num = {}

for k in rates:
    rates_num[k] = eval(rates[k], {**coef, **conc})

dcdt_num={}
for k in dcdt:
    dcdt_num[k] = eval(dcdt[k], rates_num)

# k1 = {}
# k2 = {}
# k3 = {}
# k4 = {}
# k5 = {}

# for k in conc:
#     k1[k] = dt*dcdt[k]



# for k in conc:

# print(rates_num)
# a = eval('-R1', rates_num)
print(dcdt_num)
