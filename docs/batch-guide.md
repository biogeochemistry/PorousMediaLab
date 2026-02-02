# Batch Reactor Guide

The `Batch` class simulates 0-dimensional (0D) batch reactor experiments - closed systems where only reactions occur without spatial transport.

## Creating a Batch

```python
from porousmedialab.batch import Batch

batch = Batch(tend=40, dt=1)
```

**Parameters:**
- `tend` (float): End time of simulation (in your chosen time units)
- `dt` (float): Timestep for integration

## Adding Species

```python
batch.add_species(name='O2', init_conc=0.2)
batch.add_species(name='CH4', init_conc=0.001)
batch.add_species(name='CO2', init_conc=0)
```

**Parameters:**
- `name` (str): Name of the chemical species (used in rate expressions)
- `init_conc` (float): Initial concentration

## Defining Constants

Constants are parameters used in rate expressions:

```python
batch.constants['k1'] = 0.1          # Rate constant
batch.constants['Km'] = 0.001        # Half-saturation constant
batch.constants['T'] = 25            # Temperature
```

## Defining Rate Expressions

Rate expressions are strings evaluated using [numexpr](https://github.com/pydata/numexpr):

```python
# First-order kinetics
batch.rates['R1'] = 'k1 * A'

# Michaelis-Menten kinetics
batch.rates['R2'] = 'Vmax * S / (Km + S)'

# Multiple terms
batch.rates['R3'] = 'k1 * A * B / (Km + B)'

# Inhibition term
batch.rates['R4'] = 'k1 * A * Km_inh / (Km_inh + I)'
```

**Available operators:** `+`, `-`, `*`, `/`, `**` (power)

**Available functions:** `exp`, `log`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `abs`, `where`

## Setting dcdt (Time Derivatives)

Define how each species concentration changes over time:

```python
# Simple consumption and production
batch.dcdt['A'] = '-R1'
batch.dcdt['B'] = 'R1'

# Multiple reactions affecting one species
batch.dcdt['C'] = 'R1 - R2 + 0.5 * R3'

# Stoichiometric coefficients
batch.dcdt['O2'] = '-2 * R1 - 4 * R2'
```

## Running the Simulation

```python
batch.solve(verbose=True)
```

**Parameters:**
- `verbose` (bool): If `True`, prints progress and time estimates

## Accessing Results

Results are stored in species dictionaries:

```python
# Get concentration time series (shape: [1, num_timesteps])
conc = batch.A.concentration

# Get final concentration
final = batch.A.concentration[0, -1]

# Get time array
time = batch.time

# Plot concentration vs time
import matplotlib.pyplot as plt
plt.plot(batch.time, batch.A.concentration[0])
```

## Plotting Methods

Built-in plotting methods:

```python
# Plot single species
batch.plot('A')

# Plot all species
batch.plot_profiles()

# Plot reaction rates
batch.plot_rates()

# Plot rate of change (delta)
batch.plot_deltas()
```

## Saving Results

Save results to HDF5 format:

```python
batch.save_results_in_hdf5()
```

This creates `results.h5` containing:
- `time`: Time array
- `concentrations`: All species concentrations
- `estimated_rates`: Calculated rates
- `parameters`: Constants used

## Advanced: Henry Equilibrium

For gas-liquid partitioning:

```python
# Add both aqueous and gas species
batch.add_species(name='CO2_aq', init_conc=0.001)
batch.add_species(name='CO2_gas', init_conc=0)

# Set Henry equilibrium (Hcc is dimensionless Henry constant)
batch.henry_equilibrium(aq='CO2_aq', gas='CO2_gas', Hcc=0.83)
```

## Advanced: Acid-Base Equilibrium

For pH-dependent systems:

```python
# Add carbonate species
batch.add_species(name='H2CO3', init_conc=0.001)
batch.add_species(name='HCO3', init_conc=0.01)
batch.add_species(name='CO3', init_conc=0.0001)

# Define acid with pKa values
batch.add_acid(
    species=['H2CO3', 'HCO3', 'CO3'],
    pKa=[6.35, 10.33],
    charge=0  # Charge of fully protonated form
)

# Add non-dissociating ion
batch.add_species(name='Na', init_conc=0.1)
batch.add_ion(name='Na', charge=1)

# Create the acid-base system
batch.create_acid_base_system()
```

After `create_acid_base_system()`, a `pH` species is automatically created and tracked.

## Complete Example

```python
from porousmedialab.batch import Batch

# Create batch reactor
batch = Batch(tend=40, dt=1)

# Add species
batch.add_species(name='OM', init_conc=0.01)    # Organic matter
batch.add_species(name='O2', init_conc=0.2)     # Oxygen
batch.add_species(name='CO2', init_conc=0)      # Carbon dioxide

# Set constants
batch.constants['k'] = 0.5      # Degradation rate
batch.constants['Km'] = 0.02    # Half-saturation

# Define rate
batch.rates['R_deg'] = 'k * OM * O2 / (Km + O2)'

# Set time derivatives
batch.dcdt['OM'] = '-R_deg'
batch.dcdt['O2'] = '-R_deg'
batch.dcdt['CO2'] = 'R_deg'

# Run and plot
batch.solve()
batch.plot_profiles()
```

## Tips

1. **Timestep selection**: Start with a larger `dt` and decrease if you see numerical instabilities
2. **Units**: Be consistent with units throughout (e.g., mol/L for concentrations, 1/day for rates)
3. **Debugging**: Use `verbose=True` to see simulation progress
4. **Rate expressions**: Ensure all variables in rate expressions are defined as species or constants
