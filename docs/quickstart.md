# Quickstart Guide

Get started with PorousMediaLab in 5 minutes.

## Installation

```bash
pip install porousmedialab
```

## Batch Reactor Example

A simple batch reactor with two species reacting:

```python
from porousmedialab.batch import Batch

# Create batch: 10 days simulation, 0.1 day timestep
batch = Batch(tend=10, dt=0.1)

# Add species with initial concentrations
batch.add_species(name='A', init_conc=1.0)  # 1.0 mol/L
batch.add_species(name='B', init_conc=0.0)

# Define reaction rate constant
batch.constants['k'] = 0.5  # 1/day

# Define rate expression
batch.rates['R1'] = 'k * A'

# Define how concentrations change
batch.dcdt['A'] = '-R1'      # A is consumed
batch.dcdt['B'] = 'R1'       # B is produced

# Run simulation
batch.solve()

# Access results
print(batch.A.concentration[0, -1])  # Final concentration of A
print(batch.B.concentration[0, -1])  # Final concentration of B

# Plot results
batch.plot_profiles()
```

## Column Transport Example

A 1D column with diffusion and advection:

```python
from porousmedialab.column import Column

# Create column: 10 cm long, 0.1 cm mesh, 1 day simulation
column = Column(length=10, dx=0.1, tend=1, dt=0.001, w=0.5)

# Add dissolved species
column.add_species(
    theta=0.9,              # Porosity
    name='C',               # Species name
    D=100,                  # Diffusion coefficient (cm²/day)
    init_conc=0,            # Initial concentration
    bc_top_value=1.0,       # Top boundary value
    bc_top_type='dirichlet',# Constant concentration at top
    bc_bot_value=0,         # Bottom boundary value
    bc_bot_type='flux'      # Zero flux at bottom
)

# Run simulation
column.solve()

# Plot concentration profiles
column.plot_profiles()

# Plot contour (depth vs time)
column.contour_plot('C')
```

## Next Steps

- [Batch Guide](batch-guide.md) - Detailed guide for batch reactors
- [Column Guide](column-guide.md) - Detailed guide for column transport
- [Tutorials](tutorials/batch-roden.md) - Step-by-step examples

## Key Concepts

| Concept | Description |
|---------|-------------|
| `species` | Chemical compounds being tracked |
| `constants` | Parameters used in rate expressions |
| `rates` | Mathematical expressions for reaction rates |
| `dcdt` | Time derivative expression for each species |
| `profiles` | Current concentration values at each grid point |
