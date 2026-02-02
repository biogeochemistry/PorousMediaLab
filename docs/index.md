# PorousMediaLab Documentation

PorousMediaLab is a Python toolbox for batch and 1D reactive transport modeling in porous media, designed for scientists without extensive computational backgrounds.

## Quick Links

| Document | Description |
|----------|-------------|
| [Quickstart](quickstart.md) | Get started in 5 minutes |
| [Batch Guide](batch-guide.md) | Complete guide to 0D batch reactor simulations |
| [Column Guide](column-guide.md) | Complete guide to 1D reactive transport |

## Tutorials

Step-by-step tutorials with real-world examples:

- [Organic Matter Degradation](tutorials/batch-roden.md) - Batch reactor modeling following Roden 2008
- [Sediment Diagenesis](tutorials/sediment-column.md) - 1D column transport with complex reactions

## Features

- **Batch Module** - 0D closed-system reaction simulations
- **Column Module** - 1D advection-diffusion-reaction equation solver
- **Flexible Reactions** - Define rates using string expressions with numexpr
- **pH Equilibrium** - Built-in acid-base chemistry calculations
- **Henry Equilibrium** - Gas-liquid partitioning
- **Calibration** - Parameter optimization against experimental data
- **Visualization** - Built-in plotting methods

## Installation

```bash
pip install porousmedialab
```

For development:

```bash
git clone https://github.com/biogeochemistry/PorousMediaLab
cd PorousMediaLab
poetry install
```

## Citation

Igor Markelov (2020). Modelling Biogeochemical Cycles Across Scales: From Whole-Lake Phosphorus Dynamics to Microbial Reaction Systems. UWSpace. http://hdl.handle.net/10012/15513

## Links

- [GitHub Repository](https://github.com/biogeochemistry/PorousMediaLab)
- [PyPI Package](https://pypi.org/project/porousmedialab/)
- [Example Notebooks](https://github.com/biogeochemistry/PorousMediaLab/tree/master/examples)
