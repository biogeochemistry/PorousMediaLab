[![latest version](https://badge.fury.io/py/porousmedialab.svg)](https://badge.fury.io/py/porousmedialab)
[![GitHub issues](https://img.shields.io/github/issues/biogeochemistry/PorousMediaLab.svg)](https://img.shields.io/github/issues/biogeochemistry/PorousMediaLab.svg)
[![GitHub forks](https://img.shields.io/github/forks/biogeochemistry/PorousMediaLab.svg)](https://img.shields.io/github/forks/biogeochemistry/PorousMediaLab.svg)
[![GitHub stars](https://img.shields.io/github/stars/biogeochemistry/PorousMediaLab.svg)](https://img.shields.io/github/stars/biogeochemistry/PorousMediaLab.svg)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://img.shields.io/badge/license-MIT-blue.svg)
[![Twitter](https://img.shields.io/twitter/url/https/github.com/biogeochemistry/PorousMediaLab.svg?style=social)](https://img.shields.io/twitter/url/https/github.com/biogeochemistry/PorousMediaLab.svg?style=social)
[![DOI](https://zenodo.org/badge/78385496.svg)](https://zenodo.org/badge/latestdoi/78385496)

# PorousMediaLab

The toolbox for batch and 1D reactive transport modelling in porous media aimed at the easiness of use for the user without computational background.

## What's New in v2.0.0

**6-60x faster reaction integration** with the new vectorized ODE solver:

- Reactions are now solved using a vectorized approach that processes all spatial points simultaneously
- Uses `scipy.integrate.solve_ivp` with LSODA method (auto-detects stiffness)
- Backward compatible: use `ode_method='scipy_sequential'` for the old behavior

| Spatial Points | Species | Speedup |
|----------------|---------|---------|
| 50 | 2 | 13x |
| 100 | 3 | 23x |
| 500 | 5 | 62x |

Run the benchmark yourself:
```bash
poetry run python benchmarks/benchmark_ode_solver.py
```

# How to use

Have a look at ["examples"](https://github.com/biogeochemistry/PorousMediaLab/tree/master/examples) folder.

# Installation using pip

- Install Python version 3.10 or higher ([click](https://www.python.org/downloads/));
- To install the toolbox run ```pip install porousmedialab```
- In terminal run ```jupyter notebook```;
- You will see the folders in your home folder. You can navigate in any folder and create a new notebook with your model.

# Development installation using Poetry

- Install Python version 3.10 or higher ([click](https://www.python.org/downloads/));
- Install Poetry ([click](https://python-poetry.org/docs/#installation));
- Clone this repository: ```git clone https://github.com/biogeochemistry/PorousMediaLab```
- Navigate to the project folder: ```cd PorousMediaLab```
- Install dependencies: ```poetry install```
- Run tests: ```poetry run pytest```
- Activate the virtual environment: ```poetry shell```

# Manual installation

- Install Python version 3.10 or higher ([click](https://www.python.org/downloads/));
- Download and unzip or clone (using git) this repository (PorousMediaLab);
- Open terminal and go to the PorousMediaLab folder using ```cd``` command. If you have problems with the terminal, check this [guide](https://www.davidbaumgold.com/tutorials/command-line/);
- Install dependencies: ```pip install numpy numexpr scipy matplotlib seaborn h5py scikit-learn```
- In terminal run command ```jupyter notebook```;
- You will see the folders of the PorousMediaLab project; you can go in "examples" folder and play with them.

# Testing

To run the test suite:

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

For coverage report:

```bash
pytest tests/ --cov=porousmedialab --cov-report=term-missing
```

# Citation

Igor Markelov (2020). Modelling Biogeochemical Cycles Across Scales: From Whole-Lake Phosphorus Dynamics to Microbial Reaction Systems. UWSpace. http://hdl.handle.net/10012/15513

# Contribution

I am looking for contributors specifically for incorporation of:

- sensitivity tests
- unsaturated flow
- thermodynamic calculations
- your crazy ideas and needs

If you wish to contribute in this open source project, please, create pull request or contact me via email: is.markelov@gmail.com. 

# Acknowledgements

This project was funded by 

- Lakes in Transition (Research Council of Norway project no. 244558/E50 (https://prosjektbanken.forskningsradet.no/#/project/NFR/244558)

- the Canada Excellence Research Chair in Ecohydrology (https://www.cerc.gc.ca/chairholders-titulaires/former-ancien-eng.aspx).


# MIT License

Copyright (c) 2019 Igor Markelov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
