"""Shared fixtures for PorousMediaLab tests."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


from porousmedialab.batch import Batch
from porousmedialab.column import Column
from porousmedialab.phcalc import Acid, Neutral, System


@pytest.fixture
def simple_batch():
    """Minimal Batch object for testing reactions."""
    return Batch(tend=1.0, dt=0.01)


@pytest.fixture
def batch_with_species():
    """Batch with a single species for decay tests."""
    batch = Batch(tend=1.0, dt=0.001)
    batch.add_species(name='C', init_conc=1.0)
    return batch


@pytest.fixture
def simple_column():
    """Minimal Column for transport tests (no advection)."""
    return Column(length=10, dx=0.1, tend=1.0, dt=0.001, w=0)


@pytest.fixture
def column_with_advection():
    """Column with advection for transport tests."""
    return Column(length=10, dx=0.1, tend=1.0, dt=0.001, w=0.5)


@pytest.fixture
def column_with_species():
    """Column with O2 species and Dirichlet boundary conditions."""
    col = Column(length=10, dx=0.1, tend=0.1, dt=0.001, w=0)
    col.add_species(
        theta=1,
        name='O2',
        D=1e-5,
        init_conc=0,
        bc_top_value=1,
        bc_top_type='constant',
        bc_bot_value=0,
        bc_bot_type='flux'
    )
    return col


@pytest.fixture
def monoprotic_acid():
    """Acetic acid (pKa=4.76) for pH tests."""
    return Acid(pKa=[4.76], charge=0, conc=0.1)


@pytest.fixture
def polyprotic_acid():
    """Phosphoric acid (H3PO4) for polyprotic tests."""
    return Acid(pKa=[2.148, 7.198, 12.375], charge=0, conc=0.1)


@pytest.fixture
def carbonic_acid():
    """Carbonic acid system (H2CO3)."""
    return Acid(pKa=[6.35, 10.33], charge=0, conc=0.001)


@pytest.fixture
def neutral_ion():
    """Neutral chloride ion for charge balance."""
    return Neutral(charge=-1, conc=0.1)


@pytest.fixture
def acid_base_system(monoprotic_acid):
    """Simple acid-base system with acetic acid."""
    return System(monoprotic_acid)


@pytest.fixture
def temp_hdf5_file(tmp_path):
    """Temporary HDF5 file path for I/O tests."""
    return tmp_path / "test.h5"


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Temporary directory for CSV file tests."""
    return tmp_path


@pytest.fixture
def van_genuchten_pars():
    """Standard Van Genuchten parameters for silt loam."""
    return {
        'thetaR': 0.131,
        'thetaS': 0.396,
        'alpha': 0.423,
        'n': 2.06,
        'm': 1 - 1 / 2.06,
        'Ks': 0.0496,
        'neta': 0.5,
        'Ss': 0.000001
    }


@pytest.fixture
def simple_ode_system():
    """Simple ODE system for testing: dC/dt = -k*C."""
    return {
        'C0': {'C': 1.0},
        'coef': {'k': 2.0},
        'rates': {'R': 'k*C'},
        'dcdt': {'C': '-R'},
        'dt': 0.0001,
        'T': 0.01
    }
