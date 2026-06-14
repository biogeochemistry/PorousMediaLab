# Changelog

All notable changes to PorousMediaLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] (3.0.0)

### Fixed
- **Restored 4th/5th-order accuracy of the `rk4` and `butcher5` ODE solvers.** A
  spurious second `* dt` in the Runge-Kutta stage offset (`_sum_k`) had silently
  collapsed both explicit solvers to first-order (Euler) accuracy. Output from
  `ode_method='rk4'` and `'butcher5'` now changes (improves); the default
  `ode_method='scipy'` reaction path was unaffected by this particular bug. A new
  reference-free convergence-order test locks the corrected order in. Users who
  tuned `dt` to the old behavior can increase `dt` for equivalent accuracy, or
  pin `porousmedialab<3` for byte-identical legacy runs.
- `reconstruct_rates()` / `save_results_in_hdf5()` no longer raise when a model
  mixes a constant-only (zero-order) rate with a species-dependent rate; rates
  are broadcast to the spatial shape before stacking.
- `Column.load_initial_conditions()` no longer raises for models containing a
  no-transport / `D=0` species (e.g. pH or an immobile species); transport
  matrices are only rebuilt for transported species.
- `pHsolve()` now honors its `method` argument instead of always using
  `Nelder-Mead`; the pH depth-scan now falls back to a full solve when the local
  `+/-0.1` window edge is hit, so steep pH gradients between depths are resolved
  correctly.
- `Acid` now accepts `pKa=0.0` and array inputs (previously rejected by an
  ambiguous truthiness check).

### Changed
- **Unified negative-value clipping across all ODE paths via a per-species
  `allow_negative` flag.** Previously only the `rk4` path exempted `Temperature`
  from the non-negativity floor, while the default `scipy` path silently clipped
  any legitimately-negative variable to `1e-16`. `Batch.add_species` and
  `Column.add_species` now accept `allow_negative=False`; the name `'Temperature'`
  is auto-enrolled for backward compatibility. This changes the default `scipy`
  path output only for signed variables.
- Accessing an unknown attribute/species on a `Batch`/`Column` now raises
  `AttributeError` instead of `KeyError`, so typos surface clearly and `hasattr`
  behaves correctly.
- `solve_henry_law` now rejects any non-positive Henry's constant (`HenryC <= 0`),
  not only `-1`; non-positive constants are unphysical.
- `metrics.rsquared` now raises `ValueError` on zero-variance observations
  (delegating to the local `coefficient_of_determination`), where the previous
  scikit-learn-backed implementation returned `0`/`NaN`.
- The acid-base pH solver now warm-starts from the previous timestep; solver
  strategy and tolerance-visible pH output may differ slightly from 2.2.0.
- Bumped to **3.0.0** (SemVer): the changes above alter simulation output or
  error behavior for the affected solver/variable/input combinations.

### Removed
- **scikit-learn is no longer a runtime dependency.** `metrics.rsquared` is now a
  thin wrapper over the existing pure-NumPy `coefficient_of_determination`.

## [2.2.0] - 2026-06-03

### Added
- Input validation for numerical stability:
  - Division-by-zero guards in `desolver.py` (dx, phi, diff_coef)
  - Division-by-zero guards in `metrics.py` (pc_bias, apb, norm_rmse, NS, likelihood, index_agreement)
  - Henry's constant validation in `equilibriumsolver.py`
- CFL stability warning when coefficient exceeds 0.25

### Changed
- Behavior-preserving structural refactor of the core solver and class contracts (ODE integration output is bit-for-bit identical):
  - Replaced string-based boundary-condition handling in `desolver.py` with a `BoundaryConditionType` enum (public string API and constant/flux aliases preserved)
  - Decomposed `ode_integrate` into focused module-level helpers (`_k_loop`, `_rk4_step`, `_butcher5_step`)
  - De-duplicated `Column` top/bottom flux estimators and unified `Batch`/`Column` acid-base concentration updates into `Lab`
  - Moved `Batch`/`Column` plotter aliases into mixins in `plotting_mixins.py`

### Fixed
- Global namespace pollution in `lab.py` when using `exec()` for rate functions

### Removed
- `sensitivity.py` module (contained only unimplemented stubs)
- Dead `element.py` module and legacy Python 2 compatibility code

### Tests
- Added comprehensive tests for all new input validation
- Added 28 tests for blackbox.py optimization module (0% → 85% coverage)
- Added 9 integration tests validating mathematical correctness:
  - Mass conservation in batch reactions
  - Analytical solutions (1st/2nd order kinetics)
  - Multi-species cascade reactions
  - Michaelis-Menten kinetics
  - Column transport with Dirichlet BCs
  - Advection pulse transport
  - Timestep convergence
- Characterization, golden-value, and contract tests added alongside the refactor (test suite now 389 passing)

## [2.1.0] - 2026-01-30

### Added
- `create_vectorized_rate_function()` for 45x faster rate reconstruction
- Performance optimization benchmark at `benchmarks/benchmark_optimizations.py`

### Performance
- Enabled numexpr multi-threading for parallel rate expression evaluation
- Vectorized `reconstruct_rates()` method for 45x speedup
- Pre-allocated arrays in hot loops to reduce memory allocation overhead
- Overall 2.4x performance improvement for typical simulations

## [2.0.0] - 2026-01-30

### Added
- Vectorized ODE solver for 6-60x faster reaction integration
- `ode_integrate_vectorized()` function using `scipy.integrate.solve_ivp`
- `create_vectorized_ode_function()` for generating vectorized ODE functions
- `ODESolverError` exception for better error handling when solver fails
- Benchmark script at `benchmarks/benchmark_ode_solver.py`
- Comprehensive test suite with 262 tests
- Poetry for dependency management
- GitHub Actions CI workflow

### Changed
- Default `ode_method='scipy'` now uses vectorized solver (LSODA method)
- Improved error messages for ODE solver failures
- Modernized codebase for Python 3.10+ and NumPy 2.0

### Fixed
- Critical numerical bugs in ODE integration
- Single rate case handling in `reconstruct_rates()`

### Backward Compatibility
- Use `ode_method='scipy_sequential'` to restore previous sequential behavior
- All existing APIs remain unchanged

## [1.5.1] - 2026-01-30

### Added
- Poetry support for dependency management
- GitHub Actions CI workflow
- Comprehensive test coverage

### Fixed
- Python 3 / NumPy 2.0 compatibility issues
- Critical numerical bugs

## [1.4.2] - 2020-07-02

### Fixed
- Float to int cast bug

### Changed
- Updated Column example
- Updated examples

## [1.4.1] - 2019-10-09

### Changed
- Method now respects the boundaries
- Version bump and changelog

## [1.3.7] - 2019-09-06

### Added
- Saving results in HDF5 format

## [1.3.6] - 2019-09-04

### Fixed
- Floating point error in Calibrator when finding intersection index

## [1.3.5] - 2019-06-28

### Added
- Saving final concentrations with X coordinates
- Interpolation when loading on different X grid

## Earlier Versions

See [git history](https://github.com/biogeochemistry/PorousMediaLab/commits/master) for changes prior to v1.3.5.
