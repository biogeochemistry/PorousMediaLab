# Changelog

All notable changes to PorousMediaLab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
