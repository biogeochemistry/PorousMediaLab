#!/usr/bin/env python
"""
Benchmark script for comparing vectorized vs sequential ODE solvers.

This script measures performance of the vectorized ODE solver against the
sequential (scipy_sequential) solver for various problem sizes.

Usage:
    poetry run python benchmarks/benchmark_ode_solver.py

    # Or with specific configurations:
    poetry run python benchmarks/benchmark_ode_solver.py --quick
    poetry run python benchmarks/benchmark_ode_solver.py --full
"""

import argparse
import time
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from porousmedialab.column import Column


@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run."""
    N: int
    num_species: int
    method: str
    time_seconds: float
    timesteps: int
    with_transport: bool

    @property
    def time_per_step_ms(self) -> float:
        return (self.time_seconds / self.timesteps) * 1000


def run_benchmark(
    N: int,
    num_species: int,
    method: str,
    tend: float = 0.5,
    dt: float = 0.01,
    with_transport: bool = False,
    warmup: bool = True
) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        N: Number of spatial points (actual N will be N+1)
        num_species: Number of chemical species
        method: ODE method ('scipy' for vectorized, 'scipy_sequential' for sequential)
        tend: End time of simulation
        dt: Timestep
        with_transport: If True, enable transport (advection-diffusion)
        warmup: If True, run once before timing to warm up JIT/caches

    Returns:
        BenchmarkResult with timing information
    """
    length = N * 0.1  # dx = 0.1

    def create_column():
        col = Column(
            length=length,
            dx=0.1,
            tend=tend,
            dt=dt,
            w=0.1 if with_transport else 0,
            ode_method=method
        )

        # Add species
        for i in range(num_species):
            D = 10 - i if with_transport else 0
            init_conc = 1.0 if i > 0 else (0 if with_transport else 1.0)

            col.add_species(
                theta=0.9,
                name=f'S{i}',
                D=D,
                init_conc=init_conc,
                bc_top_value=1 if i == 0 and with_transport else 0,
                bc_top_type='dirichlet' if i == 0 and with_transport else 'flux',
                bc_bot_value=0,
                bc_bot_type='flux',
                int_transport=with_transport
            )

        # Simple bimolecular reaction: S0 + S1 -> products
        col.constants['k'] = 0.1
        col.rates['R'] = 'k * S0 * S1'
        col.dcdt['S0'] = '-R'
        col.dcdt['S1'] = '-R'

        # Other species are inert
        for i in range(2, num_species):
            col.dcdt[f'S{i}'] = '0'

        return col

    # Warmup run
    if warmup:
        col = create_column()
        col.solve(verbose=False)

    # Timed run
    col = create_column()
    start = time.perf_counter()
    col.solve(verbose=False)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        N=col.N,
        num_species=num_species,
        method=method,
        time_seconds=elapsed,
        timesteps=len(col.time) - 1,
        with_transport=with_transport
    )


def run_benchmark_suite(
    N_values: List[int],
    species_values: List[int],
    with_transport: bool = False,
    verbose: bool = True
) -> List[dict]:
    """Run a suite of benchmarks comparing vectorized vs sequential.

    Args:
        N_values: List of spatial point counts to test
        species_values: List of species counts to test
        with_transport: If True, enable transport
        verbose: If True, print results as they complete

    Returns:
        List of result dictionaries with speedup calculations
    """
    results = []

    if verbose:
        mode = "with transport" if with_transport else "reactions only"
        print(f"\nBenchmark Suite ({mode}):")
        print("-" * 75)
        print(f"{'N':<8} {'Species':<10} {'Sequential':<14} {'Vectorized':<14} {'Speedup':<10}")
        print("-" * 75)

    for N in N_values:
        for num_species in species_values:
            # Run sequential
            result_seq = run_benchmark(
                N=N,
                num_species=num_species,
                method='scipy_sequential',
                with_transport=with_transport
            )

            # Run vectorized
            result_vec = run_benchmark(
                N=N,
                num_species=num_species,
                method='scipy',
                with_transport=with_transport
            )

            speedup = result_seq.time_seconds / result_vec.time_seconds

            result = {
                'N': result_seq.N,
                'num_species': num_species,
                'sequential_time': result_seq.time_seconds,
                'vectorized_time': result_vec.time_seconds,
                'speedup': speedup,
                'with_transport': with_transport,
                'timesteps': result_seq.timesteps
            }
            results.append(result)

            if verbose:
                print(
                    f"{result['N']:<8} "
                    f"{num_species:<10} "
                    f"{result_seq.time_seconds:<14.3f}s "
                    f"{result_vec.time_seconds:<14.3f}s "
                    f"{speedup:<10.1f}x"
                )

    return results


def print_summary(results: List[dict]):
    """Print summary statistics from benchmark results."""
    if not results:
        return

    speedups = [r['speedup'] for r in results]

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  Min speedup:     {min(speedups):.1f}x")
    print(f"  Max speedup:     {max(speedups):.1f}x")
    print(f"  Mean speedup:    {np.mean(speedups):.1f}x")
    print(f"  Median speedup:  {np.median(speedups):.1f}x")

    # Find best case
    best = max(results, key=lambda x: x['speedup'])
    print(f"\n  Best case: N={best['N']}, S={best['num_species']} -> {best['speedup']:.1f}x")

    # Find worst case
    worst = min(results, key=lambda x: x['speedup'])
    print(f"  Worst case: N={worst['N']}, S={worst['num_species']} -> {worst['speedup']:.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark vectorized vs sequential ODE solvers'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with fewer configurations'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full benchmark with many configurations'
    )
    parser.add_argument(
        '--transport',
        action='store_true',
        help='Include transport benchmarks'
    )
    args = parser.parse_args()

    print("=" * 75)
    print("PorousMediaLab ODE Solver Benchmark")
    print("=" * 75)
    print(f"  Vectorized solver: ode_method='scipy' (default)")
    print(f"  Sequential solver: ode_method='scipy_sequential'")
    print()

    if args.quick:
        N_values = [50, 100]
        species_values = [2, 3]
    elif args.full:
        N_values = [20, 50, 100, 200, 500]
        species_values = [2, 3, 5, 10]
    else:
        # Default: moderate benchmark
        N_values = [50, 100, 200, 500]
        species_values = [2, 5]

    # Reactions only benchmark
    results_reactions = run_benchmark_suite(
        N_values=N_values,
        species_values=species_values,
        with_transport=False
    )

    # Transport benchmark (optional)
    results_transport = []
    if args.transport:
        results_transport = run_benchmark_suite(
            N_values=N_values[:3],  # Smaller set for transport
            species_values=[2, 3],
            with_transport=True
        )

    # Print summaries
    print("\n" + "=" * 75)
    print("REACTIONS ONLY SUMMARY")
    print_summary(results_reactions)

    if results_transport:
        print("\n" + "=" * 75)
        print("WITH TRANSPORT SUMMARY")
        print_summary(results_transport)

    print("\n" + "=" * 75)
    print("NOTES:")
    print("  - Vectorized solver uses LSODA (auto-detects stiffness)")
    print("  - Sequential solver uses LSODA with BDF preference")
    print("  - Speedup is highest for large N and moderate species count")
    print("  - For stiff problems, consider using ode_method='scipy_sequential'")
    print("=" * 75)


if __name__ == '__main__':
    main()
