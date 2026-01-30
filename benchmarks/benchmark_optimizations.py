#!/usr/bin/env python
"""
Benchmark script for measuring performance optimizations.

Measures individual components to verify each optimization provides real improvement:
- Total solve() time
- reconstruct_rates() time
- Rate evaluation time

Usage:
    poetry run python benchmarks/benchmark_optimizations.py
    poetry run python benchmarks/benchmark_optimizations.py --quick
    poetry run python benchmarks/benchmark_optimizations.py --save baseline.json
    poetry run python benchmarks/benchmark_optimizations.py --compare baseline.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from porousmedialab.column import Column


@dataclass
class TimingResult:
    """Stores timing results for a benchmark run."""
    N: int
    num_species: int
    num_timesteps: int
    solve_time: float
    reconstruct_time: float
    total_time: float

    @property
    def solve_per_step_ms(self) -> float:
        return (self.solve_time / self.num_timesteps) * 1000


def create_test_column(N: int, num_species: int, tend: float = 1.0, dt: float = 0.01) -> Column:
    """Create a test Column with configurable parameters."""
    length = N * 0.1  # dx = 0.1

    col = Column(
        length=length,
        dx=0.1,
        tend=tend,
        dt=dt,
        w=0.1,
        ode_method='scipy'
    )

    # Add species
    for i in range(num_species):
        D = 10 - i
        init_conc = 1.0 if i > 0 else 0.0

        col.add_species(
            theta=0.9,
            name=f'S{i}',
            D=D,
            init_conc=init_conc,
            bc_top_value=1 if i == 0 else 0,
            bc_top_type='dirichlet' if i == 0 else 'flux',
            bc_bot_value=0,
            bc_bot_type='flux',
            int_transport=True
        )

    # Bimolecular reaction: S0 + S1 -> products
    col.constants['k'] = 0.1
    col.rates['R'] = 'k * S0 * S1'
    col.dcdt['S0'] = '-R'
    col.dcdt['S1'] = '-R'

    # Additional species are inert but participate in second reaction if present
    if num_species > 2:
        col.constants['k2'] = 0.05
        col.rates['R2'] = 'k2 * S1 * S2'
        col.dcdt['S2'] = '-R2'

    for i in range(3, num_species):
        col.dcdt[f'S{i}'] = '0'

    return col


def run_benchmark(N: int, num_species: int, tend: float = 1.0, dt: float = 0.01,
                  warmup: bool = True) -> TimingResult:
    """Run a single benchmark measuring solve and reconstruct times."""

    # Warmup run (for JIT, caches, etc.)
    if warmup:
        col = create_test_column(N, num_species, tend=0.1, dt=dt)
        col.solve(verbose=False)

    # Create fresh column for timed run
    col = create_test_column(N, num_species, tend=tend, dt=dt)

    # Time solve()
    start_total = time.perf_counter()
    start_solve = time.perf_counter()
    col.solve(verbose=False)
    solve_time = time.perf_counter() - start_solve

    # Time reconstruct_rates()
    start_reconstruct = time.perf_counter()
    col.reconstruct_rates()
    reconstruct_time = time.perf_counter() - start_reconstruct

    total_time = time.perf_counter() - start_total

    return TimingResult(
        N=col.N,
        num_species=num_species,
        num_timesteps=len(col.time) - 1,
        solve_time=solve_time,
        reconstruct_time=reconstruct_time,
        total_time=total_time
    )


def run_benchmark_suite(quick: bool = False) -> list[dict]:
    """Run the full benchmark suite."""

    if quick:
        configs = [
            (50, 2),
            (100, 3),
        ]
        tend = 0.5
    else:
        configs = [
            (50, 2),
            (100, 2),
            (100, 3),
            (200, 3),
            (200, 5),
        ]
        tend = 1.0

    results = []

    print("\n" + "=" * 80)
    print("Performance Optimization Benchmark")
    print("=" * 80)
    print(f"\n{'N':<8} {'Species':<10} {'Solve(s)':<12} {'Reconstruct(s)':<16} {'Total(s)':<12}")
    print("-" * 80)

    for N, num_species in configs:
        result = run_benchmark(N, num_species, tend=tend)
        results.append(asdict(result))

        print(
            f"{result.N:<8} "
            f"{result.num_species:<10} "
            f"{result.solve_time:<12.3f} "
            f"{result.reconstruct_time:<16.3f} "
            f"{result.total_time:<12.3f}"
        )

    return results


def print_summary(results: list[dict]):
    """Print summary statistics."""

    total_solve = sum(r['solve_time'] for r in results)
    total_reconstruct = sum(r['reconstruct_time'] for r in results)
    total_all = sum(r['total_time'] for r in results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total solve time:       {total_solve:.3f}s")
    print(f"  Total reconstruct time: {total_reconstruct:.3f}s")
    print(f"  Grand total:            {total_all:.3f}s")
    print(f"\n  Reconstruct as % of total: {(total_reconstruct/total_all)*100:.1f}%")


def compare_results(current: list[dict], baseline: list[dict]):
    """Compare current results against baseline."""

    print("\n" + "=" * 80)
    print("COMPARISON vs BASELINE")
    print("=" * 80)
    print(f"\n{'N':<8} {'Species':<10} {'Solve':<14} {'Reconstruct':<16} {'Total':<12}")
    print("-" * 80)

    # Match by N and num_species
    baseline_map = {(r['N'], r['num_species']): r for r in baseline}

    for curr in current:
        key = (curr['N'], curr['num_species'])
        if key not in baseline_map:
            continue

        base = baseline_map[key]

        solve_speedup = base['solve_time'] / curr['solve_time'] if curr['solve_time'] > 0 else float('inf')
        recon_speedup = base['reconstruct_time'] / curr['reconstruct_time'] if curr['reconstruct_time'] > 0 else float('inf')
        total_speedup = base['total_time'] / curr['total_time'] if curr['total_time'] > 0 else float('inf')

        print(
            f"{curr['N']:<8} "
            f"{curr['num_species']:<10} "
            f"{solve_speedup:>6.2f}x       "
            f"{recon_speedup:>6.2f}x          "
            f"{total_speedup:>6.2f}x"
        )

    # Overall speedup
    total_base = sum(r['total_time'] for r in baseline)
    total_curr = sum(r['total_time'] for r in current)
    overall_speedup = total_base / total_curr if total_curr > 0 else float('inf')

    print("-" * 80)
    print(f"{'OVERALL':<18} {'':<14} {'':<16} {overall_speedup:>6.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Benchmark performance optimizations')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--save', type=str, help='Save results to JSON file')
    parser.add_argument('--compare', type=str, help='Compare against baseline JSON file')
    args = parser.parse_args()

    results = run_benchmark_suite(quick=args.quick)
    print_summary(results)

    if args.save:
        path = Path(args.save)
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {path}")

    if args.compare:
        path = Path(args.compare)
        if path.exists():
            baseline = json.loads(path.read_text())
            compare_results(results, baseline)
        else:
            print(f"\nBaseline file not found: {path}")


if __name__ == '__main__':
    main()
