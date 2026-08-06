import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, DATA_DIR, P_DEFAULT
from generate_data import generate_sim_data, load_real_data, load_time_matrix
from solver import PMedianSolver
from visualize import (
    plot_population_density, plot_service_areas,
    plot_travel_time_map, plot_coverage_analysis,
    plot_objective_vs_p
)
from visualize_html import plot_html_map


def save_results(data, facilities, assignments, objective, p_value, method,
                 time_matrix=None, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    solver = PMedianSolver(data, p_value=p_value, time_matrix=time_matrix)
    _, nearest_time = solver._compute_assignment(facilities)
    travel_times_min = nearest_time / 60.0

    result_df = data.copy()
    result_df['assigned_facility_idx'] = assignments
    result_df['assigned_facility_id'] = [facilities[a] for a in assignments]
    result_df['travel_time_min'] = travel_times_min

    result_path = os.path.join(output_dir, f'results_P{p_value}_{method}.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {result_path}")

    weights = data['population'].values.astype(float)
    summary_path = os.path.join(output_dir, f'summary_P{p_value}_{method}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("P-Median Site Selection Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Method: {method.upper()}\n")
        f.write(f"Number of sites P = {p_value}\n")
        f.write(f"Number of grids n = {len(data)}\n")
        f.write(f"Total population = {weights.sum():,.0f}\n")
        f.write(f"Using real travel time matrix: {solver.n}x{solver.n}\n")
        f.write(f"Objective (weighted total travel time) = {objective:.2f} seconds\n\n")

        f.write("Selected Sites:\n")
        f.write(f"{'-' * 60}\n")
        for i, fac in enumerate(facilities):
            r = data['row'].values[fac]
            c = data['col'].values[fac]
            pop_assigned = weights[assignments == i].sum()
            n_assigned = (assignments == i).sum()
            avg_time = travel_times_min[assignments == i].mean()
            lon = (data['lon'].values[fac]
                   if 'lon' in data.columns else 'N/A')
            lat = (data['lat'].values[fac]
                   if 'lat' in data.columns else 'N/A')
            f.write(f"  Site {i + 1}: grid({r},{c}), "
                    f"serves {n_assigned} cells, pop={pop_assigned:,.0f}, "
                    f"avg travel={avg_time:.2f}min, "
                    f"coord=({lon}, {lat})\n")

        f.write("\nCoverage Statistics:\n")
        f.write(f"{'-' * 60}\n")
        for threshold in [1, 2, 3, 5, 10]:
            covered = weights[travel_times_min <= threshold].sum()
            pct = covered / weights.sum() * 100
            f.write(f"  Within {threshold}min: "
                    f"{pct:.1f}% population covered\n")

    print(f"  Saved: {summary_path}")
    return result_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description='P-Median Model for UAM Site Selection (Global Optimum)'
    )
    parser.add_argument('--p', type=int, default=None,
                        help='Number of sites P (default: from config)')
    parser.add_argument('--method', type=str, default='ilp',
                        choices=['lagrangian', 'gurobi', 'ilp', 'ga',
                                 'greedy'],
                        help='Solver method (default: ilp for exact solution)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to real data CSV')
    parser.add_argument('--K', type=int, default=None,
                        help='K-nearest neighbors for ILP/Gurobi')
    parser.add_argument('--time-limit', type=int, default=None,
                        help='Solver time limit in seconds')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Enable GPU acceleration (requires CuPy)')
    parser.add_argument('--multi-p', type=str, default=None,
                        help='Multi-P analysis, format: "2-10"')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    p_value = args.p or P_DEFAULT

    print("=" * 60)
    print("  P-Median Model for Low-Altitude Economy Site Selection")
    print("  (Exact ILP / Lagrangian Relaxation)")
    print("=" * 60)

    if args.data:
        data = load_real_data(args.data)
    else:
        print("Error: --data argument is required. Please provide real data CSV.")
        print("Usage: python main.py --data data/yangpu_grids_real.csv --p 5 --method lagrangian")
        sys.exit(1)

    print(f"\nData overview:")
    print(f"  Grids: {len(data)}")
    print(f"  Total population: {data['population'].sum():,}")
    print(f"  Non-zero pop cells: {(data['population'] > 0).sum()}")
    print(f"  Max pop in cell: {data['population'].max():,}")

    # Load time matrix for real data
    time_matrix = None
    if args.data:
        time_matrix = load_time_matrix(grid_ids=data['grid_id'].values)

    if args.multi_p:
        parts = args.multi_p.split('-')
        p_start, p_end = int(parts[0]), int(parts[1])
        p_range = range(p_start, p_end + 1)
        print(f"\nMulti-P analysis: P={list(p_range)}")
        plot_objective_vs_p(data, p_range, time_matrix=time_matrix)
    else:
        solver = PMedianSolver(data, p_value=p_value, time_matrix=time_matrix,
                               use_gpu=args.gpu)

        kwargs = {}
        if args.method == 'lagrangian':
            if args.time_limit:
                kwargs['time_limit'] = args.time_limit
        elif args.method in ('gurobi', 'ilp'):
            if args.K:
                kwargs['K'] = args.K
            if args.time_limit:
                kwargs['time_limit'] = args.time_limit
        elif args.method == 'ga':
            kwargs['seed'] = args.seed

        facilities, assignments, objective = solver.solve(
            method=args.method, verbose=True, **kwargs
        )

        print("\n" + "=" * 60)
        print("[Stage 3] Generating visualizations...")
        print("=" * 60)

        plot_population_density(data, facilities, p_value=p_value)
        plot_service_areas(data, facilities, assignments, p_value=p_value)
        plot_travel_time_map(data, facilities, time_matrix=time_matrix, p_value=p_value)
        plot_coverage_analysis(data, facilities, time_matrix=time_matrix, p_value=p_value)
        plot_html_map(data, facilities, assignments,
                      title=f"P-Median Site Selection (P={p_value}) - Yangpu, Shanghai",
                      p_value=p_value)

        print("\n[Stage 4] Saving results...")
        save_results(data, facilities, assignments, objective, p_value,
                     args.method, time_matrix=time_matrix)

    print("\n" + "=" * 60)
    print("  All tasks completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
