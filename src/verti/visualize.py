import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from config import GRID_SIZE, GRID_SPACING, OUTPUT_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def _get_grid_dims(data):
    rows = data['row'].values.astype(int)
    cols = data['col'].values.astype(int)
    n_rows = rows.max() + 1
    n_cols = cols.max() + 1
    return n_rows, n_cols


def _reshape_to_grid(data, values):
    n_rows, n_cols = _get_grid_dims(data)
    grid = np.full((n_cols, n_rows), np.nan)
    rows = data['row'].values.astype(int)
    cols = data['col'].values.astype(int)
    grid[cols, rows] = values
    return grid


def plot_population_density(data, facilities=None, output_path=None, title=None, p_value=None):
    print("  [3a] Plotting population density map...")
    n_rows, n_cols = _get_grid_dims(data)
    pop_grid = _reshape_to_grid(data, data['population'].values)

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    cmap = LinearSegmentedColormap.from_list(
        'pop_density',
        ['#eef1fc', '#c8d4f8', '#9bb5f2', '#6d96ec',
         '#4368eb', '#3355cc', '#2440a8', '#152b7a'],
        N=256
    )

    im = ax.imshow(pop_grid, cmap=cmap, origin='lower', aspect='equal',
                   interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='人口')
    cbar.ax.tick_params(labelsize=9)

    if facilities is not None and len(facilities) > 0:
        fac_rows = data['row'].values[facilities]
        fac_cols = data['col'].values[facilities]
        ax.scatter(fac_rows, fac_cols, c='#ff61a2', s=350, marker='*',
                   edgecolors='white', linewidths=1.5, zorder=10,
                   label=f'选定站点 (P={len(facilities)})')
        for i, (r, c) in enumerate(zip(fac_rows, fac_cols)):
            ax.annotate(f'P{i + 1}', (r, c), textcoords="offset points",
                        xytext=(8, 8), fontsize=10, fontweight='bold', color='#ff61a2')

    ax.set_xlabel('行（东西/经度）', fontsize=12)
    ax.set_ylabel('列（南北/纬度）', fontsize=12)

    if facilities is not None and len(facilities) > 0:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9,
                  edgecolor='#cccccc')

    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_cols - 0.5)

    plt.tight_layout()

    if output_path is None:
        p_str = f"_P{p_value}" if p_value is not None else ""
        output_path = os.path.join(OUTPUT_DIR, f'population_density_sites{p_str}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       Saved: {output_path}")
    return output_path


def plot_service_areas(data, facilities, assignments, output_path=None, p_value=None):
    print("  [3b] Plotting service area map...")
    n_rows, n_cols = _get_grid_dims(data)
    area_grid = np.full((n_cols, n_rows), -1)
    rows = data['row'].values.astype(int)
    cols = data['col'].values.astype(int)
    area_grid[cols, rows] = assignments

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    n_fac = len(facilities)
    cmap = plt.cm.get_cmap('tab20', n_fac)

    area_masked = np.ma.masked_where(area_grid < 0, area_grid)
    im = ax.imshow(area_masked, cmap=cmap, origin='lower', aspect='equal',
                   interpolation='nearest', vmin=0, vmax=n_fac - 1)

    fac_rows = data['row'].values[facilities]
    fac_cols = data['col'].values[facilities]
    ax.scatter(fac_rows, fac_cols, c='#ff61a2', s=350, marker='*',
               edgecolors='white', linewidths=2, zorder=10)

    for i, (r, c) in enumerate(zip(fac_rows, fac_cols)):
        ax.annotate(f'P{i + 1}', (r, c), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight='bold', color='#ff61a2')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_fac))
    cbar.ax.set_yticklabels([f'站点 {i + 1}' for i in range(n_fac)])
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('行（东西/经度）', fontsize=12)
    ax.set_ylabel('列（南北/纬度）', fontsize=12)

    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_cols - 0.5)

    plt.tight_layout()

    if output_path is None:
        p_str = f"_P{p_value}" if p_value is not None else ""
        output_path = os.path.join(OUTPUT_DIR, f'service_areas{p_str}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       Saved: {output_path}")
    return output_path


def plot_travel_time_map(data, facilities, time_matrix=None, output_path=None, max_time_min=10, p_value=None):
    print("  [3c] Plotting travel time map...")
    from solver import PMedianSolver

    solver = PMedianSolver(data, p_value=len(facilities), time_matrix=time_matrix)
    _, nearest_time = solver._compute_assignment(facilities)
    travel_times_min = nearest_time / 60.0

    n_rows, n_cols = _get_grid_dims(data)
    time_grid = _reshape_to_grid(data, travel_times_min)

    fig, ax = plt.subplots(1, 1, figsize=(8, 12))

    cmap_time = LinearSegmentedColormap.from_list(
        'travel_time',
        ['#eef1fc', '#9bb5f2', '#4368eb', '#3355cc',
         '#f39c12', '#e74c3c', '#b10026'],
        N=256
    )
    im = ax.imshow(time_grid, cmap=cmap_time, origin='lower', aspect='equal',
                   interpolation='nearest', vmin=0, vmax=max_time_min)

    fac_rows = data['row'].values[facilities]
    fac_cols = data['col'].values[facilities]
    ax.scatter(fac_rows, fac_cols, c='#ff61a2', s=300, marker='*',
               edgecolors='white', linewidths=1.5, zorder=10)
    for i, (r, c) in enumerate(zip(fac_rows, fac_cols)):
        ax.annotate(f'P{i + 1}', (r, c), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight='bold', color='#ff61a2')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='出行时间（分钟）')
    ax.set_xlabel('行（东西/经度）', fontsize=12)
    ax.set_ylabel('列（南北/纬度）', fontsize=12)

    ax.set_xlim(-0.5, n_rows - 0.5)
    ax.set_ylim(-0.5, n_cols - 0.5)

    plt.tight_layout()

    if output_path is None:
        p_str = f"_P{p_value}" if p_value is not None else ""
        output_path = os.path.join(OUTPUT_DIR, f'travel_time_map{p_str}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       Saved: {output_path}")
    return output_path


def plot_coverage_analysis(data, facilities, time_matrix=None, output_path=None, max_time_min=10, p_value=None):
    print("  [3d] Plotting coverage analysis...")
    from solver import PMedianSolver

    solver = PMedianSolver(data, p_value=len(facilities), time_matrix=time_matrix)
    _, nearest_time = solver._compute_assignment(facilities)
    travel_times_min = nearest_time / 60.0

    weights = data['population'].values.astype(float)
    total_pop = weights.sum()

    time_bins = np.arange(0, max_time_min + 0.5, 0.5)
    coverage_pct = []
    for t in time_bins:
        covered = weights[travel_times_min <= t].sum()
        coverage_pct.append(covered / total_pop * 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(time_bins, coverage_pct, color='#4368eb', linewidth=2.5)
    ax.fill_between(time_bins, coverage_pct, alpha=0.15, color='#4368eb')
    ax.set_xlabel('出行时间（分钟）', fontsize=12)
    ax.set_ylabel('人口覆盖率（%）', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_time_min)
    ax.set_ylim(0, 105)

    for threshold in [1, 3, 5]:
        idx = np.searchsorted(time_bins, threshold)
        if idx < len(coverage_pct):
            ax.axvline(x=threshold, color='#e74c3c', linestyle='--', alpha=0.6)
            ax.annotate(f'{threshold}分钟: {coverage_pct[idx]:.1f}%',
                        xy=(threshold, coverage_pct[idx]),
                        xytext=(threshold + 0.3, coverage_pct[idx] - 5),
                        fontsize=9, color='#e74c3c', fontweight='bold')

    plt.tight_layout()

    if output_path is None:
        p_str = f"_P{p_value}" if p_value is not None else ""
        output_path = os.path.join(OUTPUT_DIR, f'coverage_analysis{p_str}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       Saved: {output_path}")
    return output_path


def plot_objective_vs_p(data, p_range, time_matrix=None, output_path=None):
    print("  [3e] Plotting objective vs P...")
    from solver import PMedianSolver

    objectives = []
    for p in p_range:
        print(f"       Solving P={p}...")
        solver = PMedianSolver(data, p_value=p, time_matrix=time_matrix)
        _, _, obj = solver.solve_greedy(verbose=False)
        objectives.append(obj)
        print(f"       P={p}: objective={obj:.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(list(p_range), objectives, color='#4368eb', linewidth=2.5,
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor='#4368eb', markeredgewidth=2)
    ax.set_xlabel('站点数量 (P)', fontsize=12)
    ax.set_ylabel('总加权出行时间', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    if len(objectives) >= 2:
        marginal = [objectives[i] - objectives[i + 1] for i in range(len(objectives) - 1)]
        ax2 = ax.twinx()
        ax2.bar(list(p_range)[:-1], marginal, alpha=0.3, color='#4368eb', width=0.6,
                label='边际改善')
        ax2.set_ylabel('边际改善', fontsize=12, color='#4368eb')
        ax2.tick_params(axis='y', labelcolor='#4368eb')

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'objective_vs_p.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"       Saved: {output_path}")
    return output_path
