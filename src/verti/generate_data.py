import os
import numpy as np
import pandas as pd
from config import (
    GRID_SIZE, GRID_SPACING, GRID_SIZE_REAL,
    GRID_SPACING_LON_M, GRID_SPACING_LAT_M,
    TOTAL_POPULATION,
    POPULATION_CENTERS, YANGPU_LON_MIN, YANGPU_LON_MAX,
    YANGPU_LAT_MIN, YANGPU_LAT_MAX, DATA_DIR, SEED,
    TIME_MATRIX_PATH
)


def load_time_matrix(csv_path=None, grid_ids=None):
    """
    Load travel time matrix from CSV file.

    Args:
        csv_path: Path to time matrix CSV file
        grid_ids: List of grid IDs to ensure correct ordering

    Returns:
        numpy array of shape (N, N) where N is number of grids
    """
    csv_path = csv_path or TIME_MATRIX_PATH

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Time matrix file not found: {csv_path}")

    print(f"[Stage 0] Loading time matrix from: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    if 'gridId' not in df.columns:
        raise ValueError("Time matrix CSV must have 'gridId' as first column")

    n_grids = len(df)
    print(f"  Time matrix size: {n_grids}x{n_grids}")

    df_indexed = df.set_index('gridId')

    if grid_ids is not None:
        grid_ids_str = [str(gid) for gid in grid_ids]
        missing = set(grid_ids_str) - set(df_indexed.index.astype(str))
        if missing:
            raise ValueError(f"Grid IDs not found in time matrix: {missing}")
        time_matrix = df_indexed.loc[grid_ids_str, grid_ids_str].values.astype(np.float64)
    else:
        time_matrix = df_indexed.values.astype(np.float64)

    diag = np.diag(time_matrix)
    nonzero_diag = np.sum(diag > 0.01)
    if nonzero_diag > 0:
        print(f"  [WARNING] {nonzero_diag} diagonal entries are non-zero (self-travel > 0)")

    print(f"  Time matrix loaded successfully")
    print(f"  Min time: {time_matrix.min():.2f}s, Max time: {time_matrix.max():.2f}s")
    print(f"  Mean time: {time_matrix.mean():.2f}s")

    return time_matrix


def generate_sim_data(seed=SEED):
    print("=" * 60)
    print("[Stage 1] Generating simulated grid data...")
    print("=" * 60)
    np.random.seed(seed)

    N = GRID_SIZE * GRID_SIZE
    row_indices, col_indices = np.meshgrid(
        np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij'
    )
    row_flat = row_indices.ravel().astype(float)
    col_flat = col_indices.ravel().astype(float)

    center_x = (col_flat + 0.5) * GRID_SPACING
    center_y = (row_flat + 0.5) * GRID_SPACING

    population = np.zeros(N)

    for center in POPULATION_CENTERS:
        r_center = center['row_frac'] * GRID_SIZE
        c_center = center['col_frac'] * GRID_SIZE
        sigma = center['sigma'] * GRID_SIZE
        weight = center['weight']

        dr = row_flat - r_center
        dc = col_flat - c_center
        gaussian = np.exp(-(dr ** 2 + dc ** 2) / (2 * sigma ** 2))
        population += weight * gaussian

    boundary_r = np.minimum(row_flat, GRID_SIZE - 1 - row_flat) / (GRID_SIZE * 0.15)
    boundary_c = np.minimum(col_flat, GRID_SIZE - 1 - col_flat) / (GRID_SIZE * 0.15)
    boundary_factor = np.clip(np.minimum(boundary_r, boundary_c), 0, 1)
    population *= boundary_factor

    if population.sum() > 0:
        population = population / population.sum() * TOTAL_POPULATION

    noise = np.random.lognormal(0, 0.3, size=N)
    population = population * noise
    population = np.maximum(population, 0)
    population = np.round(population).astype(int)

    lon_range = YANGPU_LON_MAX - YANGPU_LON_MIN
    lat_range = YANGPU_LAT_MAX - YANGPU_LAT_MIN
    grid_width_m = GRID_SIZE * GRID_SPACING
    grid_height_m = GRID_SIZE * GRID_SPACING

    center_lon = YANGPU_LON_MIN + (center_x / grid_width_m) * lon_range
    center_lat = YANGPU_LAT_MIN + (center_y / grid_height_m) * lat_range

    df = pd.DataFrame({
        'grid_id': np.arange(N),
        'row': row_flat.astype(int),
        'col': col_flat.astype(int),
        'center_x': center_x,
        'center_y': center_y,
        'center_lon': np.round(center_lon, 6),
        'center_lat': np.round(center_lat, 6),
        'population': population,
    })

    filepath = os.path.join(DATA_DIR, 'simulated_grid_data.csv')
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE} = {N} cells")
    print(f"  Grid spacing: {GRID_SPACING}m")
    print(f"  Total population: {df['population'].sum():,}")
    print(f"  Non-zero cells: {(df['population'] > 0).sum()}")
    print(f"  Max population in cell: {df['population'].max():,}")
    print(f"  Avg population (non-zero): {df[df['population'] > 0]['population'].mean():.0f}")
    print(f"  Data saved: {filepath}")
    print()

    return df


def load_real_data(csv_path):
    print(f"[Stage 1] Loading real data: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Handle different column naming conventions
    if 'gridId' in df.columns:
        df = df.rename(columns={'gridId': 'grid_id'})
    if 'index' in df.columns and 'grid_id' not in df.columns:
        df = df.rename(columns={'index': 'grid_id'})

    required = ['grid_id', 'population', 'row', 'col']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values(['row', 'col']).reset_index(drop=True)

    actual_rows = df['row'].max() + 1
    actual_cols = df['col'].max() + 1
    print(f"  Grid dimensions from data: {actual_rows} rows x {actual_cols} cols")

    if 'maxHeight' in df.columns:
        print(f"  maxHeight available: {df['maxHeight'].notna().sum()} non-null values")

    print(f"  Total population: {df['population'].sum():,}")
    print(f"  Non-zero cells: {(df['population'] > 0).sum()}")
    print(f"  Max population in cell: {df['population'].max():,}")

    return df


if __name__ == '__main__':
    df = generate_sim_data()
