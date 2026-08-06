import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Real travel time matrix
TIME_MATRIX_PATH = os.path.join(DATA_DIR, 'time_matrix.csv')

GRID_SIZE = 100
GRID_SPACING = 200
GRID_SIZE_REAL = (40, 54)
GRID_SPACING_LON_M = 197.16
GRID_SPACING_LAT_M = 202.08
P_DEFAULT = 5
UAM_SPEED_KMH = 150
K_NEAREST_INIT = 200
SEED = 42

TOTAL_POPULATION = 1_240_000

YANGPU_LON_MIN = 121.48
YANGPU_LON_MAX = 121.565
YANGPU_LAT_MIN = 31.25
YANGPU_LAT_MAX = 31.348

POPULATION_CENTERS = [
    {'name': 'wujiaochang', 'row_frac': 0.55, 'col_frac': 0.50, 'weight': 0.18, 'sigma': 0.06},
    {'name': 'south_riverside', 'row_frac': 0.72, 'col_frac': 0.42, 'weight': 0.16, 'sigma': 0.08},
    {'name': 'fudan', 'row_frac': 0.48, 'col_frac': 0.58, 'weight': 0.10, 'sigma': 0.05},
    {'name': 'huangxing_park', 'row_frac': 0.58, 'col_frac': 0.35, 'weight': 0.13, 'sigma': 0.07},
    {'name': 'north_industrial', 'row_frac': 0.30, 'col_frac': 0.50, 'weight': 0.05, 'sigma': 0.10},
    {'name': 'east_residential', 'row_frac': 0.52, 'col_frac': 0.70, 'weight': 0.08, 'sigma': 0.06},
    {'name': 'central_residential', 'row_frac': 0.62, 'col_frac': 0.52, 'weight': 0.12, 'sigma': 0.08},
    {'name': 'west_residential', 'row_frac': 0.55, 'col_frac': 0.28, 'weight': 0.07, 'sigma': 0.07},
    {'name': 'tongji', 'row_frac': 0.50, 'col_frac': 0.65, 'weight': 0.06, 'sigma': 0.05},
    {'name': 'background', 'row_frac': 0.50, 'col_frac': 0.50, 'weight': 0.05, 'sigma': 0.20},
]

ILP_TIME_LIMIT = 7200
ILP_THREADS = 0

LAGRANGIAN_MAX_ITER = 100000
LAGRANGIAN_TIME_LIMIT = 7200
LAGRANGIAN_GAP_TOL = 0.001

GA_POP_SIZE = 300
GA_GENERATIONS = 800
GA_MUTATION_RATE = 0.15
GA_ELITE_RATIO = 0.1
GA_TOURNAMENT_SIZE = 5
GA_LOCAL_SEARCH_ITER = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
