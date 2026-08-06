import numpy as np
from a_star import astar
import pandas as pd
import time


a = np.arange(7, 20)
b = np.arange(21, 36)

grid_ids = []
for x in a:
    for y in b:
        grid_ids.append(f"{x}_{y}")
time_matrix = []
runtime_matrix = []
total = ((19-7) * (35-21)) ** 2
count = 0
for start_id in grid_ids:   # 行名对应起点
    row = []
    runtime_row = []
    for end_id in grid_ids:
        if start_id == end_id:
           row.append(0)
           runtime_row.append(0)
           continue
        s = time.time()
        t = astar(start_id, end_id)
        c = time.time() - s
        row.append(round(t, 3))
        runtime_row.append(round(c, 3))
        count += 1
        percent = count / total * 100
        print(f"\rProgress: {percent:.1f}% ({count}/{total}) at {start_id}, {end_id}; t = {round(t, 3)}", end = "\r")
    time_matrix.append(row)
    runtime_matrix.append(runtime_row)

df_time = pd.DataFrame(
    time_matrix,
    index = grid_ids,
    columns = grid_ids
)

df_time.to_excel("matrix_again.xlsx", index = True)
df_runtime = pd.DataFrame(runtime_matrix, index=grid_ids, columns=grid_ids)
df_runtime.to_excel("again.xlsx", index=True)