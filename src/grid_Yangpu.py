import numpy as np
import heapq
import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

SAFE_DISTANCE = 12
INIT_HEIGHT = 65

df = pd.read_excel("../data/yangpuGrids_.xlsx")
df["maxHeight"] = df["maxHeight"].fillna(0)

def id_xy(gridId):
    n = re.findall(r'\d+', gridId)
    x = int(n[0]) * 200 + 100
    y = int(n[1]) * 200 + 100
    return x, y

def xy_grid(x, y):
    a = int(x//200)
    b = int(y//200)
    return f"{a}_{b}", a, b

def getHeight(gridId):
    a = df.loc[df["gridId"] == gridId, "maxHeight"]
    return a.values[0] if not a.empty else 1111

def move(x, y, x_, y_, curHeight):    #xy->x_y_
    #返回穿过边界时的坐标，以及对应的高度, 以及时长
    time = 0.0
    xyz = []
    while True:
        dx = x_ - x
        dy = y_ - y
        tan = dy / dx if dx != 0 else float('inf')
        cur_grid_id, cur_grid_x, cur_grid_y = xy_grid(x, y)
        next_grid_x = 0 if dx == 0 else((cur_grid_x + 1) * 200 if dx > 0 else (cur_grid_x - 1) * 200)
        next_grid_y = 0 if dy == 0 else((cur_grid_y + 1) * 200 if dy > 0 else (cur_grid_y - 1) * 200)
        step_x = next_grid_x - x
        step_y = next_grid_y - y
        if(abs(step_x) > abs(dx) or abs(step_y) > abs(dy)):
            break
        if abs(step_x * tan) < abs(step_y):
            step_y = step_x * tan
        elif abs(step_x * tan) > abs(step_y):
            step_x = step_y / tan
        x += step_x
        y += step_y
        x = round(x, 2)
        y = round(y, 2)
        tpid, tpx, tpy = xy_grid(x, y)
        nextHeight = max(getHeight(tpid)+SAFE_DISTANCE, curHeight)
        time += (step_x**2 + step_y**2 + (nextHeight - curHeight)**2)**0.5/15.5
        curHeight = nextHeight
        curHeight = float(curHeight)
        xyz.append((x, y, curHeight))
    xyz.append((x_, y_, curHeight))
    time += ((x - x_)**2 + (y - y_)**2)**0.5/15.5
    return xyz, time, curHeight

# x, t, c = move(0, 0, 600, 600, 65)
# print(x, t, c)
dirs = [
    (-200, 0),
    (200, 0),
    (0, -200),
    (0, 200),
    (-200, -200),
    (-200, 200),
    (200, -200),
    (200, 200),
]

def eu(x, y, x_, y_):
    dh = abs(getHeight(f"{x}_{y}") - getHeight(f"{x_}_{y_}"))
    return ((x-x_)**2+(y-y_)**2+dh**2)**0.5

def astar(start, end):
    startX, startY = id_xy(start)
    endX, endY = id_xy(end)

    heap = []
    hDrone = max(INIT_HEIGHT, getHeight(start) + SAFE_DISTANCE)
    heapq.heappush(heap, (0.0, startX, startY, hDrone))

    cameFrom = {}
    cost = {}

    cameFrom[(startX, startY, hDrone)] = (-1, -1, -1)
    cost[(startX, startY, hDrone)] = 0.0

    t = 0.0
    traceX, traceY, traceH = 1, 1, 1
    while heap:
        f, x, y, hDrone = heapq.heappop(heap)
        if abs(x - endX) <= 100 and abs(y - endY) <= 100:   # 达到终点
            print("a")
            t = f
            traceX, traceY, traceH = x, y, hDrone
            break
        dirs_ = dirs
        if x != endX and y != endY:
            tan = (y - endY)/(x - endX)
            side = (tan ** 2 + 1)**0.5
            dirs_ = dirs_ + [(200 / side * (-(x - endX)/abs(x - endX)), 200 * tan / side * (-(y - endY)/abs(y - endY)))]
        for dx, dy in dirs_:
            nx = x + dx
            ny = y + dy
            xyz, time, nh = move(x, y, nx, ny, hDrone)
            if nh > 115 or (nh - hDrone) > 25:
                continue
            newC = cost[(x, y, hDrone)] + time
            if(nx, ny, nh) not in cost or newC < cost[(nx, ny, nh)]:
                cameFrom[xyz[0]] = (x, y, hDrone)
                itr = len(xyz) - 1
                while(itr >= 1):
                    cameFrom[xyz[itr]] = xyz[itr - 1]
                    itr -= 1
                cost[(nx, ny, nh)] = newC
                nf = newC + eu(nx, ny, endX, endY)/15.5
                heapq.heappush(heap, (nf, nx, ny, nh))
    path = []
    while (traceX, traceY, traceH) in cameFrom:
        path.append([traceX, traceY, traceH])
        traceX, traceY, traceH = cameFrom[(traceX, traceY, traceH)]
    path.reverse()
    return t, path

t, path = astar("6_11", "11_25")
print(t)
print(path)
def plot_2d_path(path):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, c='blue', linewidth=2.5, label='Flight Path')
    plt.scatter(xs[0], ys[0], c='red', s=120, zorder=5, label='Start')
    plt.scatter(xs[-1], ys[-1], c='green', s=120, zorder=5, label='End')
    endX, endY = id_xy("22_25")
    plt.scatter(endX, endY, c='pink', s=120, zorder=5, label='Desti')

    plt.grid(True, alpha=0.3)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('UAV Path Planning (2D Top View)')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_2d_path(path)