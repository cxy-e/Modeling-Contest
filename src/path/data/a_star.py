import numpy as np
import heapq
import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

SAFE_DISTANCE = 12  #与建筑高度的安全距离
INIT_HEIGHT = 65

df = pd.read_excel("../data/yangpuGrids_.xlsx")
df["maxHeight"] = df["maxHeight"].fillna(0)

def id_xy(gridId):      # grid_Id转成坐标
    n = re.findall(r'\d+', gridId)
    x = int(n[0]) * 200 + 100
    y = int(n[1]) * 200 + 100
    return x, y

def xy_grid(x, y):      # 坐标转成所在格子id
    a = int(x//200)
    b = int(y//200)
    return f"{a}_{b}", a, b

def getHeight(gridId):      # 返回格子最大高度
    a = df.loc[df["gridId"] == gridId, "maxHeight"]
    return a.values[0] if not a.empty else 1111

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

def eu(x, y, x_, y_):   # 欧距
    t, _, _ = xy_grid(x, y)
    t_, _, _ = xy_grid(x_, y_)
    dh = max(getHeight(t) - getHeight(t_), 0)
    return ((x-x_)**2+(y-y_)**2+dh**2)**0.5

def astar(start, end):
    if getHeight(end) + SAFE_DISTANCE > 115:
        return -1
    startX, startY = id_xy(start)
    endX, endY = id_xy(end)

    heap = []
    hDrone = max(INIT_HEIGHT, getHeight(start) + SAFE_DISTANCE)
    heapq.heappush(heap, (0.0, startX, startY, hDrone))

    cost = {}
    cost[(startX, startY, hDrone)] = 0.0

    t = 0.0
    traceX, traceY, traceH = 1, 1, 1
    while heap:
        f, x, y, hDrone = heapq.heappop(heap)
        if x == endX and y == endY:   # 达到终点
            t = f
            traceX, traceY, traceH = x, y, hDrone
            break
        dirs_ = dirs
        for dx, dy in dirs_:
            nx = x + dx
            ny = y + dy
            t_, _, _ = xy_grid(nx, ny)
            nh = max(getHeight(t_) + SAFE_DISTANCE, hDrone)
            if nh > 115 or (nh - hDrone) > 25:
                continue
            dh = max(nh - hDrone, 0)
            tan = dh / (dx**2 + dy**2)**0.5
            if(tan < 0.3):
                newC = cost[(x, y, hDrone)] + (dx**2 + dy**2 + dh**2)**0.5/15.5
            else:
                newC = cost[(x, y, hDrone)] + max((dx**2 + dy**2)**0.5/11, dh/3)
            if(nx, ny, nh) not in cost or newC < cost[(nx, ny, nh)]:
                cost[(nx, ny, nh)] = newC
                nf = newC + eu(nx, ny, endX, endY)/15.5
                heapq.heappush(heap, (nf, nx, ny, nh))
    return t