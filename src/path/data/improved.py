import heapq
import re
import pandas as pd

SAFE_DISTANCE = 12  #与建筑高度的安全距离
INIT_HEIGHT = 65

df = pd.read_excel("../data/yangpuGrids.xlsx")
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

def move(x, y, x_, y_, curHeight):    #xy->x_y_
    #返回任意点间直线移动时穿过边界时的坐标, 以及时长
    time = 0.0
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
        dHeight = nextHeight - curHeight
        if abs(dHeight / step_x) <= 0.3:
            time += (step_x**2 + step_y**2 + dHeight**2)**0.5/15.5
        else:
            time += max((step_x**2 + step_y**2)/11, dHeight/3)
        curHeight = nextHeight
        curHeight = float(curHeight)
    time += ((x - x_)**2 + (y - y_)**2)**0.5/15.5
    return time, curHeight

dirs = [    # 若不能直线飞或直线飞开销太大时的备选方向
    (-200, 0),
    (200, 0),
    (0, -200),
    (0, 200),
    (-200, -200),
    (-200, 200),
    (200, -200),
    (200, 200),
]

def astar(start, end):
    if (getHeight(end) + SAFE_DISTANCE) > 115:
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
        if abs(x - endX) <= 100 and abs(y - endY) <= 100:   # 达到终点
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
            time, nh = move(x, y, nx, ny, hDrone)
            if nh > 115 or (nh - hDrone) > 25:
                continue
            newC = cost[(x, y, hDrone)] + time
            if(nx, ny, nh) not in cost or newC < cost[(nx, ny, nh)]:
                cost[(nx, ny, nh)] = newC
                tc, te = move(nx, ny, endX, endY, hDrone)
                nf = newC + tc
                heapq.heappush(heap, (nf, nx, ny, nh))
    return t