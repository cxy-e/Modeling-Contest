# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import folium
import json
import webbrowser

# %% [markdown]
# grid定义

# %%
class Grid:
    def __init__(self, lon, lat, gridId ):
        self.lon = lon
        self.lat = lat
        self.gridId = gridId
        self.demand = 0     #网格配送需求
        self.stationCost = 0    #建配送站费用
        self.height = 0     #建筑最大高度
        self.weatherCost = 0    #天气消耗
        self.isStation = 0      #是否为配送站
    def __repr__(self):
        return f"grid({self.gridId})"
    def setDemand(self, demand):
        self.demand = demand
    def setStationCost(self, stationCost):
        self.stationCost = stationCost
    def setHeight(self, height):
        self.height = height
    def setWeatherCost(self, weatherCost):
        self.weatherCost = weatherCost
    def setStation(self):
        self.isStation = 1

# %% [markdown]
# 划分方格

# %%
def createYangpuGrid(lonMin = 121.48, lonMax = 121.565, latMin = 31.25, latMax = 31.348, gridSize = 200):
    # 坐标系转换
    wgs84 = CRS("EPSG:4326")
    cgcs_sh = CRS("EPSG:3857")
    trans = Transformer.from_crs(wgs84, cgcs_sh, always_xy=True)
    back = Transformer.from_crs(cgcs_sh, wgs84, always_xy=True)

    # 杨浦区边界
    # lonMin, lonMax = 121.46, 121.58   # 经度
    # latMin, latMax = 31.23, 31.31     # 纬度

    # 转平面坐标
    xMin, yMin = trans.transform(lonMin, latMin)
    xMax, yMax = trans.transform(lonMax, latMax)

    #范围内xy坐标
    xs = np.arange(xMin, xMax, gridSize)   # 横向x
    ys = np.arange(yMin, yMax, gridSize)   # 纵向y

    # 生成grid
    grids = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cx = x + gridSize / 2
            cy = y + gridSize / 2
            lon, lat = back.transform(cx, cy)

            grid_ = Grid(round(lon, 6), round(lat, 6), f"{i}_{j}")
            grids.append(grid_)
    return grids
YangPuGrids = createYangpuGrid()

# %% [markdown]
# 格子可视化

# %%
def visualizeGrids(grids, gridSize=200, zoomStart = 13, centerLat = 31.27, centerLon = 121.53):
    m = folium.Map(location = [centerLat, centerLon], zoom_start = zoomStart, tiles='OpenStreetMap')

    wgs84 = CRS("EPSG:4326")
    cgcs_sh = CRS("EPSG:3857")
    trans = Transformer.from_crs(wgs84, cgcs_sh, always_xy=True)
    back = Transformer.from_crs(cgcs_sh, wgs84, always_xy=True)

    with open("../data/yangpu.json", 'r', encoding='utf-8') as f:
        yangpuJson = json.load(f)

    folium.GeoJson(
        yangpuJson,
        style_function=lambda feature:{
            'color': 'black',
            'weight': 2,
            'fill': False
        }
    ).add_to(m)

    for g in grids:
        cx, cy = trans.transform(g.lon, g.lat)
        half = gridSize / 2
        corners = [
            (cx - half, cy + half),
            (cx + half, cy + half),
            (cx + half, cy - half),
            (cx - half, cy - half),
        ]
        poly = [back.transform(x, y)[::-1] for x, y in corners]
        folium.Polygon(
            locations=poly,
            color='red',
            weight=1,
            fill=True,
            fillcolor='lightred',
            fillOpacity=0.1,
            popup=f'Grid ID:{g.gridId}'
        ).add_to(m)
        folium.CircleMarker(
            location=[g.lat, g.lon],    #?
            radius=1,
            color='red',
            fill=True,
            fillColor='red'
        )
    return m

mapPic = visualizeGrids(YangPuGrids)
mapPic.save("addGrids.html")
webbrowser.open("addGrids.html")