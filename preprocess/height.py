import pandas as pd
from shapely.wkt import loads
from tqdm import tqdm

tqdm.pandas()

build = pd.read_csv("../data/test-select.csv", encoding = "utf-8")
grid = pd.read_excel("../data/yangpuGrids.xlsx")

def getCenter(p):
    try:
        poly = loads(p)
        return poly.centroid.x, poly.centroid.y
    except:
        return None, None

build[["lon", "lat"]] = build["WKT"].progress_apply(
    lambda g: pd.Series(getCenter(g))
)

build = build.dropna(subset=["lon", "lat"])

def findGrid(lon, lat):
    gridMatch = grid[
        (grid["minLon"] <= lon) &
        (grid["maxLon"] >= lon) &
        (grid["minLat"] <= lat) &
        (grid["maxLat"] >= lat)
    ]
    if gridMatch.empty:
        return None
    return gridMatch['gridId'].iloc[0]

build["grid"] = build[["lon", "lat"]].progress_apply(
    lambda r: findGrid(r["lon"], r["lat"]), axis=1)

build = build.dropna(subset = ["grid"])

build.to_csv("../data/heightGrid.csv", index=False, encoding="utf-8-sig")