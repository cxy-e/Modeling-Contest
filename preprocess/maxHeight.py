import pandas as pd
from tqdm import tqdm

tqdm.pandas()

build = pd.read_csv("../data/heightGrid.csv", encoding = "utf-8")
grid = pd.read_excel("../data/yangpuGrids.xlsx")
maxHeight = build.groupby("grid")["Height"].max().reset_index()
maxHeight.columns = ["gridId", "maxHeight"]
grid["maxHeight"] = 0.0
grid.set_index("gridId", inplace=True)
grid = grid.merge(maxHeight, on="gridId", how="left", suffixes=("", "_new"))
grid.reset_index(inplace=True)

grid.to_excel("../data/yangpuGrids_.xlsx", index=False)