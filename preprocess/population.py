import pandas as pd
import numpy as np

# ===================== 配置路径，直接运行无需修改 =====================
GRID_FILE = "../data/yangpuGridsHeight.xlsx"    # 原始网格模板
POINT_CSV = "../data/test8.csv"                 # 人口点数据
OUTPUT_FILE = "../data/yangpuGridsHeight_with_pop.xlsx"  # 输出带population的完整网格表

# 1. 读取原始网格模板（包含gridId、lon、lat、minLon、maxLon、minLat、maxLat、maxHeight）
grid_df = pd.read_excel(GRID_FILE)
print(f"读取网格模板，总网格数：{len(grid_df)}")

# 2. 读取人口点位数据（longitude, latitude, population）
point_df = pd.read_csv(POINT_CSV)
print(f"读取人口点位，总点数：{len(point_df)}")

# 3. 给每个网格计算中心经纬度，匹配点位所属格子
# 生成网格映射字典 gridId -> (minLon, maxLon, minLat, maxLat)
grid_bounds = {}
for _, row in grid_df.iterrows():
    gid = row["gridId"]
    grid_bounds[gid] = (row["minLon"], row["maxLon"], row["minLat"], row["maxLat"])

# 4. 定义函数：判断一个点落在哪个gridId里
def get_point_grid_id(lon, lat):
    for gid, (minLon, maxLon, minLat, maxLat) in grid_bounds.items():
        if minLon <= lon <= maxLon and minLat <= lat <= maxLat:
            return gid
    return None  # 点不在任何网格内

# 给每个点位匹配gridId
point_df["gridId"] = point_df.apply(lambda r: get_point_grid_id(r["longitude"], r["latitude"]), axis=1)
# 过滤掉不在网格范围内的点
point_valid = point_df[point_df["gridId"].notna()]

# 5. 按网格汇总人口总和
pop_agg = point_valid.groupby("gridId")["population"].sum().reset_index()
pop_agg.rename(columns={"population": "population"}, inplace=True)

# 6. 人口合并到原始网格表，无人口网格填充0
result_df = pd.merge(grid_df, pop_agg, on="gridId", how="left")
result_df["population"] = result_df["population"].fillna(0)

# 7. 输出文件，结构和yangpuGridsHeight.xlsx完全一致，末尾新增population列
result_df.to_excel(OUTPUT_FILE, index=False)
print(f"已生成带人口字段的完整网格文件：{OUTPUT_FILE}")
print("\n前5行预览：")
print(result_df.head())