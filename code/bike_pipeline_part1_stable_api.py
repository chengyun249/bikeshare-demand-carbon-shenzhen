"""
深圳共享单车订单数据处理 Pipeline：第一部分（数据基础与原始订单一次性固化）

核心目标：
1. 从深圳开放数据接口分页读取共享单车订单；
2. 分批清洗、聚合，不长期保存原始订单；
3. 输出后续研究不再需要原始订单的标准中间表；
4. 一次性生成所有依赖原始订单/订单级坐标的基础图像与图表数据。

运行示例：
python bike_pipeline_part1_optimized.py \
  --app-key 你的appKey \
  --start-date 20210820 \
  --end-date 20210830 \
  --out-dir ./bike_output

默认生成图像；如不需要图像，添加 --no-figures；若接口读不到数据，添加 --debug-raw-response 查看首日第一页原始响应。
你的接口样例 START_TIME/END_TIME 为 0:00:00 这类纯时分秒，已确认接口时间为 UTC，默认按 UTC 解释，并转换为北京时间用于研究分箱。

依赖：
pip install pandas numpy requests pyarrow h3 tqdm matplotlib

说明：空间图使用 H3 边界 + matplotlib 直接生成，不强制依赖 geopandas。
"""

from __future__ import annotations

import argparse
import json
import io
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import h3
except ImportError as exc:
    raise ImportError("请先安装 h3：pip install h3") from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib import colors as mcolors
    from matplotlib.ticker import FuncFormatter
except ImportError as exc:
    raise ImportError("请先安装 matplotlib：pip install matplotlib") from exc


# ============================================================
# 一、配置区
# ============================================================

@dataclass
class PipelineConfig:
    # 深圳开放数据接口：共享单车企业每日订单表
    api_url: str = "https://opendata.sz.gov.cn/api/29200_00403627/1/service.xhtml"
    app_key: str = ""  # 必须通过命令行 --app-key 传入，不在代码中固化 key

    # 日期：YYYYMMDD
    start_date: str = "20210820"
    end_date: str = "20210830"

    # 接口分页
    rows: int = 4000
    sleep_seconds: float = 0.20
    max_retry: int = 3
    debug_raw_response: bool = False  # 调试用：保存首日第一页原始响应与解析后的字段

    # 接口时间字段时区。
    # 你的接口样例 START_TIME/END_TIME 是 0:00:00 这类纯时分秒。
    # 已确认接口时间为 UTC；脚本默认先按接口日期补全 UTC 时间，再转换为北京时间。
    source_timezone: str = "utc"  # utc / bj
    timezone_offset_hours: int = 8

    # 默认不截断全天数据。若只研究 8:00-22:00，可命令行设置 --start-hour-bj 8 --end-hour-bj 22
    start_hour_bj: int = 0
    end_hour_bj: int = 23

    # 时间分箱
    time_bin_minutes: int = 60

    # H3 网格精度
    h3_resolution: int = 8

    # 坐标系处理：raw / bd09 / gcj02
    coord_system: str = "bd09"

    # 清洗阈值
    min_duration_seconds: int = 30
    max_duration_seconds: int = 2 * 3600
    min_distance_km: float = 0.05
    max_distance_km: float = 20.0
    max_speed_kmh: float = 25.0

    # 深圳经纬度粗边界
    min_lat: float = 22.30
    max_lat: float = 22.90
    min_lng: float = 113.70
    max_lng: float = 114.70

    # 减碳代理参数：kg CO2 / km
    emission_factor_kg_per_km: float = 0.192

    # 输出目录
    out_dir: str = "./bike_output"

    # 图像输出
    make_figures: bool = True
    figure_dpi: int = 300
    representative_grid_count: int = 3

    # 富裕区 / 紧张区识别参数
    # 由于没有真实库存，初始库存只能作为代理值；默认用网格活跃订单强度估计。
    initial_stock_scale: float = 2.0
    stock_smooth_window: int = 3
    rich_quantile: float = 0.75
    tight_quantile: float = 0.75
    active_grid_min_orders: int = 5
    min_active_grids_per_time: int = 10



# ============================================================
# 二、H3 兼容函数
# ============================================================

def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def h3_to_latlng(cell: str) -> Tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)


def h3_boundary_lonlat(cell: str) -> List[Tuple[float, float]]:
    """返回 GeoJSON 所需的 lon/lat 边界坐标。"""
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(cell)
    else:
        boundary = h3.h3_to_geo_boundary(cell)
    coords = [(lng, lat) for lat, lng in boundary]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


# ============================================================
# 三、坐标转换
# ============================================================

X_PI = math.pi * 3000.0 / 180.0
PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323


def _out_of_china(lng: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return (lng < 72.004) | (lng > 137.8347) | (lat < 0.8293) | (lat > 55.8271)


def _transform_lat(lng: np.ndarray, lat: np.ndarray) -> np.ndarray:
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * np.sqrt(np.abs(lng))
    ret += (20.0 * np.sin(6.0 * lng * PI) + 20.0 * np.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lat * PI) + 40.0 * np.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * np.sin(lat / 12.0 * PI) + 320 * np.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lng(lng: np.ndarray, lat: np.ndarray) -> np.ndarray:
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * np.sqrt(np.abs(lng))
    ret += (20.0 * np.sin(6.0 * lng * PI) + 20.0 * np.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * np.sin(lng * PI) + 40.0 * np.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * np.sin(lng / 12.0 * PI) + 300.0 * np.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lng: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lng = lng.astype(float)
    lat = lat.astype(float)
    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * PI
    magic = np.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrt_magic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrt_magic) * PI)
    dlng = (dlng * 180.0) / (A / sqrt_magic * np.cos(radlat) * PI)
    mglat = lat + dlat
    mglng = lng + dlng
    out = _out_of_china(lng, lat)
    return lng * out + (lng * 2 - mglng) * (~out), lat * out + (lat * 2 - mglat) * (~out)


def bd09_to_gcj02(lng: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = lng - 0.0065
    y = lat - 0.006
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * X_PI)
    theta = np.arctan2(y, x) - 0.000003 * np.cos(x * X_PI)
    return z * np.cos(theta), z * np.sin(theta)


def convert_coords(df: pd.DataFrame, coord_system: str) -> pd.DataFrame:
    coord_system = coord_system.lower()
    slng = df["start_lng"].astype(float).to_numpy()
    slat = df["start_lat"].astype(float).to_numpy()
    elng = df["end_lng"].astype(float).to_numpy()
    elat = df["end_lat"].astype(float).to_numpy()

    if coord_system == "raw":
        s_lng_wgs, s_lat_wgs, e_lng_wgs, e_lat_wgs = slng, slat, elng, elat
    elif coord_system == "gcj02":
        s_lng_wgs, s_lat_wgs = gcj02_to_wgs84(slng, slat)
        e_lng_wgs, e_lat_wgs = gcj02_to_wgs84(elng, elat)
    elif coord_system == "bd09":
        s_lng_gcj, s_lat_gcj = bd09_to_gcj02(slng, slat)
        e_lng_gcj, e_lat_gcj = bd09_to_gcj02(elng, elat)
        s_lng_wgs, s_lat_wgs = gcj02_to_wgs84(s_lng_gcj, s_lat_gcj)
        e_lng_wgs, e_lat_wgs = gcj02_to_wgs84(e_lng_gcj, e_lat_gcj)
    else:
        raise ValueError("coord_system 只能是 raw / bd09 / gcj02")

    df["start_lng_wgs"] = s_lng_wgs
    df["start_lat_wgs"] = s_lat_wgs
    df["end_lng_wgs"] = e_lng_wgs
    df["end_lat_wgs"] = e_lat_wgs
    return df


# ============================================================
# 四、基础工具函数
# ============================================================

def iter_dates(start_date: str, end_date: str) -> Iterable[str]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    cur = start
    while cur <= end:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def canonical_colname(name: object) -> str:
    """
    字段名标准化：去除 BOM、普通空格、全角空格、换行、连字符、点号等。
    你的原始接口字段顺序是 START_LAT, END_LNG, START_LNG, END_LAT，
    本函数只做字段名清洗，不依赖字段顺序。
    """
    s = str(name).replace("\ufeff", "").replace("\u200b", "").replace("\u3000", " ")
    s = s.strip().upper()
    s = re.sub(r"[\s\-\.\/\\()\[\]{}]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一接口字段名。适配你给出的字段格式：
    START_TIME, END_TIME, START_LAT, END_LNG, START_LNG, END_LAT

    注意：这里严格按字段名取值，不按列位置取值，因此 END_LNG 出现在 START_LNG 前面不会影响结果。
    """
    alias_map = {
        # 时间字段；START_TIM/END_TIM 是为兼容 Excel 列宽导出或人工复制时的截断写法。
        "START_TIME": "start_time",
        "START_TIM": "start_time",
        "STARTTIME": "start_time",
        "S_TIME": "start_time",
        "END_TIME": "end_time",
        "END_TIM": "end_time",
        "ENDTIME": "end_time",
        "E_TIME": "end_time",

        # 起点坐标
        "START_LAT": "start_lat",
        "STARTLAT": "start_lat",
        "START_LATITUDE": "start_lat",
        "START_Y": "start_lat",
        "SLAT": "start_lat",
        "START_LNG": "start_lng",
        "START_LON": "start_lng",
        "START_LONG": "start_lng",
        "START_LONGITUDE": "start_lng",
        "START_X": "start_lng",
        "SLNG": "start_lng",
        "SLON": "start_lng",

        # 终点坐标
        "END_LAT": "end_lat",
        "ENDLAT": "end_lat",
        "END_LATITUDE": "end_lat",
        "END_Y": "end_lat",
        "ELAT": "end_lat",
        "END_LNG": "end_lng",
        "END_LON": "end_lng",
        "END_LONG": "end_lng",
        "END_LONGITUDE": "end_lng",
        "END_X": "end_lng",
        "ELNG": "end_lng",
        "ELON": "end_lng",
    }

    renamed = {}
    for col in df.columns:
        key = canonical_colname(col)
        if key in alias_map:
            renamed[col] = alias_map[key]
    df = df.rename(columns=renamed)

    required = ["start_time", "end_time", "start_lat", "start_lng", "end_lat", "end_lng"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        available = [canonical_colname(c) for c in df.columns]
        preview_cols = list(df.columns[:12])
        raise ValueError(
            f"缺少必要字段：{missing}；当前字段标准化后为：{available}；原始前12个字段为：{preview_cols}。"
            "请检查接口字段是否包含 START_TIME/END_TIME/START_LAT/START_LNG/END_LAT/END_LNG。"
            "注意：END_LNG 放在 START_LNG 前面没有问题，代码按字段名读取。"
        )

    # 如果因为大小写/空格等问题产生重复列，只保留第一个同名字段。
    out = df.loc[:, required].copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


_TIME_ONLY_RE = re.compile(r"^\s*\d{1,3}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?\s*$")


def is_time_only_series(s: pd.Series) -> pd.Series:
    """识别 0:00:00 / 00:00:01 这类纯时分秒字段。"""
    return s.astype(str).str.match(_TIME_ONLY_RE, na=False)


def parse_time_series(s: pd.Series, file_date: str) -> pd.Series:
    """
    兼容两类时间：
    1. 纯时分秒：0:00:00、00:08:11 —— 按接口日期 file_date 补全；
    2. 完整日期时间：2021-08-20 00:00:00 —— 直接解析。

    关键修正：不能先用 pd.to_datetime 解析纯时分秒，否则 pandas 可能把 0:00:00 解释为运行当天，导致日期错位。
    """
    s_str = s.astype(str).str.strip()
    base = pd.to_datetime(file_date, format="%Y%m%d")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    time_only = is_time_only_series(s_str)
    if time_only.any():
        td = pd.to_timedelta(s_str[time_only], errors="coerce")
        out.loc[time_only] = base + td

    remaining = ~time_only
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s_str[remaining], errors="coerce")

    return out


def apply_source_timezone(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    将接口时间转换为北京时间与 UTC 两套字段。
    - source_timezone='utc'：接口时间是 UTC，需要 +8 小时得到北京时间；
    - source_timezone='bj'：接口 START_TIME/END_TIME 已经是北京时间，不再加 8 小时。
    """
    mode = config.source_timezone.lower()
    offset = pd.Timedelta(hours=config.timezone_offset_hours)
    if mode in {"bj", "beijing", "asia/shanghai", "local"}:
        df["start_dt_bj"] = df["start_dt_source"]
        df["end_dt_bj"] = df["end_dt_source"]
        df["start_dt_utc"] = df["start_dt_source"] - offset
        df["end_dt_utc"] = df["end_dt_source"] - offset
    elif mode == "utc":
        df["start_dt_utc"] = df["start_dt_source"]
        df["end_dt_utc"] = df["end_dt_source"]
        df["start_dt_bj"] = df["start_dt_source"] + offset
        df["end_dt_bj"] = df["end_dt_source"] + offset
    else:
        raise ValueError("source_timezone 只能是 bj 或 utc")
    return df


def repair_coordinate_swaps(df: pd.DataFrame, config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    基于深圳经纬度范围做字段防呆。
    正常格式应为：lat≈22.x，lng≈113-114.x。
    如果某端出现 lat/lng 明显反置，则自动交换。
    """
    repair_stats = {"start_coord_swap_fixed": 0, "end_coord_swap_fixed": 0}

    def valid_pair(lat_col: str, lng_col: str) -> pd.Series:
        return (
            df[lat_col].between(config.min_lat, config.max_lat)
            & df[lng_col].between(config.min_lng, config.max_lng)
        )

    cur_start = valid_pair("start_lat", "start_lng").mean() if len(df) else 0
    swap_start = (
        df["start_lng"].between(config.min_lat, config.max_lat)
        & df["start_lat"].between(config.min_lng, config.max_lng)
    ).mean() if len(df) else 0
    if swap_start > cur_start + 0.5:
        df[["start_lat", "start_lng"]] = df[["start_lng", "start_lat"]]
        repair_stats["start_coord_swap_fixed"] = 1

    cur_end = valid_pair("end_lat", "end_lng").mean() if len(df) else 0
    swap_end = (
        df["end_lng"].between(config.min_lat, config.max_lat)
        & df["end_lat"].between(config.min_lng, config.max_lng)
    ).mean() if len(df) else 0
    if swap_end > cur_end + 0.5:
        df[["end_lat", "end_lng"]] = df[["end_lng", "end_lat"]]
        repair_stats["end_coord_swap_fixed"] = 1

    return df, repair_stats

def vectorized_haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    R = 6371.0088
    lat1 = np.radians(lat1.astype(float))
    lng1 = np.radians(lng1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lng2 = np.radians(lng2.astype(float))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def zscore_by_group(df: pd.DataFrame, group_cols: List[str], value_col: str, out_col: str) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)[value_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    df[out_col] = ((df[value_col] - mean) / std).fillna(0)
    return df


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """保证空表也有 merge 所需字段，避免单侧为空时报错。"""
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = 0
    return df


def add_time_features_from_bin(time_df: pd.DataFrame) -> pd.DataFrame:
    """由 time_bin_bj 重建日期和时段字段。"""
    time_df = time_df.copy()
    time_df["date_bj"] = time_df["time_bin_bj"].dt.strftime("%Y-%m-%d")
    time_df["hour_bj"] = time_df["time_bin_bj"].dt.hour
    time_df["weekday"] = time_df["time_bin_bj"].dt.dayofweek
    time_df["is_weekend"] = time_df["weekday"].isin([5, 6]).astype(int)
    time_df["period"] = np.select(
        [time_df["hour_bj"].between(7, 9), time_df["hour_bj"].between(17, 19)],
        ["morning_peak", "evening_peak"],
        default="off_peak",
    )
    return time_df


def complete_grid_time_panel(panel: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    补齐 grid × time_bin 面板。
    这是富裕区识别的关键：无订单时段也必须保留，否则库存累计会在静默时段断裂。
    """
    if panel.empty:
        return panel

    panel = panel.copy()
    panel["time_bin_bj"] = pd.to_datetime(panel["time_bin_bj"])
    grids = pd.Series(panel["grid_id"].dropna().unique(), name="grid_id")
    if grids.empty:
        return panel

    freq = f"{config.time_bin_minutes}min"
    start = panel["time_bin_bj"].min().floor(freq)
    end = panel["time_bin_bj"].max().ceil(freq)
    if pd.isna(start) or pd.isna(end):
        return panel

    time_bins = pd.DataFrame({"time_bin_bj": pd.date_range(start=start, end=end, freq=freq)})
    time_bins = add_time_features_from_bin(time_bins)
    skeleton = time_bins.merge(grids.to_frame(), how="cross")

    keys = ["date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period", "grid_id"]
    completed = skeleton.merge(panel, on=keys, how="left")

    fill_zero_cols = [
        "orders_out", "orders_in", "sum_distance_km", "sum_duration_s",
        "carbon_proxy_kg", "avg_distance_km", "avg_duration_s", "net_flow",
    ]
    for c in fill_zero_cols:
        if c not in completed.columns:
            completed[c] = 0
        completed[c] = completed[c].fillna(0)

    return completed.sort_values(["grid_id", "time_bin_bj"]).reset_index(drop=True)


def to_jsonable(value):
    """将 numpy / pandas 标量转成 JSON 可序列化类型。"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ============================================================
# 五、接口读取
# ============================================================

def extract_records_from_response(obj) -> List:
    """
    从深圳开放数据接口返回值中提取订单记录。
    兼容以下常见结构：
    1. {"data": [ {...}, {...} ]}
    2. {"data": {"rows": [ {...} ]}}
    3. {"rows": [ {...} ]} / {"records": [ {...} ]}
    4. data 是制表符文本或字符串列表的情况。
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        if not obj:
            return []
        # 正常情况：list[dict]
        if all(isinstance(x, dict) for x in obj):
            return obj
        # 少数接口会返回 list[str]，后续 records_to_dataframe 会解析
        if all(isinstance(x, str) for x in obj):
            return obj
        # list 里嵌套 dict/list 时递归找第一个可用记录表
        for x in obj:
            found = extract_records_from_response(x)
            if found:
                return found
        return []

    if isinstance(obj, str):
        return [obj] if obj.strip() else []

    if isinstance(obj, dict):
        # 优先查找常见分页字段
        preferred_keys = [
            "data", "rows", "records", "record", "items", "list", "result",
            "results", "content", "values", "value", "retData", "retdata",
        ]
        for key in preferred_keys:
            if key in obj:
                found = extract_records_from_response(obj.get(key))
                if found:
                    return found

        # 兜底：在所有 value 中递归寻找第一个 list[dict]
        for value in obj.values():
            found = extract_records_from_response(value)
            if found:
                return found

    return []


def records_to_dataframe(records: List) -> pd.DataFrame:
    """
    将接口记录转换为 DataFrame。
    正常接口应为 list[dict]；若返回制表符文本，也尽量解析。
    """
    if not records:
        return pd.DataFrame()

    if all(isinstance(x, dict) for x in records):
        return pd.DataFrame(records)

    if all(isinstance(x, str) for x in records):
        text = "\n".join(x.strip() for x in records if str(x).strip())
        if not text:
            return pd.DataFrame()

        # 优先按 tab 解析；若只有一列，再按连续空白或逗号解析。
        for sep in ["\t", r"\s+", ","]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if len(df.columns) >= 6:
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    return pd.DataFrame(records)


def request_api_page(config: PipelineConfig, date: str, page: int, session: requests.Session) -> pd.DataFrame:
    """
    稳定版接口读取函数。

    这里刻意保留原始可运行脚本中的读取方式：
    data = resp.json(); items = data.get("data", [])。

    原因：深圳开放数据该接口的真实返回结构就是顶层 JSON 中的 data 列表。
    之前过度通用的递归解析虽然理论上更灵活，但在这个固定接口上没有必要，
    也会增加排查难度。因此读取阶段保持最小改动，只在后面清洗阶段做字段适配。
    """
    params = {
        "appKey": config.app_key,
        "page": page,
        "rows": config.rows,
        "startDate": date,
        "endDate": date,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    last_error = None
    for attempt in range(1, config.max_retry + 1):
        try:
            resp = session.get(config.api_url, headers=headers, params=params, timeout=60)
            if resp.status_code != 200:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                time.sleep(config.sleep_seconds)
                continue

            data = resp.json()
            items = data.get("data", [])

            if config.debug_raw_response and page == 1:
                debug_dir = Path(config.out_dir) / "_debug_api"
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"raw_response_{date}_page1.txt").write_text(resp.text[:20000], encoding="utf-8")
                meta = {
                    "date": date,
                    "page": page,
                    "http_status": resp.status_code,
                    "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
                    "data_type": type(items).__name__,
                    "data_length": len(items) if hasattr(items, "__len__") else None,
                    "first_record_keys": list(items[0].keys()) if isinstance(items, list) and items and isinstance(items[0], dict) else None,
                    "first_record": items[0] if isinstance(items, list) and items else None,
                }
                (debug_dir / f"parsed_meta_{date}_page1.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            if not items:
                return pd.DataFrame()
            if not isinstance(items, list):
                # 理论上该接口不会走到这里；保留兜底，避免直接崩溃。
                if isinstance(items, dict):
                    for key in ["rows", "records", "items", "list", "result"]:
                        if isinstance(items.get(key), list):
                            items = items[key]
                            break
                else:
                    return pd.DataFrame()
            return pd.DataFrame(items)

        except Exception as exc:
            last_error = exc
            time.sleep(config.sleep_seconds)

    raise RuntimeError(f"{date} 第 {page} 页请求失败：{last_error}")


# ============================================================
# 六、订单级清洗与分批聚合
# ============================================================

def clean_chunk(raw_df: pd.DataFrame, file_date: str, config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "raw_rows": len(raw_df),
        "after_required_fields": 0,
        "start_time_only_rows": 0,
        "end_time_only_rows": 0,
        "overnight_rows": 0,
        "start_coord_swap_fixed": 0,
        "end_coord_swap_fixed": 0,
        "after_time_filter": 0,
        "after_coord_filter": 0,
        "after_duration_filter": 0,
        "after_distance_speed_filter": 0,
    }
    if raw_df.empty:
        return pd.DataFrame(), stats

    df = normalize_columns(raw_df)
    df = df.drop_duplicates()
    stats["after_required_fields"] = len(df)

    for c in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df, coord_repair_stats = repair_coordinate_swaps(df, config)
    stats.update(coord_repair_stats)

    stats["start_time_only_rows"] = int(is_time_only_series(df["start_time"]).sum())
    stats["end_time_only_rows"] = int(is_time_only_series(df["end_time"]).sum())

    df["start_dt_source"] = parse_time_series(df["start_time"], file_date)
    df["end_dt_source"] = parse_time_series(df["end_time"], file_date)
    # time-only 字段若跨过日期零点，end_dt 会小于 start_dt；这里补到次日。
    overnight = df["end_dt_source"] < df["start_dt_source"]
    stats["overnight_rows"] = int(overnight.sum())
    if overnight.any():
        df.loc[overnight, "end_dt_source"] = df.loc[overnight, "end_dt_source"] + pd.Timedelta(days=1)

    df = apply_source_timezone(df, config)

    df = df[(df["start_dt_bj"].dt.hour >= config.start_hour_bj) & (df["start_dt_bj"].dt.hour <= config.end_hour_bj)].copy()
    stats["after_time_filter"] = len(df)

    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng", "start_dt_utc", "end_dt_utc"])
    df = df[
        df["start_lat"].between(config.min_lat, config.max_lat)
        & df["end_lat"].between(config.min_lat, config.max_lat)
        & df["start_lng"].between(config.min_lng, config.max_lng)
        & df["end_lng"].between(config.min_lng, config.max_lng)
    ].copy()
    stats["after_coord_filter"] = len(df)

    df["duration_s"] = (df["end_dt_utc"] - df["start_dt_utc"]).dt.total_seconds()
    df = df[df["duration_s"].between(config.min_duration_seconds, config.max_duration_seconds)].copy()
    stats["after_duration_filter"] = len(df)

    df = convert_coords(df, config.coord_system)
    df["distance_km"] = vectorized_haversine_km(df["start_lat_wgs"], df["start_lng_wgs"], df["end_lat_wgs"], df["end_lng_wgs"])
    df["speed_kmh"] = df["distance_km"] / (df["duration_s"] / 3600.0)
    df = df[df["distance_km"].between(config.min_distance_km, config.max_distance_km) & (df["speed_kmh"] <= config.max_speed_kmh)].copy()
    stats["after_distance_speed_filter"] = len(df)

    if df.empty:
        return df, stats

    df["date_bj"] = df["start_dt_bj"].dt.strftime("%Y-%m-%d")
    df["hour_bj"] = df["start_dt_bj"].dt.hour
    df["time_bin_bj"] = df["start_dt_bj"].dt.floor(f"{config.time_bin_minutes}min")
    df["weekday"] = df["start_dt_bj"].dt.dayofweek
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["period"] = np.select(
        [df["hour_bj"].between(7, 9), df["hour_bj"].between(17, 19)],
        ["morning_peak", "evening_peak"],
        default="off_peak",
    )

    df["start_grid"] = [latlng_to_h3(lat, lng, config.h3_resolution) for lat, lng in zip(df["start_lat_wgs"], df["start_lng_wgs"])]
    df["end_grid"] = [latlng_to_h3(lat, lng, config.h3_resolution) for lat, lng in zip(df["end_lat_wgs"], df["end_lng_wgs"])]
    df["carbon_proxy_kg"] = df["distance_km"] * config.emission_factor_kg_per_km

    keep_cols = [
        "date_bj", "start_dt_bj", "end_dt_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period",
        "start_lat_wgs", "start_lng_wgs", "end_lat_wgs", "end_lng_wgs",
        "duration_s", "distance_km", "speed_kmh", "start_grid", "end_grid", "carbon_proxy_kg",
    ]
    return df[keep_cols].copy(), stats


def aggregate_chunk(clean_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if clean_df.empty:
        return {"out": pd.DataFrame(), "in": pd.DataFrame(), "od": pd.DataFrame(), "hour": pd.DataFrame()}

    base_cols = ["date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period"]
    out_agg = clean_df.groupby(base_cols + ["start_grid"], observed=True).agg(
        orders_out=("start_grid", "size"),
        sum_distance_km=("distance_km", "sum"),
        sum_duration_s=("duration_s", "sum"),
        carbon_proxy_kg=("carbon_proxy_kg", "sum"),
    ).reset_index()

    in_agg = clean_df.groupby(base_cols + ["end_grid"], observed=True).agg(orders_in=("end_grid", "size")).reset_index()

    od_agg = clean_df.groupby(["date_bj", "time_bin_bj", "start_grid", "end_grid"], observed=True).agg(
        od_orders=("start_grid", "size"),
        od_sum_distance_km=("distance_km", "sum"),
        od_carbon_proxy_kg=("carbon_proxy_kg", "sum"),
    ).reset_index()

    hour_agg = clean_df.groupby(["date_bj", "hour_bj"], observed=True).agg(
        orders=("start_grid", "size"),
        sum_distance_km=("distance_km", "sum"),
        carbon_proxy_kg=("carbon_proxy_kg", "sum"),
    ).reset_index()

    return {"out": out_agg, "in": in_agg, "od": od_agg, "hour": hour_agg}


# ============================================================
# 七、合并面板与衍生数据
# ============================================================

def combine_aggregates(parts: List[pd.DataFrame], group_cols: List[str], sum_cols: List[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return df
    return df.groupby(group_cols, observed=True)[sum_cols].sum().reset_index()


def finalize_panel(out_df: pd.DataFrame, in_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    生成网格-时间面板，并构造优化后的隐含库存与富裕区识别指标。

    关键改动：
    1. 先补齐 grid × time_bin 的完整面板，保留无订单时段；
    2. 使用初始库存代理 B_i + 累计净流构造隐含库存；
    3. 富裕区不再只看 cum_net_flow，而是综合库存水平、净流入、库存趋势、流出消耗和短缺代理；
    4. 只在活跃网格中做分位数识别，避免低活动/噪声网格被误判为富裕区。
    """
    base_keys = ["date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period"]
    out_cols = base_keys + ["start_grid", "orders_out", "sum_distance_km", "sum_duration_s", "carbon_proxy_kg"]
    in_cols = base_keys + ["end_grid", "orders_in"]
    out_df = ensure_columns(out_df, out_cols)
    in_df = ensure_columns(in_df, in_cols)

    out = out_df.rename(columns={"start_grid": "grid_id"}).copy()
    inn = in_df.rename(columns={"end_grid": "grid_id"}).copy()
    keys = base_keys + ["grid_id"]
    panel = pd.merge(out, inn, on=keys, how="outer")

    for c in ["orders_out", "orders_in", "sum_distance_km", "sum_duration_s", "carbon_proxy_kg"]:
        if c not in panel.columns:
            panel[c] = 0
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    panel["time_bin_bj"] = pd.to_datetime(panel["time_bin_bj"])
    panel["avg_distance_km"] = np.where(panel["orders_out"] > 0, panel["sum_distance_km"] / panel["orders_out"], 0)
    panel["avg_duration_s"] = np.where(panel["orders_out"] > 0, panel["sum_duration_s"] / panel["orders_out"], 0)
    panel["net_flow"] = panel["orders_in"] - panel["orders_out"]

    panel = complete_grid_time_panel(panel, config)
    panel = panel.sort_values(["date_bj", "grid_id", "time_bin_bj"]).reset_index(drop=True)

    # 网格活跃度：用于剔除低活动网格造成的分位数误判。
    panel["grid_total_orders"] = panel.groupby("grid_id", observed=True)["orders_out"].transform("sum")
    panel["grid_total_activity"] = panel.groupby("grid_id", observed=True)[["orders_out", "orders_in"]].transform("sum").sum(axis=1)
    panel["is_active_grid"] = (panel["grid_total_activity"] >= config.active_grid_min_orders).astype(int)

    # 初始库存代理 B_i：用该网格非零时段的中位活跃强度估计，不把所有网格机械设为 0。
    panel["activity"] = panel["orders_out"] + panel["orders_in"]
    active_median = panel.loc[panel["activity"] > 0].groupby("grid_id", observed=True)["activity"].median()
    active_mean = panel.groupby("grid_id", observed=True)["activity"].mean()
    initial_stock = active_median.reindex(active_mean.index).fillna(active_mean).fillna(0) * config.initial_stock_scale
    panel["initial_stock_proxy"] = panel["grid_id"].map(initial_stock).fillna(0)

    # 隐含库存动态：raw 可为负，表示短缺压力；stock_proxy 截断为非负，用于解释库存水平。
    panel["cum_net_flow"] = panel.groupby(["date_bj", "grid_id"], observed=True)["net_flow"].cumsum()
    panel["stock_raw"] = panel["initial_stock_proxy"] + panel["cum_net_flow"]
    panel["stock_proxy"] = panel["stock_raw"].clip(lower=0)
    panel["shortage_proxy"] = (-panel["stock_raw"]).clip(lower=0)
    panel["stock_smooth"] = panel.groupby("grid_id", observed=True)["stock_proxy"].transform(
        lambda s: s.rolling(config.stock_smooth_window, min_periods=1).mean()
    )
    panel["stock_trend"] = panel.groupby("grid_id", observed=True)["stock_smooth"].diff().fillna(0)

    # 同一时段横向标准化；富裕区是“同一时间下相对更富裕”的空间概念。
    score_cols = [
        ("stock_smooth", "z_stock_smooth"),
        ("stock_trend", "z_stock_trend"),
        ("net_flow", "z_net_flow"),
        ("orders_out", "z_orders_out"),
        ("shortage_proxy", "z_shortage_proxy"),
    ]
    for raw_col, z_col in score_cols:
        panel = zscore_by_group(panel, ["date_bj", "time_bin_bj"], raw_col, z_col)

    panel["supply_rich_score"] = (
        0.60 * panel["z_stock_smooth"]
        + 0.30 * panel["z_net_flow"]
        + 0.20 * panel["z_stock_trend"]
        - 0.25 * panel["z_orders_out"]
        - 0.35 * panel["z_shortage_proxy"]
    )
    panel["supply_tight_score"] = (
        -0.60 * panel["z_stock_smooth"]
        -0.30 * panel["z_net_flow"]
        -0.15 * panel["z_stock_trend"]
        +0.35 * panel["z_orders_out"]
        +0.45 * panel["z_shortage_proxy"]
    )

    active = panel["is_active_grid"] == 1
    group_cols = ["date_bj", "time_bin_bj"]
    active_counts = panel.loc[active].groupby(group_cols, observed=True)["grid_id"].nunique().rename("active_grid_count")
    panel = panel.merge(active_counts.reset_index(), on=group_cols, how="left")
    panel["active_grid_count"] = panel["active_grid_count"].fillna(0).astype(int)

    rich_threshold = panel.loc[active].groupby(group_cols, observed=True)["supply_rich_score"].quantile(config.rich_quantile).rename("rich_threshold")
    tight_threshold = panel.loc[active].groupby(group_cols, observed=True)["supply_tight_score"].quantile(config.tight_quantile).rename("tight_threshold")
    stock_median = panel.loc[active].groupby(group_cols, observed=True)["stock_smooth"].median().rename("stock_median_active")
    threshold_df = pd.concat([rich_threshold, tight_threshold, stock_median], axis=1).reset_index()
    panel = panel.merge(threshold_df, on=group_cols, how="left")

    enough_active = panel["active_grid_count"] >= config.min_active_grids_per_time
    panel["is_supply_rich_proxy"] = (
        active
        & enough_active
        & (panel["supply_rich_score"] >= panel["rich_threshold"])
        & (panel["stock_smooth"] >= panel["stock_median_active"])
    ).astype(int)
    panel["is_supply_tight_proxy"] = (
        active
        & enough_active
        & (panel["supply_tight_score"] >= panel["tight_threshold"])
        & ((panel["shortage_proxy"] > 0) | (panel["stock_smooth"] <= panel["stock_median_active"]))
    ).astype(int)

    centers = {}
    for cell in panel["grid_id"].dropna().unique():
        try:
            centers[cell] = h3_to_latlng(cell)
        except Exception:
            centers[cell] = (np.nan, np.nan)
    panel["grid_center_lat"] = panel["grid_id"].map(lambda x: centers.get(x, (np.nan, np.nan))[0])
    panel["grid_center_lng"] = panel["grid_id"].map(lambda x: centers.get(x, (np.nan, np.nan))[1])

    panel["actual_demand"] = panel["orders_out"]
    return panel


def build_grid_summary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    summary = panel.groupby("grid_id", observed=True).agg(
        total_orders_out=("orders_out", "sum"),
        total_orders_in=("orders_in", "sum"),
        mean_orders_out=("orders_out", "mean"),
        mean_net_flow=("net_flow", "mean"),
        mean_stock_proxy=("stock_proxy", "mean"),
        mean_stock_smooth=("stock_smooth", "mean"),
        mean_shortage_proxy=("shortage_proxy", "mean"),
        mean_supply_rich_score=("supply_rich_score", "mean"),
        mean_supply_tight_score=("supply_tight_score", "mean"),
        rich_time_share=("is_supply_rich_proxy", "mean"),
        tight_time_share=("is_supply_tight_proxy", "mean"),
        total_carbon_proxy_kg=("carbon_proxy_kg", "sum"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
        is_active_grid=("is_active_grid", "max"),
    ).reset_index()
    summary["net_flow_total"] = summary["total_orders_in"] - summary["total_orders_out"]
    return summary


def write_h3_grid_geojson(grid_summary: pd.DataFrame, out_path: Path) -> None:
    features = []
    for row in grid_summary.to_dict("records"):
        cell = row["grid_id"]
        try:
            coords = h3_boundary_lonlat(cell)
        except Exception:
            continue
        props = {k: to_jsonable(v) for k, v in row.items() if k != "grid_id"}
        props["grid_id"] = cell
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")


# ============================================================
# 八、图像生成：论文风格制图，后续不再依赖原始订单
# ============================================================

def ensure_fig_dir(out_dir: Path) -> Path:
    fig_dir = out_dir / "figures_part1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def set_plot_style() -> None:
    """统一论文图风格。中文字体按常见系统字体回退。"""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
            "Arial Unicode MS", "DejaVu Sans"
        ],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
    })


def savefig(path: Path, dpi: int) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close()


def fmt_count(x, _pos=None) -> str:
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def add_source_note(ax, text: str = "注：时间已由 UTC 转换为北京时间；空间单元为 H3 网格。") -> None:
    ax.text(
        0.0, -0.075, text,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color="#666666"
    )


def polygon_rows(grid_summary: pd.DataFrame, value_col: Optional[str] = None) -> Tuple[List[List[Tuple[float, float]]], pd.DataFrame]:
    if grid_summary.empty:
        return [], grid_summary
    cols = ["grid_id", "center_lng", "center_lat"] + ([value_col] if value_col else [])
    cols = [c for c in cols if c in grid_summary.columns]
    df = grid_summary[cols].dropna(subset=["grid_id"]).copy()
    if value_col and value_col in df.columns:
        df = df.dropna(subset=[value_col])
    polygons, kept_idx = [], []
    for idx, row in df.iterrows():
        try:
            polygons.append(h3_boundary_lonlat(row["grid_id"]))
            kept_idx.append(idx)
        except Exception:
            continue
    return polygons, df.loc[kept_idx].reset_index(drop=True)


def style_map_axis(ax, title: str) -> None:
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=10)
    ax.set_axis_off()


def plot_h3_grid_overview(grid_summary: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    polygons, df = polygon_rows(grid_summary)
    if not polygons:
        return
    df.to_csv(fig_dir / "figure_data_h3_grid_overview.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    collection = PolyCollection(
        polygons,
        facecolors="#f7f7f7",
        edgecolors="#4d4d4d",
        linewidths=0.12,
        alpha=0.9,
    )
    ax.add_collection(collection)
    style_map_axis(ax, "研究区域 H3 网格划分")
    ax.text(0.01, 0.01, f"H3 网格数：{len(polygons):,}", transform=ax.transAxes, fontsize=9, color="#555555")
    savefig(fig_dir / "fig00_h3_grid_overview.png", dpi)


def plot_h3_choropleth(
    grid_summary: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    fig_dir: Path,
    dpi: int,
    cmap: str = "viridis",
    diverging_zero: bool = False,
    log_transform: bool = False,
    label: Optional[str] = None,
) -> None:
    if grid_summary.empty or value_col not in grid_summary.columns:
        return
    polygons, df = polygon_rows(grid_summary, value_col)
    if not polygons or df.empty:
        return

    df[["grid_id", value_col, "center_lng", "center_lat"]].to_csv(
        fig_dir / f"figure_data_{Path(filename).stem}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_values = df[value_col].astype(float).to_numpy()
    plot_values = np.log1p(np.clip(raw_values, a_min=0, a_max=None)) if log_transform else raw_values.copy()

    if diverging_zero:
        vmax = np.nanquantile(np.abs(plot_values), 0.98)
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        vmin = np.nanquantile(plot_values, 0.02)
        vmax = np.nanquantile(plot_values, 0.98)
        if not np.isfinite(vmin):
            vmin = np.nanmin(plot_values)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.nanmax(plot_values) if np.nanmax(plot_values) > vmin else vmin + 1
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    collection = PolyCollection(
        polygons,
        array=plot_values,
        cmap=cmap,
        norm=norm,
        edgecolors="#ffffff",
        linewidths=0.05,
        alpha=0.96,
    )
    ax.add_collection(collection)
    style_map_axis(ax, title)

    cbar = fig.colorbar(collection, ax=ax, shrink=0.72, pad=0.015)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5)
    if log_transform:
        cbar.set_label(label or f"log(1 + {value_col})", fontsize=9)
    else:
        cbar.set_label(label or value_col, fontsize=9)

    if log_transform:
        note = "注：色阶采用 log(1+x) 转换，以降低极端高值网格对地图判读的影响。"
    elif diverging_zero:
        note = "注：红蓝色带以 0 为中心，正值表示净流入，负值表示净流出。"
    else:
        note = "注：色阶按 2%—98% 分位数截尾，减少极端值影响。"
    ax.text(0.0, -0.035, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#666666")
    savefig(fig_dir / filename, dpi)


def plot_cleaning_quality(quality_log: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    if quality_log.empty:
        return
    daily = quality_log.groupby("date", observed=True).agg(
        raw_rows=("raw_rows", "sum"),
        clean_rows=("after_distance_speed_filter", "sum"),
    ).reset_index()
    daily["retention_rate"] = np.where(daily["raw_rows"] > 0, daily["clean_rows"] / daily["raw_rows"], np.nan)
    daily.to_csv(fig_dir / "figure_data_cleaning_quality.csv", index=False, encoding="utf-8-sig")

    x = np.arange(len(daily))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
    ax1.bar(x - width / 2, daily["raw_rows"], width, label="原始订单", color="#bdbdbd")
    ax1.bar(x + width / 2, daily["clean_rows"], width, label="清洗后订单", color="#3182bd")
    ax1.set_xticks(x)
    ax1.set_xticklabels(daily["date"], rotation=45, ha="right")
    ax1.set_ylabel("订单数")
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_count))
    ax1.grid(axis="y")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, daily["retention_rate"], marker="o", color="#de2d26", label="保留率")
    ax2.set_ylabel("清洗保留率")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.0%}"))
    ax2.legend(loc="upper right")
    ax1.set_title("数据清洗质量：原始订单、清洗后订单与保留率")
    add_source_note(ax1, "注：保留率 = 距离、速度、时长、坐标等规则清洗后的订单数 / 原始订单数。")
    savefig(fig_dir / "fig01_cleaning_quality.png", dpi)


def plot_hourly_distribution(hour_df: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    if hour_df.empty:
        return
    hourly = hour_df.groupby("hour_bj", observed=True)["orders"].sum().reindex(range(24), fill_value=0).reset_index()
    hourly["orders_smooth"] = hourly["orders"].rolling(3, center=True, min_periods=1).mean()
    hourly.to_csv(fig_dir / "figure_data_hourly_orders.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(hourly["hour_bj"], hourly["orders"], color="#c6dbef", width=0.72, label="小时订单数")
    ax.plot(hourly["hour_bj"], hourly["orders_smooth"], color="#08519c", marker="o", markersize=4, label="3小时平滑")
    ax.axvspan(7, 9, color="#fdae6b", alpha=0.18, label="早高峰")
    ax.axvspan(17, 19, color="#fb6a4a", alpha=0.14, label="晚高峰")
    ax.set_xticks(range(24))
    ax.set_xlabel("北京时间小时")
    ax.set_ylabel("订单数")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_count))
    ax.grid(axis="y")
    ax.set_title("共享单车订单 24 小时分布")
    ax.legend(ncol=2, loc="upper left")
    add_source_note(ax)
    savefig(fig_dir / "fig02_hourly_order_distribution.png", dpi)


def plot_supply_status(panel: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    if panel.empty:
        return
    sub = panel[panel["hour_bj"].between(18, 19)].copy()
    if sub.empty:
        sub = panel.copy()
    status = sub.groupby("grid_id", observed=True).agg(
        rich_share=("is_supply_rich_proxy", "mean"),
        tight_share=("is_supply_tight_proxy", "mean"),
        mean_rich_score=("supply_rich_score", "mean"),
        mean_tight_score=("supply_tight_score", "mean"),
        mean_stock_smooth=("stock_smooth", "mean"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
    ).reset_index()
    status["status"] = np.select(
        [status["rich_share"] >= 0.5, status["tight_share"] >= 0.5],
        ["供给富裕", "供给紧张"],
        default="其他",
    )
    status.to_csv(fig_dir / "figure_data_evening_supply_status.csv", index=False, encoding="utf-8-sig")

    color_map = {"供给富裕": "#2ca25f", "供给紧张": "#de2d26", "其他": "#d9d9d9"}
    polygons, df = polygon_rows(status)
    if polygons:
        if "status" not in df.columns:
            df = df.merge(status[["grid_id", "status"]], on="grid_id", how="left")
        colors = [color_map.get(v, "#d9d9d9") for v in df["status"]]
        fig, ax = plt.subplots(figsize=(7.4, 7.4))
        collection = PolyCollection(polygons, facecolors=colors, edgecolors="white", linewidths=0.05, alpha=0.94)
        ax.add_collection(collection)
        style_map_axis(ax, "晚高峰供给富裕区与紧张区识别")
        for label, color in color_map.items():
            ax.scatter([], [], c=color, label=label, s=36)
        ax.legend(loc="lower left", title="网格状态")
        ax.text(0.0, -0.035, "注：统计北京时间 18—19 点；富裕/紧张基于隐含库存代理得分分位数识别。", transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#666666")
        savefig(fig_dir / "fig05_evening_supply_status.png", dpi)
        return


def plot_inventory_dynamics(panel: pd.DataFrame, fig_dir: Path, dpi: int, grid_count: int) -> None:
    if panel.empty:
        return
    candidates = panel.groupby("grid_id", observed=True)["orders_out"].sum().sort_values(ascending=False)
    if candidates.empty:
        return
    selected = list(candidates.head(grid_count).index)
    sub = panel[panel["grid_id"].isin(selected)].copy()
    data = sub[["time_bin_bj", "grid_id", "stock_smooth", "net_flow", "orders_out"]].sort_values(["grid_id", "time_bin_bj"])
    data.to_csv(fig_dir / "figure_data_representative_inventory.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    palette = ["#08519c", "#238b45", "#d94801", "#6a51a3", "#636363"]
    for i, (grid_id, group) in enumerate(data.groupby("grid_id", observed=True)):
        label = f"网格 {str(grid_id)[-6:]}"
        ax.plot(group["time_bin_bj"], group["stock_smooth"], label=label, color=palette[i % len(palette)])
    ax.axhline(0, color="#969696", linewidth=0.9, linestyle="--")
    ax.set_xlabel("北京时间")
    ax.set_ylabel("平滑库存代理值")
    ax.set_title("代表性高需求网格的隐含库存动态")
    ax.grid(axis="y")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", title="代表性网格")
    add_source_note(ax, "注：选取出发订单总量最高的若干网格；库存为订单流构造的代理值，不是真实车辆库存。")
    savefig(fig_dir / "fig06_representative_inventory_dynamics.png", dpi)


def make_part1_figures(panel: pd.DataFrame, grid_summary: pd.DataFrame, hour_df: pd.DataFrame, quality_log: pd.DataFrame, config: PipelineConfig) -> None:
    set_plot_style()
    out_dir = Path(config.out_dir)
    fig_dir = ensure_fig_dir(out_dir)
    plot_cleaning_quality(quality_log, fig_dir, config.figure_dpi)
    plot_hourly_distribution(hour_df, fig_dir, config.figure_dpi)
    plot_h3_grid_overview(grid_summary, fig_dir, config.figure_dpi)
    plot_h3_choropleth(
        grid_summary,
        "total_orders_out",
        "起点订单空间分布",
        "fig03_origin_order_spatial_distribution.png",
        fig_dir,
        config.figure_dpi,
        cmap="YlOrRd",
        log_transform=True,
        label="log(1 + 起点订单数)",
    )
    plot_h3_choropleth(
        grid_summary,
        "net_flow_total",
        "累计净流入空间分布",
        "fig04_net_flow_spatial_distribution.png",
        fig_dir,
        config.figure_dpi,
        cmap="RdBu_r",
        diverging_zero=True,
        label="累计净流入",
    )
    plot_h3_choropleth(
        grid_summary,
        "rich_time_share",
        "供给富裕时段占比空间分布",
        "fig05a_supply_rich_time_share.png",
        fig_dir,
        config.figure_dpi,
        cmap="Greens",
        label="富裕时段占比",
    )
    plot_h3_choropleth(
        grid_summary,
        "tight_time_share",
        "供给紧张时段占比空间分布",
        "fig05b_supply_tight_time_share.png",
        fig_dir,
        config.figure_dpi,
        cmap="Reds",
        label="紧张时段占比",
    )
    plot_supply_status(panel, fig_dir, config.figure_dpi)
    plot_inventory_dynamics(panel, fig_dir, config.figure_dpi, config.representative_grid_count)


# ============================================================
# 九、输出与主流程
# ============================================================

def save_outputs(panel: pd.DataFrame, od: pd.DataFrame, hour_df: pd.DataFrame, quality_log: pd.DataFrame, config: PipelineConfig) -> None:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_summary = build_grid_summary(panel)

    panel.to_parquet(out_dir / "grid_hour_panel.parquet", index=False)
    panel.to_parquet(out_dir / "grid_time_panel.parquet", index=False)  # 兼容旧文件名
    panel.to_parquet(out_dir / "grid_hour_inventory.parquet", index=False)  # 富裕区/库存识别主表
    od.to_parquet(out_dir / "od_grid_flow.parquet", index=False)
    hour_df.to_parquet(out_dir / "order_hour_summary.parquet", index=False)
    grid_summary.to_parquet(out_dir / "h3_grid_summary.parquet", index=False)
    quality_log.to_csv(out_dir / "quality_log.csv", index=False, encoding="utf-8-sig")

    panel.head(20000).to_csv(out_dir / "grid_hour_panel_sample.csv", index=False, encoding="utf-8-sig")
    od.head(20000).to_csv(out_dir / "od_grid_flow_sample.csv", index=False, encoding="utf-8-sig")
    grid_summary.to_csv(out_dir / "h3_grid_summary.csv", index=False, encoding="utf-8-sig")
    panel.loc[panel["is_supply_rich_proxy"].eq(1) | panel["is_supply_tight_proxy"].eq(1), [
        "date_bj", "time_bin_bj", "hour_bj", "grid_id", "is_supply_rich_proxy",
        "is_supply_tight_proxy", "supply_rich_score", "supply_tight_score",
        "stock_smooth", "shortage_proxy", "grid_center_lat", "grid_center_lng"
    ]].to_csv(out_dir / "rich_tight_grid_time.csv", index=False, encoding="utf-8-sig")

    write_h3_grid_geojson(grid_summary, out_dir / "h3_grid_boundary.geojson")

    with open(out_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        public_config = asdict(config)
        public_config["app_key"] = "***"
        json.dump(public_config, f, ensure_ascii=False, indent=2)

    if config.make_figures:
        make_part1_figures(panel, grid_summary, hour_df, quality_log, config)

    manifest = {
        "principle": "原始订单只在内存中分批使用；本流程不保存 raw_orders，也不要求后续步骤读取原始订单。",
        "core_tables": [
            "grid_hour_panel.parquet",
            "grid_hour_inventory.parquet",
            "od_grid_flow.parquet",
            "order_hour_summary.parquet",
            "h3_grid_summary.parquet",
            "h3_grid_boundary.geojson",
            "quality_log.csv",
            "rich_tight_grid_time.csv",
        ],
        "figures_part1": sorted([p.name for p in (out_dir / "figures_part1").glob("*.png")]) if (out_dir / "figures_part1").exists() else [],
        "figure_data": sorted([p.name for p in (out_dir / "figures_part1").glob("figure_data_*.csv")]) if (out_dir / "figures_part1").exists() else [],
    }
    with open(out_dir / "output_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def run_pipeline(config: PipelineConfig) -> None:
    from datetime import datetime as _dt

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_out_parts: List[pd.DataFrame] = []
    all_in_parts: List[pd.DataFrame] = []
    all_od_parts: List[pd.DataFrame] = []
    all_hour_parts: List[pd.DataFrame] = []
    quality_rows: List[Dict[str, int]] = []

    session = requests.Session()

    dates = list(iter_dates(config.start_date, config.end_date))
    total_dates = len(dates)
    total_clean = 0
    total_raw = 0

    for di, date in enumerate(dates, 1):
        t0 = time.time()
        date_clean = 0
        date_raw = 0
        print(f"\n{'='*55}")
        print(f"[{_dt.now().strftime('%H:%M:%S')}] 日期 {date}（{di}/{total_dates}）")
        print(f"{'='*55}")
        page = 1
        while True:
            raw_df = request_api_page(config, date, page, session)
            if raw_df.empty:
                print(f"[{_dt.now().strftime('%H:%M:%S')}] {date} 第 {page} 页为空，结束该日期。")
                break

            clean_df, stats = clean_chunk(raw_df, date, config)
            stats.update({"date": date, "page": page})
            quality_rows.append(stats)

            date_clean += stats["after_distance_speed_filter"]
            date_raw += stats["raw_rows"]

            aggs = aggregate_chunk(clean_df)
            if not aggs["out"].empty:
                all_out_parts.append(aggs["out"])
            if not aggs["in"].empty:
                all_in_parts.append(aggs["in"])
            if not aggs["od"].empty:
                all_od_parts.append(aggs["od"])
            if not aggs["hour"].empty:
                all_hour_parts.append(aggs["hour"])

            elapsed = time.time() - t0
            rate = page / elapsed if elapsed > 0 else 0
            print(
                f"[{_dt.now().strftime('%H:%M:%S')}] {date} 第{page}页 | "
                f"原始{stats['raw_rows']}→清洗{stats['after_distance_speed_filter']} | "
                f"当日累计 {date_raw}→{date_clean} | "
                f"已用{elapsed:.0f}s | 速度{rate:.1f}页/s"
            )

            del raw_df, clean_df

            if stats["raw_rows"] < config.rows:
                total_raw += date_raw
                total_clean += date_clean
                print(
                    f"[{_dt.now().strftime('%H:%M:%S')}] {date} 完成，"
                    f"共{page}页 耗时{elapsed:.0f}s | "
                    f"全局累计 {total_raw}→{total_clean}"
                )
                break
            page += 1
            time.sleep(config.sleep_seconds)

    print(f"\n[{_dt.now().strftime('%H:%M:%S')}] ========== 合并聚合结果 ==========")
    out_group = ["date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period", "start_grid"]
    in_group = ["date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period", "end_grid"]
    od_group = ["date_bj", "time_bin_bj", "start_grid", "end_grid"]

    out_df = combine_aggregates(all_out_parts, out_group, ["orders_out", "sum_distance_km", "sum_duration_s", "carbon_proxy_kg"])
    in_df = combine_aggregates(all_in_parts, in_group, ["orders_in"])
    od_df = combine_aggregates(all_od_parts, od_group, ["od_orders", "od_sum_distance_km", "od_carbon_proxy_kg"])
    hour_df = combine_aggregates(all_hour_parts, ["date_bj", "hour_bj"], ["orders", "sum_distance_km", "carbon_proxy_kg"])

    panel = finalize_panel(out_df, in_df, config)
    quality_log = pd.DataFrame(quality_rows)
    save_outputs(panel, od_df, hour_df, quality_log, config)

    print("\n========== 完成 ==========")
    print(f"输出目录：{out_dir.resolve()}")
    print("后续不再需要原始订单；请从 output_manifest.json 查看全部中间表和第一部分图像。")


# ============================================================
# 十、命令行入口
# ============================================================

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="深圳共享单车订单数据处理 Pipeline：第一部分")
    parser.add_argument("--app-key", required=True, help="深圳开放数据平台 appKey")
    parser.add_argument("--start-date", default="20210820", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default="20210830", help="结束日期 YYYYMMDD")
    parser.add_argument("--out-dir", default="./bike_output", help="输出目录")
    parser.add_argument("--rows", type=int, default=4000, help="每页条数")
    parser.add_argument("--debug-raw-response", action="store_true", help="保存首日第一页原始响应和解析字段，用于排查接口读不到数据")
    parser.add_argument("--source-timezone", default="utc", choices=["utc", "bj"], help="接口 START_TIME/END_TIME 的时区；已确认接口时间为 UTC，默认 utc")
    parser.add_argument("--time-bin-minutes", type=int, default=60, help="时间分箱分钟数")
    parser.add_argument("--h3-resolution", type=int, default=8, help="H3 网格精度")
    parser.add_argument("--coord-system", default="bd09", choices=["raw", "bd09", "gcj02"], help="原始坐标系假设")
    parser.add_argument("--start-hour-bj", type=int, default=0, help="北京时间开始小时，默认保留全天")
    parser.add_argument("--end-hour-bj", type=int, default=23, help="北京时间结束小时，默认保留全天")
    parser.add_argument("--no-figures", action="store_true", help="只生成中间表，不生成第一部分图像")
    parser.add_argument("--figure-dpi", type=int, default=300, help="图像 dpi")
    parser.add_argument("--initial-stock-scale", type=float, default=2.0, help="初始库存代理放大系数")
    parser.add_argument("--stock-smooth-window", type=int, default=3, help="库存代理滚动平滑窗口")
    parser.add_argument("--rich-quantile", type=float, default=0.75, help="富裕区分位数阈值")
    parser.add_argument("--tight-quantile", type=float, default=0.75, help="紧张区分位数阈值，基于 tight_score")
    parser.add_argument("--active-grid-min-orders", type=int, default=5, help="参与富裕/紧张识别的网格最低活动量")
    args = parser.parse_args()

    return PipelineConfig(
        app_key=args.app_key,
        start_date=args.start_date,
        end_date=args.end_date,
        out_dir=args.out_dir,
        rows=args.rows,
        debug_raw_response=args.debug_raw_response,
        source_timezone=args.source_timezone,
        time_bin_minutes=args.time_bin_minutes,
        h3_resolution=args.h3_resolution,
        coord_system=args.coord_system,
        start_hour_bj=args.start_hour_bj,
        end_hour_bj=args.end_hour_bj,
        make_figures=not args.no_figures,
        figure_dpi=args.figure_dpi,
        initial_stock_scale=args.initial_stock_scale,
        stock_smooth_window=args.stock_smooth_window,
        rich_quantile=args.rich_quantile,
        tight_quantile=args.tight_quantile,
        active_grid_min_orders=args.active_grid_min_orders,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
