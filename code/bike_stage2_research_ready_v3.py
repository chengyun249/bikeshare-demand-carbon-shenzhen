"""
共享单车研究第二阶段：分析样本冻结、全天指标构造、POI聚合与模型数据准备

用途：
1. 基于第一阶段已经生成的聚合结果，不重新读取原始订单；
2. 自动检查日期完整性，冻结论文主分析样本；
3. 构造全天分时段指标、夜间沉淀指标、早高峰前库存代理、日内库存循环指标；
4. 可选：把高德/其他来源 POI 点位聚合到 H3 网格；
5. 输出后续 Poisson/负二项面板模型、MGWR、预测和减碳情景所需的标准表。

推荐运行：
python bike_stage2_research_ready.py ^
  --out-dir ./bike_output_bj_corrected ^
  --analysis-end-date 2021-08-29

如已有 POI CSV：
python bike_stage2_research_ready.py ^
  --out-dir ./bike_output_bj_corrected ^
  --analysis-end-date 2021-08-29 ^
  --poi-csv ./poi.csv ^
  --poi-lng-col lng ^
  --poi-lat-col lat ^
  --poi-type-col category ^
  --poi-coord-system gcj02

依赖：
pip install pandas numpy pyarrow matplotlib h3

说明：
- 本脚本默认接口时间已经按北京时间修正，不再做 UTC + 8。
- 供给富裕/紧张仍然是订单流代理指标，不是实时库存真值。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import h3
except ImportError as exc:
    raise ImportError("请先安装 h3：pip install h3") from exc

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError as exc:
    raise ImportError("请先安装 matplotlib：pip install matplotlib") from exc


# ============================================================
# 0. 论文制图基础设置
# ============================================================

def setup_plot_style() -> None:
    mpl.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


setup_plot_style()


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class Stage2Config:
    out_dir: str
    stage2_dir_name: str = "stage2_research_ready"

    # 主样本日期；若为空，由数据自动判断。建议明确设置 end_date，剔除明显不完整日。
    analysis_start_date: Optional[str] = None  # YYYY-MM-DD
    analysis_end_date: Optional[str] = None    # YYYY-MM-DD

    # 日期完整性判断
    expected_hours_per_day: int = 24
    min_hourly_nonzero_hours: int = 22
    min_daily_orders_ratio_to_median: float = 0.70

    # 富裕/紧张识别参数。与第一阶段保持一致，可用于筛选后重算。
    initial_stock_scale: float = 2.0
    stock_smooth_window: int = 3
    rich_quantile: float = 0.75
    tight_quantile: float = 0.75
    active_grid_min_orders: int = 5
    min_active_grids_per_time: int = 10

    # POI 输入，可选
    poi_csv: Optional[str] = None
    poi_lng_col: str = "lng"
    poi_lat_col: str = "lat"
    poi_type_col: str = "type"
    poi_location_col: Optional[str] = None  # 高德常见字段：location="lng,lat"
    poi_coord_system: str = "gcj02"  # gcj02 / bd09 / wgs84 / raw
    poi_encoding: str = "utf-8-sig"

    # H3 精度：默认从 panel/grid_summary 推断；不能推断时用 8。
    h3_resolution: int = 8


# ============================================================
# 2. 通用读写
# ============================================================

def read_table(path_no_ext: Path) -> pd.DataFrame:
    """优先读 parquet；若不存在则读 csv。"""
    parquet_path = path_no_ext.with_suffix(".parquet")
    csv_path = path_no_ext.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    raise FileNotFoundError(f"找不到表：{parquet_path} 或 {csv_path}")


def write_table(df: pd.DataFrame, path_no_ext: Path, csv_also: bool = True, sample_rows: int = 20000) -> None:
    """写 parquet，并可同步写 CSV 或样例 CSV。"""
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_no_ext.with_suffix(".parquet"), index=False)
    if csv_also:
        if len(df) <= sample_rows:
            df.to_csv(path_no_ext.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        else:
            df.head(sample_rows).to_csv(path_no_ext.with_name(path_no_ext.name + "_sample").with_suffix(".csv"), index=False, encoding="utf-8-sig")


def safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 3. H3 与坐标转换
# ============================================================

X_PI = math.pi * 3000.0 / 180.0
PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323


def latlng_to_h3(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(float(lat), float(lng), resolution)
    return h3.geo_to_h3(float(lat), float(lng), resolution)


def h3_to_latlng(cell: str) -> Tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)


def infer_h3_resolution(grid_id: str, default: int = 8) -> int:
    try:
        if hasattr(h3, "get_resolution"):
            return h3.get_resolution(grid_id)
        return h3.h3_get_resolution(grid_id)
    except Exception:
        return default


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


def coords_to_wgs84(lng: pd.Series, lat: pd.Series, coord_system: str) -> Tuple[np.ndarray, np.ndarray]:
    lng_arr = pd.to_numeric(lng, errors="coerce").astype(float).to_numpy()
    lat_arr = pd.to_numeric(lat, errors="coerce").astype(float).to_numpy()
    mode = coord_system.lower()
    if mode in {"wgs84", "wgs", "raw"}:
        return lng_arr, lat_arr
    if mode == "gcj02":
        return gcj02_to_wgs84(lng_arr, lat_arr)
    if mode == "bd09":
        lng_gcj, lat_gcj = bd09_to_gcj02(lng_arr, lat_arr)
        return gcj02_to_wgs84(lng_gcj, lat_gcj)
    raise ValueError("poi_coord_system 只能是 gcj02 / bd09 / wgs84 / raw")


# ============================================================
# 4. 时间与库存代理重算
# ============================================================

def add_period_24h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h = df["hour_bj"].astype(int)
    df["period_24h"] = np.select(
        [
            h.between(0, 5),
            h.eq(6),
            h.between(7, 9),
            h.between(10, 16),
            h.between(17, 19),
            h.between(20, 23),
        ],
        ["night_low", "pre_morning", "morning_peak", "daytime_offpeak", "evening_peak", "late_evening"],
        default="unknown",
    )
    df["period_cn"] = df["period_24h"].map({
        "night_low": "夜间低谷(0-5)",
        "pre_morning": "早高峰前(6)",
        "morning_peak": "早高峰(7-9)",
        "daytime_offpeak": "日间平峰(10-16)",
        "evening_peak": "晚高峰(17-19)",
        "late_evening": "夜间尾段(20-23)",
    })
    df["is_morning_peak"] = df["period_24h"].eq("morning_peak").astype(int)
    df["is_evening_peak"] = df["period_24h"].eq("evening_peak").astype(int)
    df["is_night"] = df["period_24h"].isin(["night_low", "late_evening"]).astype(int)
    df["is_pre_morning"] = df["period_24h"].eq("pre_morning").astype(int)
    return df


def zscore_by_group(df: pd.DataFrame, group_cols: List[str], value_col: str, out_col: str) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)[value_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    df[out_col] = ((df[value_col] - mean) / std).fillna(0)
    return df


def recompute_inventory(panel: pd.DataFrame, cfg: Stage2Config) -> pd.DataFrame:
    """筛选样本后重新计算库存代理与富裕/紧张标签。

    说明：输入的 panel 往往已经带有第一阶段计算过的库存/阈值字段。
    第二阶段筛选日期后必须重新计算这些字段，因此先删除旧衍生列，
    避免 merge 时生成 active_grid_count_x / active_grid_count_y 导致 KeyError。
    """
    panel = panel.copy()

    old_derived_cols = [
        "grid_total_orders", "grid_total_activity", "is_active_grid", "activity",
        "initial_stock_proxy", "cum_net_flow", "cum_net_flow_daily",
        "stock_raw", "stock_proxy", "shortage_proxy",
        "cum_net_flow_continuous", "stock_raw_continuous", "stock_proxy_continuous",
        "stock_smooth", "stock_trend",
        "z_stock_smooth", "z_stock_trend", "z_net_flow", "z_orders_out", "z_shortage_proxy",
        "supply_rich_score", "supply_tight_score",
        "active_grid_count", "rich_threshold", "tight_threshold", "stock_median_active",
        "is_supply_rich_proxy", "is_supply_tight_proxy",
        "actual_demand",
    ]
    panel = panel.drop(columns=[c for c in old_derived_cols if c in panel.columns], errors="ignore")

    panel["time_bin_bj"] = pd.to_datetime(panel["time_bin_bj"])
    panel["date_bj"] = panel["time_bin_bj"].dt.strftime("%Y-%m-%d")
    panel["hour_bj"] = panel["time_bin_bj"].dt.hour
    panel["weekday"] = panel["time_bin_bj"].dt.dayofweek
    panel["is_weekend"] = panel["weekday"].isin([5, 6]).astype(int)
    panel["period"] = np.select(
        [panel["hour_bj"].between(7, 9), panel["hour_bj"].between(17, 19)],
        ["morning_peak", "evening_peak"],
        default="off_peak",
    )
    panel = add_period_24h(panel)

    for c in ["orders_out", "orders_in", "sum_distance_km", "sum_duration_s", "carbon_proxy_kg"]:
        if c not in panel.columns:
            panel[c] = 0
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)
    panel["avg_distance_km"] = np.where(panel["orders_out"] > 0, panel["sum_distance_km"] / panel["orders_out"], 0)
    panel["avg_duration_s"] = np.where(panel["orders_out"] > 0, panel["sum_duration_s"] / panel["orders_out"], 0)
    panel["net_flow"] = panel["orders_in"] - panel["orders_out"]

    panel = panel.sort_values(["grid_id", "time_bin_bj"]).reset_index(drop=True)
    panel["grid_total_orders"] = panel.groupby("grid_id", observed=True)["orders_out"].transform("sum")
    panel["grid_total_activity"] = panel.groupby("grid_id", observed=True)[["orders_out", "orders_in"]].transform("sum").sum(axis=1)
    panel["is_active_grid"] = (panel["grid_total_activity"] >= cfg.active_grid_min_orders).astype(int)
    panel["activity"] = panel["orders_out"] + panel["orders_in"]

    active_median = panel.loc[panel["activity"] > 0].groupby("grid_id", observed=True)["activity"].median()
    active_mean = panel.groupby("grid_id", observed=True)["activity"].mean()
    initial_stock = active_median.reindex(active_mean.index).fillna(active_mean).fillna(0) * cfg.initial_stock_scale
    panel["initial_stock_proxy"] = panel["grid_id"].map(initial_stock).fillna(0)

    # 日内累计库存代理：适合解释日内循环。
    panel["cum_net_flow_daily"] = panel.groupby(["date_bj", "grid_id"], observed=True)["net_flow"].cumsum()
    panel["stock_raw"] = panel["initial_stock_proxy"] + panel["cum_net_flow_daily"]
    panel["stock_proxy"] = panel["stock_raw"].clip(lower=0)
    panel["shortage_proxy"] = (-panel["stock_raw"]).clip(lower=0)

    # 跨日连续累计库存代理：作为稳健性和夜间沉淀的辅助指标。
    panel["cum_net_flow_continuous"] = panel.groupby("grid_id", observed=True)["net_flow"].cumsum()
    panel["stock_raw_continuous"] = panel["initial_stock_proxy"] + panel["cum_net_flow_continuous"]
    panel["stock_proxy_continuous"] = panel["stock_raw_continuous"].clip(lower=0)

    panel["stock_smooth"] = panel.groupby("grid_id", observed=True)["stock_proxy"].transform(
        lambda s: s.rolling(cfg.stock_smooth_window, min_periods=1).mean()
    )
    panel["stock_trend"] = panel.groupby("grid_id", observed=True)["stock_smooth"].diff().fillna(0)

    for raw_col, z_col in [
        ("stock_smooth", "z_stock_smooth"),
        ("stock_trend", "z_stock_trend"),
        ("net_flow", "z_net_flow"),
        ("orders_out", "z_orders_out"),
        ("shortage_proxy", "z_shortage_proxy"),
    ]:
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
        - 0.30 * panel["z_net_flow"]
        - 0.15 * panel["z_stock_trend"]
        + 0.35 * panel["z_orders_out"]
        + 0.45 * panel["z_shortage_proxy"]
    )

    group_cols = ["date_bj", "time_bin_bj"]
    active_mask = panel["is_active_grid"].eq(1)
    active_panel = panel.loc[active_mask].copy()

    active_counts = active_panel.groupby(group_cols, observed=True)["grid_id"].nunique().rename("active_grid_count")
    panel = panel.merge(active_counts.reset_index(), on=group_cols, how="left")
    if "active_grid_count" not in panel.columns:
        panel["active_grid_count"] = 0
    panel["active_grid_count"] = panel["active_grid_count"].fillna(0).astype(int)

    rich_threshold = active_panel.groupby(group_cols, observed=True)["supply_rich_score"].quantile(cfg.rich_quantile).rename("rich_threshold")
    tight_threshold = active_panel.groupby(group_cols, observed=True)["supply_tight_score"].quantile(cfg.tight_quantile).rename("tight_threshold")
    stock_median = active_panel.groupby(group_cols, observed=True)["stock_smooth"].median().rename("stock_median_active")
    thresholds = pd.concat([rich_threshold, tight_threshold, stock_median], axis=1).reset_index()
    panel = panel.merge(thresholds, on=group_cols, how="left")

    active_mask = panel["is_active_grid"].eq(1)
    enough_active = panel["active_grid_count"] >= cfg.min_active_grids_per_time
    panel["is_supply_rich_proxy"] = (
        active_mask & enough_active &
        (panel["supply_rich_score"] >= panel["rich_threshold"]) &
        (panel["stock_smooth"] >= panel["stock_median_active"])
    ).fillna(False).astype(int)
    panel["is_supply_tight_proxy"] = (
        active_mask & enough_active &
        (panel["supply_tight_score"] >= panel["tight_threshold"]) &
        ((panel["shortage_proxy"] > 0) | (panel["stock_smooth"] <= panel["stock_median_active"]))
    ).fillna(False).astype(int)
    panel["actual_demand"] = panel["orders_out"]
    return panel


# ============================================================
# 5. 日期完整性检查与样本冻结
# ============================================================

def build_hourly_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["time_bin_bj"] = pd.to_datetime(p["time_bin_bj"])
    p["date_bj"] = p["time_bin_bj"].dt.strftime("%Y-%m-%d")
    p["hour_bj"] = p["time_bin_bj"].dt.hour
    return p.groupby(["date_bj", "hour_bj"], observed=True).agg(
        orders=("orders_out", "sum"),
        orders_in=("orders_in", "sum"),
        net_flow=("net_flow", "sum"),
        sum_distance_km=("sum_distance_km", "sum"),
        carbon_proxy_kg=("carbon_proxy_kg", "sum"),
    ).reset_index()


def read_quality_daily(out_dir: Path) -> pd.DataFrame:
    """读取 quality_log.csv，按接口日期汇总原始/清洗订单量。

    用于区分“后处理统计偏低”和“接口当天实际抓取量偏低”。
    如果某天 interface_raw_rows 本身明显低于其他日期，问题发生在接口抓取或该日数据完整性，
    不是 H3 面板统计造成的。
    """
    path = out_dir / "quality_log.csv"
    if not path.exists():
        return pd.DataFrame()
    q = pd.read_csv(path)
    if "date" not in q.columns:
        return pd.DataFrame()
    q = q.copy()
    q["date_bj"] = pd.to_datetime(q["date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    agg = q.groupby("date_bj", observed=True).agg(
        interface_pages=("page", "max"),
        interface_raw_rows=("raw_rows", "sum"),
        interface_clean_rows=("after_distance_speed_filter", "sum"),
    ).reset_index()
    agg["interface_retention_rate"] = np.where(
        agg["interface_raw_rows"] > 0,
        agg["interface_clean_rows"] / agg["interface_raw_rows"],
        np.nan,
    )
    return agg


def check_daily_completeness(hour_df: pd.DataFrame, cfg: Stage2Config, out_dir: Optional[Path] = None) -> pd.DataFrame:
    h = hour_df.copy()
    h["date_bj"] = h["date_bj"].astype(str)
    daily = h.groupby("date_bj", observed=True).agg(
        hours_present=("hour_bj", "nunique"),
        nonzero_hours=("orders", lambda s: int((s > 0).sum())),
        daily_orders=("orders", "sum"),
    ).reset_index()

    if out_dir is not None:
        q_daily = read_quality_daily(out_dir)
        if not q_daily.empty:
            daily = daily.merge(q_daily, on="date_bj", how="left")

    median_orders = daily.loc[daily["daily_orders"] > 0, "daily_orders"].median()
    daily["daily_orders_ratio_to_median"] = daily["daily_orders"] / median_orders if median_orders else np.nan

    if "interface_raw_rows" in daily.columns:
        median_raw = daily.loc[daily["interface_raw_rows"] > 0, "interface_raw_rows"].median()
        daily["interface_raw_ratio_to_median"] = daily["interface_raw_rows"] / median_raw if median_raw else np.nan
    else:
        daily["interface_raw_ratio_to_median"] = np.nan

    daily["is_complete_by_hours"] = daily["hours_present"].ge(cfg.expected_hours_per_day) & daily["nonzero_hours"].ge(cfg.min_hourly_nonzero_hours)
    daily["is_complete_by_volume"] = daily["daily_orders_ratio_to_median"].ge(cfg.min_daily_orders_ratio_to_median)
    daily["is_complete_by_interface_raw"] = daily["interface_raw_ratio_to_median"].isna() | daily["interface_raw_ratio_to_median"].ge(cfg.min_daily_orders_ratio_to_median)
    daily["is_recommended"] = daily["is_complete_by_hours"] & daily["is_complete_by_volume"] & daily["is_complete_by_interface_raw"]

    def _reason(row) -> str:
        reasons = []
        if not row["is_complete_by_hours"]:
            reasons.append("小时覆盖不足")
        if not row["is_complete_by_volume"]:
            reasons.append(f"面板订单量偏低({row['daily_orders_ratio_to_median']:.2f}×中位数)")
        if not row["is_complete_by_interface_raw"]:
            reasons.append(f"接口原始抓取量偏低({row['interface_raw_ratio_to_median']:.2f}×中位数)")
        return "；".join(reasons) if reasons else "推荐保留"

    daily["qc_reason"] = daily.apply(_reason, axis=1)
    return daily

def select_analysis_dates(day_qc: pd.DataFrame, cfg: Stage2Config) -> List[str]:
    dates = sorted(day_qc.loc[day_qc["is_recommended"], "date_bj"].astype(str).unique())
    if cfg.analysis_start_date:
        dates = [d for d in dates if d >= cfg.analysis_start_date]
    if cfg.analysis_end_date:
        dates = [d for d in dates if d <= cfg.analysis_end_date]
    return dates


# ============================================================
# 6. 阶段性汇总表
# ============================================================

def build_grid_summary(panel: pd.DataFrame) -> pd.DataFrame:
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


def build_grid_period_summary(panel: pd.DataFrame) -> pd.DataFrame:
    gp = panel.groupby(["grid_id", "period_24h", "period_cn"], observed=True).agg(
        orders_out=("orders_out", "sum"),
        orders_in=("orders_in", "sum"),
        net_flow=("net_flow", "sum"),
        avg_orders_out=("orders_out", "mean"),
        avg_stock_smooth=("stock_smooth", "mean"),
        avg_shortage_proxy=("shortage_proxy", "mean"),
        rich_share=("is_supply_rich_proxy", "mean"),
        tight_share=("is_supply_tight_proxy", "mean"),
        avg_distance_km=("avg_distance_km", "mean"),
        carbon_proxy_kg=("carbon_proxy_kg", "sum"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
    ).reset_index()
    return gp


def build_night_flow_summary(panel: pd.DataFrame) -> pd.DataFrame:
    night = panel[panel["period_24h"].isin(["night_low", "late_evening"])].copy()
    return night.groupby("grid_id", observed=True).agg(
        night_orders_out=("orders_out", "sum"),
        night_orders_in=("orders_in", "sum"),
        night_net_flow=("net_flow", "sum"),
        night_stock_smooth_mean=("stock_smooth", "mean"),
        night_rich_share=("is_supply_rich_proxy", "mean"),
        night_tight_share=("is_supply_tight_proxy", "mean"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
    ).reset_index()


def build_pre_morning_stock_summary(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel[panel["period_24h"].eq("pre_morning")].copy()
    return pre.groupby("grid_id", observed=True).agg(
        pre_morning_stock_proxy=("stock_smooth", "mean"),
        pre_morning_orders_out=("orders_out", "sum"),
        pre_morning_orders_in=("orders_in", "sum"),
        pre_morning_net_flow=("net_flow", "sum"),
        pre_morning_rich_share=("is_supply_rich_proxy", "mean"),
        pre_morning_tight_share=("is_supply_tight_proxy", "mean"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
    ).reset_index()


def build_daily_stock_cycle_summary(panel: pd.DataFrame) -> pd.DataFrame:
    daily = panel.groupby(["date_bj", "grid_id"], observed=True).agg(
        daily_orders_out=("orders_out", "sum"),
        daily_orders_in=("orders_in", "sum"),
        daily_net_flow=("net_flow", "sum"),
        stock_min=("stock_smooth", "min"),
        stock_max=("stock_smooth", "max"),
        stock_mean=("stock_smooth", "mean"),
        rich_share=("is_supply_rich_proxy", "mean"),
        tight_share=("is_supply_tight_proxy", "mean"),
        center_lat=("grid_center_lat", "first"),
        center_lng=("grid_center_lng", "first"),
    ).reset_index()
    daily["daily_stock_amplitude"] = daily["stock_max"] - daily["stock_min"]
    return daily


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["grid_id", "time_bin_bj"]).copy()
    for lag in [1, 2, 24]:
        p[f"orders_out_lag_{lag}"] = p.groupby("grid_id", observed=True)["orders_out"].shift(lag).fillna(0)
        p[f"orders_in_lag_{lag}"] = p.groupby("grid_id", observed=True)["orders_in"].shift(lag).fillna(0)
        p[f"net_flow_lag_{lag}"] = p.groupby("grid_id", observed=True)["net_flow"].shift(lag).fillna(0)
    p["orders_out_rolling_3h"] = p.groupby("grid_id", observed=True)["orders_out"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    p["orders_out_rolling_24h"] = p.groupby("grid_id", observed=True)["orders_out"].transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean()).fillna(0)
    p["log_orders_out_plus1"] = np.log1p(p["orders_out"])
    p["log_orders_in_plus1"] = np.log1p(p["orders_in"])
    return p


# ============================================================
# 7. POI 聚合
# ============================================================

def sanitize_type_name(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


def build_poi_grid_features(poi_csv: Path, cfg: Stage2Config, grid_ids: pd.Series, centers: pd.DataFrame, stage2_dir: Path) -> pd.DataFrame:
    poi = pd.read_csv(poi_csv, encoding=cfg.poi_encoding)

    # 高德 POI 常见坐标字段为 location="lng,lat"；若经纬度列不存在，则自动拆分。
    loc_col = cfg.poi_location_col or ("location" if "location" in poi.columns else None)
    if (cfg.poi_lng_col not in poi.columns or cfg.poi_lat_col not in poi.columns) and loc_col and loc_col in poi.columns:
        loc = poi[loc_col].astype(str).str.split(",", n=1, expand=True)
        if loc.shape[1] >= 2:
            poi[cfg.poi_lng_col] = loc[0]
            poi[cfg.poi_lat_col] = loc[1]

    missing = [c for c in [cfg.poi_lng_col, cfg.poi_lat_col, cfg.poi_type_col] if c not in poi.columns]
    if missing:
        raise ValueError(f"POI 文件缺少字段：{missing}；当前字段：{list(poi.columns)}")

    poi = poi.dropna(subset=[cfg.poi_lng_col, cfg.poi_lat_col]).copy()
    lng_wgs, lat_wgs = coords_to_wgs84(poi[cfg.poi_lng_col], poi[cfg.poi_lat_col], cfg.poi_coord_system)
    poi["lng_wgs"] = lng_wgs
    poi["lat_wgs"] = lat_wgs
    poi = poi.dropna(subset=["lng_wgs", "lat_wgs"])

    poi["grid_id"] = [latlng_to_h3(lat, lng, cfg.h3_resolution) for lat, lng in zip(poi["lat_wgs"], poi["lng_wgs"])]
    poi["poi_type_raw"] = poi[cfg.poi_type_col].astype(str)
    poi["poi_type_clean"] = poi["poi_type_raw"].map(sanitize_type_name)

    # 只保留研究区出现过的网格，避免行政区外 POI 混入。
    valid_grids = set(grid_ids.dropna().astype(str).unique())
    poi_in = poi[poi["grid_id"].astype(str).isin(valid_grids)].copy()
    poi.to_csv(stage2_dir / "poi_with_h3_all.csv", index=False, encoding="utf-8-sig")
    poi_in.to_csv(stage2_dir / "poi_with_h3_in_study_area.csv", index=False, encoding="utf-8-sig")

    count = poi_in.groupby(["grid_id", "poi_type_clean"], observed=True).size().rename("count").reset_index()
    wide = count.pivot_table(index="grid_id", columns="poi_type_clean", values="count", fill_value=0, aggfunc="sum")
    wide.columns = [f"poi_{c}_count" for c in wide.columns]
    wide = wide.reset_index()
    wide["poi_total_count"] = wide[[c for c in wide.columns if c.startswith("poi_") and c.endswith("_count")]].sum(axis=1)

    # 保证所有订单网格都有一行 POI 特征。
    base = pd.DataFrame({"grid_id": sorted(valid_grids)})
    features = base.merge(wide, on="grid_id", how="left").fillna(0)
    features = features.merge(centers[["grid_id", "center_lat", "center_lng"]].drop_duplicates("grid_id"), on="grid_id", how="left")
    return features


def write_poi_template(grid_summary: pd.DataFrame, out_path: Path) -> None:
    cols = ["grid_id", "center_lat", "center_lng"]
    tmp = grid_summary[cols].copy()
    for c in [
        "poi_residential_count", "poi_office_count", "poi_commercial_count", "poi_metro_count",
        "poi_bus_count", "poi_education_count", "poi_recreation_count", "poi_public_service_count",
    ]:
        tmp[c] = 0
    tmp.to_csv(out_path, index=False, encoding="utf-8-sig")


# ============================================================
# 8. 图表
# ============================================================

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_hour_date_heatmap(hour_df: pd.DataFrame, fig_dir: Path) -> None:
    pivot = hour_df.pivot_table(index="date_bj", columns="hour_bj", values="orders", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("北京时间小时")
    ax.set_ylabel("日期")
    ax.set_title("订单量：日期 × 小时热力图")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("订单数")
    savefig(fig_dir / "fig_stage2_hour_date_heatmap.png")


def plot_period_total(grid_period: pd.DataFrame, fig_dir: Path) -> None:
    order = ["night_low", "pre_morning", "morning_peak", "daytime_offpeak", "evening_peak", "late_evening"]
    period = grid_period.groupby(["period_24h", "period_cn"], observed=True)["orders_out"].sum().reset_index()
    period["period_24h"] = pd.Categorical(period["period_24h"], categories=order, ordered=True)
    period = period.sort_values("period_24h")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(period["period_cn"], period["orders_out"], edgecolor="black", linewidth=0.3)
    ax.set_ylabel("起点订单数")
    ax.set_title("不同时段订单总量")
    ax.tick_params(axis="x", rotation=25)
    savefig(fig_dir / "fig_stage2_period_orders.png")


def plot_grid_scatter(df: pd.DataFrame, value_col: str, title: str, filename: str, fig_dir: Path, cmap: str = "viridis") -> None:
    d = df.dropna(subset=["center_lng", "center_lat", value_col]).copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    vals = d[value_col].astype(float)
    sc = ax.scatter(d["center_lng"], d["center_lat"], c=vals, s=12, cmap=cmap, alpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75)
    cbar.set_label(value_col)
    ax.grid(False)
    savefig(fig_dir / filename)


def make_figures(hour_df: pd.DataFrame, grid_period: pd.DataFrame, night_summary: pd.DataFrame, pre_summary: pd.DataFrame, grid_summary: pd.DataFrame, fig_dir: Path) -> None:
    plot_hour_date_heatmap(hour_df, fig_dir)
    plot_period_total(grid_period, fig_dir)
    plot_grid_scatter(night_summary, "night_net_flow", "夜间净流入空间分布", "fig_stage2_night_net_flow.png", fig_dir, cmap="coolwarm")
    plot_grid_scatter(pre_summary, "pre_morning_stock_proxy", "早高峰前库存代理空间分布", "fig_stage2_pre_morning_stock.png", fig_dir, cmap="YlGnBu")
    plot_grid_scatter(grid_summary, "rich_time_share", "供给富裕代理时段占比", "fig_stage2_rich_time_share.png", fig_dir, cmap="Greens")
    plot_grid_scatter(grid_summary, "tight_time_share", "供给紧张代理时段占比", "fig_stage2_tight_time_share.png", fig_dir, cmap="Reds")


# ============================================================
# 9. 主流程
# ============================================================

def run(cfg: Stage2Config) -> None:
    out_dir = Path(cfg.out_dir)
    stage2_dir = out_dir / cfg.stage2_dir_name
    fig_dir = stage2_dir / "figures_stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = read_table(out_dir / "grid_hour_panel")
    panel["time_bin_bj"] = pd.to_datetime(panel["time_bin_bj"])

    if "grid_center_lat" not in panel.columns or "grid_center_lng" not in panel.columns:
        centers = {}
        for cell in panel["grid_id"].dropna().unique():
            try:
                centers[cell] = h3_to_latlng(cell)
            except Exception:
                centers[cell] = (np.nan, np.nan)
        panel["grid_center_lat"] = panel["grid_id"].map(lambda x: centers.get(x, (np.nan, np.nan))[0])
        panel["grid_center_lng"] = panel["grid_id"].map(lambda x: centers.get(x, (np.nan, np.nan))[1])

    # 推断 H3 精度。
    if len(panel) and panel["grid_id"].notna().any():
        cfg.h3_resolution = infer_h3_resolution(str(panel["grid_id"].dropna().iloc[0]), cfg.h3_resolution)

    # 小时表：优先读已生成表；失败则从 panel 重建。
    try:
        hour_df = read_table(out_dir / "order_hour_summary")
    except Exception:
        hour_df = build_hourly_from_panel(panel)
    hour_df["date_bj"] = hour_df["date_bj"].astype(str)

    day_qc = check_daily_completeness(hour_df, cfg, out_dir)
    selected_dates = select_analysis_dates(day_qc, cfg)
    if not selected_dates:
        raise RuntimeError("没有找到满足完整性条件的分析日期。请降低完整性阈值或手动指定日期。")

    # 冻结主样本。
    panel["date_bj"] = panel["time_bin_bj"].dt.strftime("%Y-%m-%d")
    panel_selected = panel[panel["date_bj"].isin(selected_dates)].copy()
    panel_selected = recompute_inventory(panel_selected, cfg)
    panel_selected = add_lag_features(panel_selected)

    hour_selected = build_hourly_from_panel(panel_selected)
    grid_summary = build_grid_summary(panel_selected)
    grid_period = build_grid_period_summary(panel_selected)
    night_summary = build_night_flow_summary(panel_selected)
    pre_summary = build_pre_morning_stock_summary(panel_selected)
    daily_cycle = build_daily_stock_cycle_summary(panel_selected)

    # 输出核心表。
    day_qc.to_csv(stage2_dir / "analysis_day_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date_bj": selected_dates}).to_csv(stage2_dir / "analysis_dates_selected.csv", index=False, encoding="utf-8-sig")
    write_table(panel_selected, stage2_dir / "model_ready_grid_hour_panel", csv_also=True)
    write_table(hour_selected, stage2_dir / "order_hour_summary_selected", csv_also=True)
    write_table(grid_summary, stage2_dir / "grid_summary_selected", csv_also=True)
    write_table(grid_period, stage2_dir / "grid_period_summary", csv_also=True)
    write_table(night_summary, stage2_dir / "night_flow_summary", csv_also=True)
    write_table(pre_summary, stage2_dir / "pre_morning_stock_summary", csv_also=True)
    write_table(daily_cycle, stage2_dir / "daily_stock_cycle_summary", csv_also=True)

    # POI 模板或 POI 聚合。
    write_poi_template(grid_summary, stage2_dir / "poi_grid_features_template.csv")
    poi_output = None
    if cfg.poi_csv:
        poi_features = build_poi_grid_features(Path(cfg.poi_csv), cfg, grid_summary["grid_id"], grid_summary.rename(columns={"center_lat": "center_lat", "center_lng": "center_lng"}), stage2_dir)
        poi_features.to_csv(stage2_dir / "poi_grid_features.csv", index=False, encoding="utf-8-sig")
        model_with_poi = panel_selected.merge(poi_features.drop(columns=["center_lat", "center_lng"], errors="ignore"), on="grid_id", how="left")
        poi_cols = [c for c in model_with_poi.columns if c.startswith("poi_")]
        model_with_poi[poi_cols] = model_with_poi[poi_cols].fillna(0)
        write_table(model_with_poi, stage2_dir / "model_ready_grid_hour_panel_with_poi", csv_also=True)
        poi_output = "model_ready_grid_hour_panel_with_poi.parquet"

    # 图表。
    make_figures(hour_selected, grid_period, night_summary, pre_summary, grid_summary, fig_dir)

    # 模型说明。
    model_notes = f"""# 第二阶段分析数据说明

## 主分析日期
{', '.join(selected_dates)}

## 核心表
- `model_ready_grid_hour_panel.parquet`：格网—小时主模型表。
- `grid_period_summary.parquet`：格网—全天分时段汇总表。
- `night_flow_summary.parquet`：夜间车辆沉淀/净流入代理表。
- `pre_morning_stock_summary.parquet`：早高峰前库存代理表。
- `daily_stock_cycle_summary.parquet`：日内库存循环幅度表。
- `poi_grid_features_template.csv`：POI 聚合模板。

## 建议主模型方向
1. 描述性分析：24小时订单、分时段空间图、夜间净流入、早高峰前库存代理。
2. 主解释模型：Poisson/负二项面板，因变量为 `orders_out` 或 `actual_demand`。
3. 交互项：POI 类型 × 时段，例如商业 × 晚高峰、居住 × 早高峰、地铁 × 早高峰。
4. MGWR：只做早高峰、平峰、晚高峰的截面辅助分析，不作为唯一主模型。
5. 预测：用 `model_ready_grid_hour_panel_with_poi.parquet` 接入天气、POI 后做 RF/XGBoost 短窗预测。

## 表述边界
- `is_supply_rich_proxy` 是供给富裕代理，不是实时库存真值。
- `is_supply_tight_proxy` 是供给紧张风险代理，不是真实缺车。
- 后续缺口应写作潜在未满足需求代理量。
"""
    (stage2_dir / "stage2_model_notes.md").write_text(model_notes, encoding="utf-8")

    manifest = {
        "principle": "第二阶段基于第一阶段聚合结果，不重新读取原始订单。",
        "stage2_config": asdict(cfg),
        "selected_dates": selected_dates,
        "input_out_dir": str(out_dir),
        "stage2_dir": str(stage2_dir),
        "core_outputs": [
            "analysis_day_qc.csv",
            "analysis_dates_selected.csv",
            "model_ready_grid_hour_panel.parquet",
            "order_hour_summary_selected.parquet",
            "grid_summary_selected.parquet",
            "grid_period_summary.parquet",
            "night_flow_summary.parquet",
            "pre_morning_stock_summary.parquet",
            "daily_stock_cycle_summary.parquet",
            "poi_grid_features_template.csv",
            "stage2_model_notes.md",
        ],
        "poi_output": poi_output,
        "figures": sorted([p.name for p in fig_dir.glob("*.png")]),
    }
    with open(stage2_dir / "stage2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n========== 第二阶段处理完成 ==========")
    print(f"输出目录：{stage2_dir.resolve()}")
    print(f"主分析日期：{', '.join(selected_dates)}")
    print("核心文件：model_ready_grid_hour_panel.parquet, grid_period_summary.parquet, night_flow_summary.parquet")
    if cfg.poi_csv:
        print("已生成 POI 合并模型表：model_ready_grid_hour_panel_with_poi.parquet")
    else:
        print("未提供 POI 文件，已生成 poi_grid_features_template.csv。")


# ============================================================
# 10. 命令行入口
# ============================================================

def parse_args() -> Stage2Config:
    parser = argparse.ArgumentParser(description="共享单车研究第二阶段：指标构造、POI聚合与模型数据准备")
    parser.add_argument("--out-dir", required=True, help="第一阶段修正后的输出目录，例如 ./bike_output_bj_corrected")
    parser.add_argument("--stage2-dir-name", default="stage2_research_ready", help="第二阶段输出子目录名")
    parser.add_argument("--analysis-start-date", default=None, help="主分析开始日期 YYYY-MM-DD，可为空")
    parser.add_argument("--analysis-end-date", default=None, help="主分析结束日期 YYYY-MM-DD，建议先设为 2021-08-29")
    parser.add_argument("--min-daily-orders-ratio-to-median", type=float, default=0.70, help="日订单量低于中位数该比例则判为疑似不完整日；默认0.70会排除明显偏低日期")
    parser.add_argument("--initial-stock-scale", type=float, default=2.0)
    parser.add_argument("--stock-smooth-window", type=int, default=3)
    parser.add_argument("--rich-quantile", type=float, default=0.75)
    parser.add_argument("--tight-quantile", type=float, default=0.75)
    parser.add_argument("--active-grid-min-orders", type=int, default=5)
    parser.add_argument("--min-active-grids-per-time", type=int, default=10)
    parser.add_argument("--poi-csv", default=None, help="可选：POI 点位 CSV")
    parser.add_argument("--poi-lng-col", default="lng")
    parser.add_argument("--poi-lat-col", default="lat")
    parser.add_argument("--poi-type-col", default="type")
    parser.add_argument("--poi-location-col", default=None, help="可选：高德 location 字段，格式为 lng,lat")
    parser.add_argument("--poi-coord-system", default="gcj02", choices=["gcj02", "bd09", "wgs84", "raw"])
    parser.add_argument("--poi-encoding", default="utf-8-sig")
    parser.add_argument("--h3-resolution", type=int, default=8)
    args = parser.parse_args()

    return Stage2Config(
        out_dir=args.out_dir,
        stage2_dir_name=args.stage2_dir_name,
        analysis_start_date=args.analysis_start_date,
        analysis_end_date=args.analysis_end_date,
        min_daily_orders_ratio_to_median=args.min_daily_orders_ratio_to_median,
        initial_stock_scale=args.initial_stock_scale,
        stock_smooth_window=args.stock_smooth_window,
        rich_quantile=args.rich_quantile,
        tight_quantile=args.tight_quantile,
        active_grid_min_orders=args.active_grid_min_orders,
        min_active_grids_per_time=args.min_active_grids_per_time,
        poi_csv=args.poi_csv,
        poi_lng_col=args.poi_lng_col,
        poi_lat_col=args.poi_lat_col,
        poi_type_col=args.poi_type_col,
        poi_location_col=args.poi_location_col,
        poi_coord_system=args.poi_coord_system,
        poi_encoding=args.poi_encoding,
        h3_resolution=args.h3_resolution,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
