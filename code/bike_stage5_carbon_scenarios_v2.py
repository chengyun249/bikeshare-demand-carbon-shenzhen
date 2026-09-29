#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Stage 5 v2: 情景化减碳潜力估算与距离敏感性分析

用途：
1. 读取 Stage 4 输出的 demand_prediction_panel.parquet；
2. 基于潜在未满足需求代理量计算保守/中性/积极情景下的减碳潜力；
3. 修复 period_24h 可能缺失导致的夜间尾段 NaN 问题；
4. 若缺少距离字段，使用 default_trip_distance_km，并输出距离敏感性分析；
5. 不重新读取原始订单，不调用任何外部 API。

PowerShell 示例：
python .\bike_stage5_carbon_scenarios_v2.py --stage4-dir .\bike_output_bj_corrected\stage4_demand_gap

若要显式设定默认距离和敏感性距离：
python .\bike_stage5_carbon_scenarios_v2.py --stage4-dir .\bike_output_bj_corrected\stage4_demand_gap --default-trip-distance-km 1.5 --distance-sensitivity-km 1.0,1.5,2.0,2.5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


DEFAULT_SCENARIOS = [
    {
        "scenario": "conservative",
        "label": "保守情景",
        "realizable_share": 0.25,
        "dispatch_kg_per_added_trip": 0.030,
        "weighted_baseline_kg_per_km": 0.04442,
        "share_private_car": 0.03,
        "share_taxi": 0.03,
        "share_bus": 0.22,
        "share_metro": 0.30,
        "share_ebike": 0.12,
        "share_walk": 0.30,
    },
    {
        "scenario": "neutral",
        "label": "中性情景",
        "realizable_share": 0.50,
        "dispatch_kg_per_added_trip": 0.025,
        "weighted_baseline_kg_per_km": 0.06155,
        "share_private_car": 0.08,
        "share_taxi": 0.07,
        "share_bus": 0.25,
        "share_metro": 0.25,
        "share_ebike": 0.10,
        "share_walk": 0.25,
    },
    {
        "scenario": "optimistic",
        "label": "积极情景",
        "realizable_share": 0.75,
        "dispatch_kg_per_added_trip": 0.020,
        "weighted_baseline_kg_per_km": 0.07820,
        "share_private_car": 0.15,
        "share_taxi": 0.10,
        "share_bus": 0.25,
        "share_metro": 0.20,
        "share_ebike": 0.08,
        "share_walk": 0.22,
    },
]

PERIOD_ORDER = ["night_low", "pre_morning", "morning_peak", "daytime_offpeak", "evening_peak", "night_tail"]
PERIOD_LABEL = {
    "night_low": "夜间低谷(0–5时)",
    "pre_morning": "早高峰前(6时)",
    "morning_peak": "早高峰(7–9时)",
    "daytime_offpeak": "日间平峰(10–16时)",
    "evening_peak": "晚高峰(17–19时)",
    "night_tail": "夜间尾段(20–23时)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 v2: carbon scenario estimation")
    p.add_argument("--stage4-dir", required=True, help="Stage 4 输出目录")
    p.add_argument("--out-dir-name", default="stage5_carbon_scenarios_v2", help="输出子目录名")
    p.add_argument("--input-file", default="demand_prediction_panel.parquet")
    p.add_argument("--scenario-json", default=None, help="可选，自定义情景 JSON")
    p.add_argument("--default-trip-distance-km", type=float, default=1.5)
    p.add_argument("--min-trip-distance-km", type=float, default=0.05)
    p.add_argument("--max-trip-distance-km", type=float, default=10.0)
    p.add_argument("--distance-sensitivity-km", default="1.0,1.5,2.0,2.5", help="距离敏感性分析，逗号分隔")
    p.add_argument("--figure-dpi", type=int, default=300)
    p.add_argument("--carbon-unit", default="kgCO2")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_scenarios(path: Optional[str]) -> pd.DataFrame:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "scenarios" in obj:
            obj = obj["scenarios"]
        scenarios = pd.DataFrame(obj)
    else:
        scenarios = pd.DataFrame(DEFAULT_SCENARIOS)
    required = ["scenario", "label", "realizable_share", "dispatch_kg_per_added_trip", "weighted_baseline_kg_per_km"]
    missing = [c for c in required if c not in scenarios.columns]
    if missing:
        raise ValueError(f"情景参数缺少字段: {missing}")
    return scenarios


def assign_period(hour: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour, errors="coerce")
    out = pd.Series(index=hour.index, dtype="object")
    out[(h >= 0) & (h <= 5)] = "night_low"
    out[h == 6] = "pre_morning"
    out[(h >= 7) & (h <= 9)] = "morning_peak"
    out[(h >= 10) & (h <= 16)] = "daytime_offpeak"
    out[(h >= 17) & (h <= 19)] = "evening_peak"
    out[(h >= 20) & (h <= 23)] = "night_tail"
    return out


def find_distance_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "avg_distance_km", "mean_distance_km", "distance_km", "trip_distance_km",
        "actual_avg_distance_km", "predicted_avg_distance_km"
    ]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0 and s.gt(0).sum() > 0:
                return c
    return None


def prepare_panel(stage4_dir: Path, input_file: str, default_distance: float, min_d: float, max_d: float) -> tuple[pd.DataFrame, str]:
    path = stage4_dir / input_file
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    required = ["orders_out", "predicted_demand", "unmet_demand_proxy", "hour_bj", "grid_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 4 面板缺少字段: {missing}")

    for c in ["orders_out", "predicted_demand", "unmet_demand_proxy", "hour_bj"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["unmet_demand_proxy"] = df["unmet_demand_proxy"].fillna(0).clip(lower=0)
    df["predicted_demand"] = df["predicted_demand"].fillna(0).clip(lower=0)
    df["orders_out"] = df["orders_out"].fillna(0).clip(lower=0)

    df["period_24h"] = assign_period(df["hour_bj"])
    df["period_label"] = df["period_24h"].map(PERIOD_LABEL)

    dist_col = find_distance_col(df)
    if dist_col:
        dist = pd.to_numeric(df[dist_col], errors="coerce")
        df["trip_distance_km_for_carbon"] = dist.clip(lower=min_d, upper=max_d).fillna(default_distance)
        source = dist_col
    else:
        df["trip_distance_km_for_carbon"] = float(default_distance)
        source = "default_trip_distance_km"

    if "grid_center_lng" not in df.columns and "center_lng" in df.columns:
        df["grid_center_lng"] = df["center_lng"]
    if "grid_center_lat" not in df.columns and "center_lat" in df.columns:
        df["grid_center_lat"] = df["center_lat"]

    return df, source


def compute_scenarios(df: pd.DataFrame, scenarios: pd.DataFrame, distance_override: Optional[float] = None) -> pd.DataFrame:
    pieces = []
    if distance_override is not None:
        dist = pd.Series(float(distance_override), index=df.index)
    else:
        dist = df["trip_distance_km_for_carbon"]
    for _, sc in scenarios.iterrows():
        tmp = df[["grid_id", "hour_bj", "period_24h", "period_label", "orders_out", "predicted_demand", "unmet_demand_proxy"]].copy()
        for c in ["date_bj", "time_bin_bj", "grid_center_lng", "grid_center_lat"]:
            if c in df.columns:
                tmp[c] = df[c]
        tmp["scenario"] = sc["scenario"]
        tmp["scenario_label"] = sc["label"]
        tmp["realizable_share"] = float(sc["realizable_share"])
        tmp["baseline_kg_per_km"] = float(sc["weighted_baseline_kg_per_km"])
        tmp["dispatch_kg_per_added_trip"] = float(sc["dispatch_kg_per_added_trip"])
        tmp["mean_distance_km"] = dist.values
        tmp["additional_realized_trips"] = tmp["unmet_demand_proxy"] * tmp["realizable_share"]
        tmp["gross_carbon_reduction_kg"] = tmp["additional_realized_trips"] * tmp["mean_distance_km"] * tmp["baseline_kg_per_km"]
        tmp["dispatch_carbon_kg"] = tmp["additional_realized_trips"] * tmp["dispatch_kg_per_added_trip"]
        tmp["net_carbon_reduction_kg"] = tmp["gross_carbon_reduction_kg"] - tmp["dispatch_carbon_kg"]
        # 若全部缺口实现、且不扣调度排放，表示最大机会损失口径。
        tmp["opportunity_loss_kg_if_all_gap_realized"] = tmp["unmet_demand_proxy"] * tmp["mean_distance_km"] * tmp["baseline_kg_per_km"]
        tmp["net_carbon_reduction_kg"] = tmp["net_carbon_reduction_kg"].clip(lower=0)
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True)


def scenario_summary(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.groupby(["scenario", "scenario_label"], as_index=False).agg(
        actual_orders=("orders_out", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        additional_realized_trips=("additional_realized_trips", "sum"),
        gross_carbon_reduction_kg=("gross_carbon_reduction_kg", "sum"),
        dispatch_carbon_kg=("dispatch_carbon_kg", "sum"),
        net_carbon_reduction_kg=("net_carbon_reduction_kg", "sum"),
        opportunity_loss_kg_if_all_gap_realized=("opportunity_loss_kg_if_all_gap_realized", "sum"),
        mean_distance_km=("mean_distance_km", "mean"),
        rows=("grid_id", "size"),
    )
    for col in ["gross_carbon_reduction", "dispatch_carbon", "net_carbon_reduction", "opportunity_loss"]:
        kg_col = f"{col}_kg" if col != "opportunity_loss" else "opportunity_loss_kg_if_all_gap_realized"
        t_col = f"{col}_t" if col != "opportunity_loss" else "opportunity_loss_t_if_all_gap_realized"
        out[t_col] = out[kg_col] / 1000
    return out


def hour_summary(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby(["scenario", "scenario_label", "hour_bj"], as_index=False).agg(
        actual_orders=("orders_out", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        additional_realized_trips=("additional_realized_trips", "sum"),
        net_carbon_reduction_kg=("net_carbon_reduction_kg", "sum"),
        gross_carbon_reduction_kg=("gross_carbon_reduction_kg", "sum"),
        dispatch_carbon_kg=("dispatch_carbon_kg", "sum"),
    )


def period_summary(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.groupby(["scenario", "scenario_label", "period_24h", "period_label"], as_index=False, dropna=False).agg(
        actual_orders=("orders_out", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        additional_realized_trips=("additional_realized_trips", "sum"),
        net_carbon_reduction_kg=("net_carbon_reduction_kg", "sum"),
        gross_carbon_reduction_kg=("gross_carbon_reduction_kg", "sum"),
        dispatch_carbon_kg=("dispatch_carbon_kg", "sum"),
    )
    out["period_order"] = out["period_24h"].map({v: i for i, v in enumerate(PERIOD_ORDER)})
    out = out.sort_values(["scenario", "period_order"]).drop(columns=["period_order"])
    return out


def grid_summary(panel: pd.DataFrame, neutral_id: str = "neutral") -> pd.DataFrame:
    out = panel.groupby(["scenario", "scenario_label", "grid_id"], as_index=False).agg(
        actual_orders=("orders_out", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        additional_realized_trips=("additional_realized_trips", "sum"),
        net_carbon_reduction_kg=("net_carbon_reduction_kg", "sum"),
        gross_carbon_reduction_kg=("gross_carbon_reduction_kg", "sum"),
        dispatch_carbon_kg=("dispatch_carbon_kg", "sum"),
        mean_distance_km=("mean_distance_km", "mean"),
        grid_center_lng=("grid_center_lng", "first") if "grid_center_lng" in panel.columns else ("orders_out", "size"),
        grid_center_lat=("grid_center_lat", "first") if "grid_center_lat" in panel.columns else ("orders_out", "size"),
    )
    out["net_carbon_rank_pct"] = out.groupby("scenario")["net_carbon_reduction_kg"].rank(pct=True)
    return out


def parse_distance_sensitivity(text: str, default_distance: float) -> List[float]:
    vals = []
    for x in str(text).split(','):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:
            pass
    if default_distance not in vals:
        vals.append(default_distance)
    return sorted(set(vals))


def sensitivity_summary(df: pd.DataFrame, scenarios: pd.DataFrame, distances: List[float]) -> pd.DataFrame:
    rows = []
    for d in distances:
        p = compute_scenarios(df, scenarios, distance_override=d)
        s = scenario_summary(p)
        s.insert(0, "distance_km", d)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def plot_scenario_comparison(summary: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    s = summary.copy()
    labels = s["scenario_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width/2, s["gross_carbon_reduction_t"], width, label="总减碳潜力")
    ax.bar(x + width/2, -s["dispatch_carbon_t"], width, label="调度排放扣减")
    ax.plot(x, s["net_carbon_reduction_t"], marker="o", linewidth=2, label="净减碳潜力")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("吨 CO$_2$")
    ax.set_title("不同情景下的减碳潜力估算")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage5_scenario_carbon_comparison.png", dpi=dpi)
    plt.close(fig)


def plot_hourly(hour_df: pd.DataFrame, fig_dir: Path, dpi: int, scenario: str = "neutral") -> None:
    h = hour_df[hour_df["scenario"] == scenario].sort_values("hour_bj")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(h["hour_bj"], h["net_carbon_reduction_kg"] / 1000, alpha=0.75)
    ax.set_xlabel("北京时间小时")
    ax.set_ylabel("净减碳潜力（吨 CO$_2$）")
    ax.set_title("按小时汇总的净减碳潜力（中性情景）")
    ax.set_xticks(range(24))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage5_hourly_net_carbon_neutral.png", dpi=dpi)
    plt.close(fig)


def plot_period(period_df: pd.DataFrame, fig_dir: Path, dpi: int, scenario: str = "neutral") -> None:
    p = period_df[period_df["scenario"] == scenario].copy()
    p["period_order"] = p["period_24h"].map({v: i for i, v in enumerate(PERIOD_ORDER)})
    p = p.sort_values("period_order")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(p["period_label"], p["net_carbon_reduction_kg"] / 1000, alpha=0.8)
    ax.set_ylabel("净减碳潜力（吨 CO$_2$）")
    ax.set_title("分时段净减碳潜力（中性情景）")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage5_period_net_carbon_neutral.png", dpi=dpi)
    plt.close(fig)


def plot_spatial(grid_df: pd.DataFrame, fig_dir: Path, dpi: int, scenario: str = "neutral") -> None:
    g = grid_df[(grid_df["scenario"] == scenario)].dropna(subset=["grid_center_lng", "grid_center_lat"]).copy()
    if g.empty:
        return
    vals = g["net_carbon_reduction_kg"] / 1000
    vmax = vals.quantile(0.99) if vals.notna().sum() else None
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(g["grid_center_lng"], g["grid_center_lat"], c=vals.clip(upper=vmax), s=12, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("净减碳潜力（吨 CO$_2$，截断至99分位）")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_title("网格净减碳潜力空间分布（中性情景）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage5_spatial_net_carbon_neutral.png", dpi=dpi)
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario, grp in sens.groupby("scenario"):
        label = grp["scenario_label"].iloc[0]
        grp = grp.sort_values("distance_km")
        ax.plot(grp["distance_km"], grp["net_carbon_reduction_t"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("假设平均骑行距离（km）")
    ax.set_ylabel("净减碳潜力（吨 CO$_2$）")
    ax.set_title("平均骑行距离敏感性分析")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage5_distance_sensitivity.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    stage4_dir = Path(cfg.stage4_dir)
    out_dir = stage4_dir.parent / cfg.out_dir_name
    fig_dir = out_dir / "figures_stage5"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    scenarios = read_scenarios(cfg.scenario_json)
    df, distance_source = prepare_panel(
        stage4_dir,
        cfg.input_file,
        cfg.default_trip_distance_km,
        cfg.min_trip_distance_km,
        cfg.max_trip_distance_km,
    )

    panel = compute_scenarios(df, scenarios)
    ssum = scenario_summary(panel)
    hsum = hour_summary(panel)
    psum = period_summary(panel)
    gsum = grid_summary(panel)

    distances = parse_distance_sensitivity(cfg.distance_sensitivity_km, cfg.default_trip_distance_km)
    sens = sensitivity_summary(df, scenarios, distances)

    scenarios.to_csv(out_dir / "stage5_scenario_assumptions.csv", index=False, encoding="utf-8-sig")
    panel.to_parquet(out_dir / "carbon_scenario_panel.parquet", index=False)
    panel.head(5000).to_csv(out_dir / "carbon_scenario_panel_sample.csv", index=False, encoding="utf-8-sig")
    ssum.to_csv(out_dir / "stage5_scenario_summary.csv", index=False, encoding="utf-8-sig")
    hsum.to_csv(out_dir / "stage5_hour_summary.csv", index=False, encoding="utf-8-sig")
    psum.to_csv(out_dir / "stage5_period_summary.csv", index=False, encoding="utf-8-sig")
    gsum.to_csv(out_dir / "stage5_grid_summary.csv", index=False, encoding="utf-8-sig")
    high = gsum[(gsum["scenario"] == "neutral") & (gsum["net_carbon_rank_pct"] >= 0.9)].copy()
    high.to_csv(out_dir / "stage5_high_carbon_grids_top10pct.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(out_dir / "stage5_distance_sensitivity.csv", index=False, encoding="utf-8-sig")

    plot_scenario_comparison(ssum, fig_dir, cfg.figure_dpi)
    plot_hourly(hsum, fig_dir, cfg.figure_dpi)
    plot_period(psum, fig_dir, cfg.figure_dpi)
    plot_spatial(gsum, fig_dir, cfg.figure_dpi)
    plot_sensitivity(sens, fig_dir, cfg.figure_dpi)

    notes = """# Stage 5 v2 模型说明\n\n本阶段基于 Stage 4 的潜在未满足需求代理量进行情景化减碳估算，不重新读取原始订单，也不调用外部接口。\n\n## 口径\n\n- `unmet_demand_proxy` 为理论预测需求高于实际订单的代理差额，不等同真实未成交需求。\n- `net_carbon_reduction_kg` 为情景化净减碳潜力，已扣除简化调度排放，不等同核证减排量。\n- 若 Stage 4 面板未包含有效距离字段，本阶段使用 `default_trip_distance_km`，并通过 `stage5_distance_sensitivity.csv` 给出平均骑行距离敏感性分析。\n"""
    (out_dir / "stage5_model_notes.md").write_text(notes, encoding="utf-8")

    manifest = {
        "principle": "Stage 5 v2 基于 Stage 4 预测缺口结果，不重新读取原始订单，不调用高德或深圳开放平台接口。",
        "config": vars(cfg),
        "input": str(stage4_dir / cfg.input_file),
        "out_dir": str(out_dir),
        "rows_input": int(len(df)),
        "distance_source": distance_source,
        "default_trip_distance_km": cfg.default_trip_distance_km,
        "distance_sensitivity_km": distances,
        "period_fix_applied": True,
        "scenario_ids": scenarios["scenario"].tolist(),
        "core_outputs": [
            "carbon_scenario_panel.parquet",
            "stage5_scenario_summary.csv",
            "stage5_grid_summary.csv",
            "stage5_hour_summary.csv",
            "stage5_period_summary.csv",
            "stage5_high_carbon_grids_top10pct.csv",
            "stage5_distance_sensitivity.csv",
        ],
        "figures": [
            "fig_stage5_scenario_carbon_comparison.png",
            "fig_stage5_hourly_net_carbon_neutral.png",
            "fig_stage5_period_net_carbon_neutral.png",
            "fig_stage5_spatial_net_carbon_neutral.png",
            "fig_stage5_distance_sensitivity.png",
        ],
        "note": "减碳结果为情景化潜力估算，依赖替代交通比例、排放因子、平均距离和调度排放假设，不等同于核证减排量或真实净减排量。",
    }
    (out_dir / "stage5_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== Stage 5 v2 完成 ==========")
    print(f"输入：{stage4_dir / cfg.input_file}")
    print(f"输出目录：{out_dir.resolve()}")
    print(f"距离来源：{distance_source}; 默认距离={cfg.default_trip_distance_km} km")
    print("核心文件：stage5_scenario_summary.csv, stage5_period_summary.csv, stage5_distance_sensitivity.csv")


if __name__ == "__main__":
    main()
