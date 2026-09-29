#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 6: 稳健性检验与最终结果归档

本脚本只读取 Stage 1-5 已经生成的本地结果，不重新读取原始订单，
不调用深圳开放平台、高德或任何外部接口。

功能：
1. 汇总样本、POI机制、预测、缺口与减碳结果；
2. 检查 Stage 5 基准距离、敏感性距离、时段标签是否一致；
3. 生成论文写作可直接引用的总表、口径说明和最终结果摘要；
4. 复制关键图表到 final_figures 目录，生成图表清单。

PowerShell 示例：
python .\bike_stage6_robustness_archive.py --root-dir .\bike_output_bj_corrected
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PERIOD_HOURS = {
    "night_low": 6,
    "pre_morning": 1,
    "morning_peak": 3,
    "daytime_offpeak": 7,
    "evening_peak": 3,
    "night_tail": 4,
}

PERIOD_ORDER = [
    "night_low",
    "pre_morning",
    "morning_peak",
    "daytime_offpeak",
    "evening_peak",
    "night_tail",
]

PERIOD_CN = {
    "night_low": "夜间低谷(0–5时)",
    "pre_morning": "早高峰前(6时)",
    "morning_peak": "早高峰(7–9时)",
    "daytime_offpeak": "日间平峰(10–16时)",
    "evening_peak": "晚高峰(17–19时)",
    "night_tail": "夜间尾段(20–23时)",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def copy_if_exists(src: Path, dst_dir: Path, label: str, records: List[Dict[str, str]]) -> None:
    if src.exists():
        ensure_dir(dst_dir)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        records.append({"label": label, "filename": src.name, "source_path": str(src), "copied_path": str(dst)})


def compute_distance_audit(root_dir: Path, stage2_dir: Path, selected_dates: Optional[List[str]]) -> Dict[str, Any]:
    """
    从本地聚合表计算订单加权平均距离。优先读取 root/grid_hour_panel.parquet。
    如果字段不存在，则返回 available=False。
    """
    candidates = [
        root_dir / "grid_hour_panel.parquet",
        root_dir / "grid_time_panel.parquet",
        stage2_dir / "model_ready_grid_hour_panel.parquet",
        stage2_dir / "model_ready_grid_hour_panel_with_poi.parquet",
    ]
    panel = None
    source = None
    for p in candidates:
        if p.exists():
            try:
                tmp = pd.read_parquet(p, columns=None)
                if {"orders_out", "sum_distance_km"}.issubset(tmp.columns):
                    panel = tmp
                    source = p
                    break
            except Exception:
                continue

    if panel is None:
        return {"available": False, "reason": "未找到包含 orders_out 与 sum_distance_km 的聚合表。"}

    if "date_bj" in panel.columns and selected_dates:
        dset = set(pd.to_datetime(pd.Series(selected_dates)).dt.date)
        panel["date_bj_tmp"] = pd.to_datetime(panel["date_bj"]).dt.date
        panel = panel[panel["date_bj_tmp"].isin(dset)].copy()

    panel["orders_out"] = pd.to_numeric(panel["orders_out"], errors="coerce").fillna(0)
    panel["sum_distance_km"] = pd.to_numeric(panel["sum_distance_km"], errors="coerce").fillna(0)

    total_orders = float(panel["orders_out"].sum())
    total_distance = float(panel["sum_distance_km"].sum())
    weighted_mean = total_distance / total_orders if total_orders > 0 else np.nan

    nonzero = panel[panel["orders_out"] > 0].copy()
    if len(nonzero):
        nonzero["avg_distance_km_calc"] = nonzero["sum_distance_km"] / nonzero["orders_out"]
        median = float(nonzero["avg_distance_km_calc"].median())
        p25 = float(nonzero["avg_distance_km_calc"].quantile(0.25))
        p75 = float(nonzero["avg_distance_km_calc"].quantile(0.75))
    else:
        median = p25 = p75 = np.nan

    return {
        "available": True,
        "source_file": str(source),
        "total_orders_out": total_orders,
        "total_distance_km": total_distance,
        "weighted_mean_distance_km": weighted_mean,
        "grid_hour_avg_distance_median_km": median,
        "grid_hour_avg_distance_p25_km": p25,
        "grid_hour_avg_distance_p75_km": p75,
    }


def build_sample_overview(root_dir: Path, stage2_dir: Path, stage2_manifest: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    selected_dates = stage2_manifest.get("selected_dates", [])

    quality = safe_read_csv(root_dir / "quality_log.csv")
    if quality is not None and len(quality):
        # 尽量兼容不同字段
        date_col = "date" if "date" in quality.columns else ("query_date" if "query_date" in quality.columns else None)
        raw_col = "raw_rows" if "raw_rows" in quality.columns else None
        clean_col = None
        for c in ["after_distance_speed_filter", "clean_rows", "after_clean", "cleaned_rows"]:
            if c in quality.columns:
                clean_col = c
                break
        if date_col and raw_col and clean_col:
            q = quality.copy()
            q[date_col] = q[date_col].astype(str)
            if selected_dates:
                # quality 可能为 20210820，也可能为 2021-08-20
                selected_compact = {x.replace("-", "") for x in selected_dates}
                q = q[q[date_col].str.replace("-", "").isin(selected_compact)]
            raw = float(pd.to_numeric(q[raw_col], errors="coerce").fillna(0).sum())
            clean = float(pd.to_numeric(q[clean_col], errors="coerce").fillna(0).sum())
            rows.append({
                "item": "订单清洗样本",
                "value": clean,
                "unit": "单",
                "note": f"主分析日期内清洗后订单；原始记录约 {raw:,.0f}，保留率 {clean/raw:.2%}" if raw else "主分析日期内清洗后订单",
            })

    panel = safe_read_parquet(stage2_dir / "model_ready_grid_hour_panel.parquet")
    if panel is not None:
        rows.append({"item": "网格-小时面板行数", "value": len(panel), "unit": "行", "note": "Stage 2 主模型面板"})
        if "grid_id" in panel.columns:
            rows.append({"item": "H3 活跃网格数", "value": panel["grid_id"].nunique(), "unit": "个", "note": "Stage 2 主模型面板"})
        if "date_bj" in panel.columns:
            rows.append({"item": "主分析日期数", "value": panel["date_bj"].nunique(), "unit": "日", "note": "Stage 2 选择日期"})

    return pd.DataFrame(rows)


def build_stage3_summary(stage3b_summary: Dict[str, Any], stage3b_dir: Path) -> pd.DataFrame:
    rows = []
    prep = stage3b_summary.get("preparation", {})
    rows.append({
        "module": "Stage 3B",
        "metric": "模型样本行数",
        "value": prep.get("rows_used"),
        "note": f"active_grid_only={prep.get('use_active_grid_only')}",
    })
    ols = stage3b_summary.get("ols_reduced_hc1", {})
    if ols:
        rows.append({"module": "Stage 3B", "metric": "OLS-log R2", "value": ols.get("rsquared"), "note": "用于机制方向对照"})
    pois = stage3b_summary.get("poisson_reduced_hc1", {})
    if pois:
        rows.append({"module": "Stage 3B", "metric": "Poisson Pearson Chi2/df", "value": pois.get("pearson_chi2_over_df"), "note": "大于1说明过度离散"})
        rows.append({"module": "Stage 3B", "metric": "Poisson AIC", "value": pois.get("aic"), "note": "计数模型对照"})
    nb = stage3b_summary.get("negative_binomial_main", {})
    if nb:
        rows.append({"module": "Stage 3B", "metric": "Negative Binomial AIC", "value": nb.get("aic"), "note": "推荐作为主计数模型"})
    rich = stage3b_summary.get("regularized_logit_is_supply_rich_proxy", {})
    if rich:
        rows.append({"module": "Stage 3B", "metric": "富裕代理 Logit ROC-AUC", "value": rich.get("roc_auc"), "note": "L2正则化Logistic"})
        rows.append({"module": "Stage 3B", "metric": "富裕代理 Balanced Accuracy", "value": rich.get("balanced_accuracy"), "note": "L2正则化Logistic"})
    tight = stage3b_summary.get("regularized_logit_is_supply_tight_proxy", {})
    if tight:
        rows.append({"module": "Stage 3B", "metric": "紧张代理 Logit ROC-AUC", "value": tight.get("roc_auc"), "note": "L2正则化Logistic，解释力较弱"})
        rows.append({"module": "Stage 3B", "metric": "紧张代理 Balanced Accuracy", "value": tight.get("balanced_accuracy"), "note": "L2正则化Logistic"})
    return pd.DataFrame(rows)


def build_stage4_summary(stage4_dir: Path, stage4_manifest: Dict[str, Any]) -> pd.DataFrame:
    perf = safe_read_csv(stage4_dir / "stage4_model_performance.csv")
    rows = []
    if perf is not None:
        # 只从测试集行提取测试指标；训练拟合行的 RMSE 不可标为测试表现。
        metric_col = "rmse" if "rmse" in perf.columns else ("RMSE" if "RMSE" in perf.columns else None)
        model_col = "model" if "model" in perf.columns else ("model_name" if "model_name" in perf.columns else None)
        test_perf = perf.loc[perf["split"].astype(str).str.lower().eq("test")] if "split" in perf.columns else perf.iloc[0:0]
        if metric_col and model_col and len(test_perf):
            best = test_perf.sort_values(metric_col).iloc[0]
            rows.append({"module": "Stage 4", "metric": "测试集最优RMSE", "value": best[metric_col], "note": str(best[model_col])})
            base = test_perf[test_perf[model_col].astype(str).str.contains("baseline", case=False, na=False)]
            if len(base):
                rows.append({"module": "Stage 4", "metric": "历史均值基线RMSE", "value": base.iloc[0][metric_col], "note": str(base.iloc[0][model_col])})
    rows.append({"module": "Stage 4", "metric": "模型样本行数", "value": stage4_manifest.get("rows_model"), "note": "预测缺口面板"})
    rows.append({"module": "Stage 4", "metric": "训练行数", "value": stage4_manifest.get("train_rows_fit"), "note": f"训练日期：{stage4_manifest.get('train_dates')}"})
    rows.append({"module": "Stage 4", "metric": "测试行数", "value": stage4_manifest.get("test_rows"), "note": f"测试日期：{stage4_manifest.get('test_dates')}"})
    return pd.DataFrame(rows)


def build_stage5_tables(stage5_dir: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    scenario = safe_read_csv(stage5_dir / "stage5_scenario_summary.csv")
    if scenario is not None:
        out["stage5_scenario_summary"] = scenario.copy()

    dist = safe_read_csv(stage5_dir / "stage5_distance_sensitivity.csv")
    if dist is not None:
        out["stage5_distance_sensitivity"] = dist.copy()

    period = safe_read_csv(stage5_dir / "stage5_period_summary.csv")
    if period is not None:
        p = period.copy()
        p["hours_in_period"] = p["period_24h"].map(PERIOD_HOURS)
        p["net_carbon_reduction_t"] = pd.to_numeric(p["net_carbon_reduction_kg"], errors="coerce") / 1000
        p["net_carbon_reduction_t_per_hour"] = p["net_carbon_reduction_t"] / p["hours_in_period"]
        total_by_scn = p.groupby("scenario")["net_carbon_reduction_t"].transform("sum")
        p["net_carbon_share"] = np.where(total_by_scn != 0, p["net_carbon_reduction_t"] / total_by_scn, np.nan)
        p["period_order"] = p["period_24h"].map({k: i for i, k in enumerate(PERIOD_ORDER)})
        p = p.sort_values(["scenario", "period_order"]).drop(columns=["period_order"])
        out["stage5_period_summary_with_intensity"] = p

    hour = safe_read_csv(stage5_dir / "stage5_hour_summary.csv")
    if hour is not None:
        h = hour.copy()
        h["net_carbon_reduction_t"] = pd.to_numeric(h["net_carbon_reduction_kg"], errors="coerce") / 1000
        out["stage5_hour_summary_with_tons"] = h

    return out


def write_markdown_report(
    path: Path,
    sample_overview: pd.DataFrame,
    distance_audit: Dict[str, Any],
    stage3_summary: pd.DataFrame,
    stage4_summary: pd.DataFrame,
    stage5_scenario: Optional[pd.DataFrame],
    stage5_period: Optional[pd.DataFrame],
    stage5_manifest: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# 最终研究结果摘要")
    lines.append("")
    lines.append("本摘要由 Stage 6 脚本自动生成。所有结果均基于本地中间表，不重新读取原始订单，不调用高德或深圳开放平台接口。")
    lines.append("")

    lines.append("## 1. 样本与距离口径")
    if not sample_overview.empty:
        for _, r in sample_overview.iterrows():
            lines.append(f"- {r['item']}：{r['value']} {r['unit']}。{r['note']}")
    if distance_audit.get("available"):
        lines.append(
            f"- 订单加权平均骑行距离：{distance_audit['weighted_mean_distance_km']:.3f} km；"
            f"网格小时平均距离中位数：{distance_audit['grid_hour_avg_distance_median_km']:.3f} km；"
            f"P25–P75：{distance_audit['grid_hour_avg_distance_p25_km']:.3f}–{distance_audit['grid_hour_avg_distance_p75_km']:.3f} km。"
        )
    default_distance = stage5_manifest.get("default_trip_distance_km")
    if default_distance is not None:
        lines.append(f"- Stage 5 基准新增骑行距离：{default_distance} km。")
    lines.append("")

    lines.append("## 2. POI 机制分析")
    if not stage3_summary.empty:
        for _, r in stage3_summary.iterrows():
            val = r["value"]
            if isinstance(val, (float, np.floating)):
                val = f"{val:.4f}"
            lines.append(f"- {r['metric']}：{val}。{r['note']}")
    lines.append("- 推荐口径：负二项 GLM 作为主计数模型，OLS-log 与 Poisson 作为对照；正则化 Logistic 用于供给状态代理的辅助解释。")
    lines.append("")

    lines.append("## 3. 潜在需求预测与缺口代理")
    if not stage4_summary.empty:
        for _, r in stage4_summary.iterrows():
            val = r["value"]
            if isinstance(val, (float, np.floating)):
                val = f"{val:.4f}"
            lines.append(f"- {r['metric']}：{val}。{r['note']}")
    lines.append("- `unmet_demand_proxy` 表示预测理论需求高于实际订单的代理差额，不等同真实未成交需求。")
    lines.append("")

    lines.append("## 4. 情景化减碳潜力")
    if stage5_scenario is not None and len(stage5_scenario):
        for _, r in stage5_scenario.iterrows():
            lines.append(
                f"- {r.get('scenario_label', r.get('scenario'))}："
                f"净减碳潜力 {r['net_carbon_reduction_t']:.2f} 吨 CO₂；"
                f"总减碳潜力 {r['gross_carbon_reduction_t']:.2f} 吨 CO₂；"
                f"调度排放扣减 {r['dispatch_carbon_t']:.2f} 吨 CO₂。"
            )
    if stage5_period is not None and len(stage5_period):
        neutral = stage5_period[stage5_period["scenario_label"].astype(str).str.contains("中性", na=False)].copy()
        if len(neutral):
            top_total = neutral.sort_values("net_carbon_reduction_t", ascending=False).iloc[0]
            top_hour = neutral.sort_values("net_carbon_reduction_t_per_hour", ascending=False).iloc[0]
            lines.append(
                f"- 中性情景下，总量最高时段为 {top_total['period_label']}，"
                f"净减碳潜力约 {top_total['net_carbon_reduction_t']:.2f} 吨 CO₂。"
            )
            lines.append(
                f"- 中性情景下，单位小时强度最高时段为 {top_hour['period_label']}，"
                f"约 {top_hour['net_carbon_reduction_t_per_hour']:.2f} 吨 CO₂/小时。"
            )
    lines.append("- 减碳结果为情景化潜力估算，不等同核证减排量或真实净减排量。")
    lines.append("")

    lines.append("## 5. 后续写作建议")
    lines.append("- 先写方法与数据链条，再写结果；所有图表优先引用 Stage 6 归档目录中的版本。")
    lines.append("- 供给富裕、供给紧张、潜在缺口、减碳均使用“代理”或“情景化潜力”口径。")
    lines.append("- 地图终稿如需统一风格，可将散点图进一步转为 H3 面状专题图。")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="bike_output_bj_corrected 根目录")
    parser.add_argument("--stage2-dir-name", default="stage2_research_ready")
    parser.add_argument("--stage3-dir-name", default="stage3_poi_mechanism")
    parser.add_argument("--stage3b-dir-name", default="stage3b_refined_models")
    parser.add_argument("--stage4-dir-name", default="stage4_demand_gap")
    parser.add_argument("--stage5-dir-name", default="stage5_carbon_scenarios_v2")
    parser.add_argument("--out-dir-name", default="stage6_final_archive")
    args = parser.parse_args()

    root = Path(args.root_dir)
    stage2 = root / args.stage2_dir_name
    stage3 = root / args.stage3_dir_name
    stage3b = stage3 / args.stage3b_dir_name
    stage4 = root / args.stage4_dir_name
    stage5 = root / args.stage5_dir_name
    out = root / args.out_dir_name
    ensure_dir(out)
    tables_dir = out / "tables"
    figs_dir = out / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    stage2_manifest = read_json(stage2 / "stage2_manifest.json")
    stage3_manifest = read_json(stage3 / "stage3_manifest.json")
    stage3b_summary = read_json(stage3b / "stage3b_model_summary.json")
    stage4_manifest = read_json(stage4 / "stage4_manifest.json")
    stage5_manifest = read_json(stage5 / "stage5_manifest.json")
    selected_dates = stage2_manifest.get("selected_dates", [])

    # 1. 样本概览
    sample_overview = build_sample_overview(root, stage2, stage2_manifest)
    write_csv(sample_overview, tables_dir / "table_01_sample_overview.csv")

    # 2. 距离核验
    distance_audit = compute_distance_audit(root, stage2, selected_dates)
    pd.DataFrame([distance_audit]).to_csv(tables_dir / "table_02_distance_audit.csv", index=False, encoding="utf-8-sig")

    # 3. Stage 3 模型摘要
    stage3_summary = build_stage3_summary(stage3b_summary, stage3b)
    write_csv(stage3_summary, tables_dir / "table_03_poi_mechanism_model_summary.csv")

    # 4. Stage 4 摘要与性能表
    stage4_summary = build_stage4_summary(stage4, stage4_manifest)
    write_csv(stage4_summary, tables_dir / "table_04_demand_prediction_summary.csv")
    perf = safe_read_csv(stage4 / "stage4_model_performance.csv")
    if perf is not None:
        write_csv(perf, tables_dir / "table_04b_stage4_model_performance.csv")
    imp = safe_read_csv(stage4 / "stage4_hgb_permutation_importance.csv")
    if imp is not None:
        write_csv(imp, tables_dir / "table_04c_stage4_feature_importance.csv")

    # 5. Stage 5 摘要
    stage5_tables = build_stage5_tables(stage5)
    for name, df in stage5_tables.items():
        write_csv(df, tables_dir / f"table_05_{name}.csv")

    # 6. 复制关键模型表
    copy_records: List[Dict[str, str]] = []
    key_tables = [
        (stage3b / "model_negative_binomial_orders_main.csv", "负二项主模型"),
        (stage3b / "model_count_comparison.csv", "计数模型比较"),
        (stage3b / "model_regularized_logit_is_supply_rich_proxy.csv", "富裕代理正则化Logit"),
        (stage3b / "model_regularized_logit_is_supply_tight_proxy.csv", "紧张代理正则化Logit"),
        (stage4 / "stage4_grid_gap_summary.csv", "缺口网格汇总"),
        (stage4 / "stage4_period_gap_summary.csv", "缺口时段汇总"),
        (stage5 / "stage5_grid_summary.csv", "减碳网格汇总"),
        (stage5 / "stage5_high_carbon_grids_top10pct.csv", "高减碳潜力网格"),
    ]
    for src, label in key_tables:
        copy_if_exists(src, tables_dir, label, copy_records)

    # 7. 复制关键图
    figure_sources = [
        (stage2 / "figures_stage2" / "fig_stage2_hour_date_heatmap.png", "日期×小时订单热力图"),
        (stage2 / "figures_stage2" / "fig_stage2_night_net_flow.png", "夜间净流入空间图"),
        (stage2 / "figures_stage2" / "fig_stage2_pre_morning_stock.png", "早高峰前库存代理图"),
        (stage3 / "figures_stage3" / "fig_stage3_poi_correlation_heatmap.png", "POI相关矩阵"),
        (stage3 / "figures_stage3" / "fig_stage3_period_orders_by_poi_metro_count.png", "地铁POI高低组分时段订单"),
        (stage3b / "fig_stage3b_negative_binomial_coefficients.png", "负二项模型系数图"),
        (stage3b / "fig_stage3b_regularized_logit_is_supply_rich_proxy.png", "富裕代理Logit系数图"),
        (stage3b / "fig_stage3b_regularized_logit_is_supply_tight_proxy.png", "紧张代理Logit系数图"),
        (stage4 / "figures_stage4" / "fig_stage4_model_rmse.png", "预测模型RMSE对比"),
        (stage4 / "figures_stage4" / "fig_stage4_hourly_gap.png", "小时缺口代理图"),
        (stage4 / "figures_stage4" / "fig_stage4_unmet_gap_spatial.png", "空间缺口代理图"),
        (stage5 / "figures_stage5" / "fig_stage5_scenario_carbon_comparison.png", "情景减碳对比"),
        (stage5 / "figures_stage5" / "fig_stage5_distance_sensitivity.png", "距离敏感性分析"),
        (stage5 / "figures_stage5" / "fig_stage5_period_net_carbon_neutral.png", "分时段净减碳潜力"),
        (stage5 / "figures_stage5" / "fig_stage5_spatial_net_carbon_neutral.png", "空间净减碳潜力"),
    ]
    figure_records: List[Dict[str, str]] = []
    for src, label in figure_sources:
        copy_if_exists(src, figs_dir, label, figure_records)

    write_csv(pd.DataFrame(copy_records), tables_dir / "copied_key_tables_manifest.csv")
    write_csv(pd.DataFrame(figure_records), out / "final_figure_manifest.csv")

    # 8. 检查项
    checks = []
    # Stage 5 距离
    checks.append({
        "check": "stage5_default_distance",
        "status": "ok" if abs(float(stage5_manifest.get("default_trip_distance_km", np.nan)) - float(distance_audit.get("weighted_mean_distance_km", stage5_manifest.get("default_trip_distance_km", np.nan)))) < 0.05 else "warn",
        "message": f"Stage 5 基准距离={stage5_manifest.get('default_trip_distance_km')}；数据加权平均距离={distance_audit.get('weighted_mean_distance_km')}",
    })
    # period NaN
    ptable = stage5_tables.get("stage5_period_summary_with_intensity")
    if ptable is not None:
        has_nan_period = ptable["period_24h"].isna().any()
        checks.append({
            "check": "stage5_period_label",
            "status": "ok" if not has_nan_period else "fail",
            "message": "stage5_period_summary 无 NaN 时段标签" if not has_nan_period else "stage5_period_summary 存在 NaN 时段标签",
        })
    # model
    pois = stage3b_summary.get("poisson_reduced_hc1", {})
    nb = stage3b_summary.get("negative_binomial_main", {})
    if pois and nb:
        checks.append({
            "check": "count_model_overdispersion",
            "status": "ok" if float(pois.get("pearson_chi2_over_df", 0)) > 1 else "warn",
            "message": f"Poisson Pearson Chi2/df={pois.get('pearson_chi2_over_df')}; NB AIC={nb.get('aic')}，建议以负二项为主模型。",
        })

    write_csv(pd.DataFrame(checks), out / "stage6_quality_checks.csv")

    # 9. 汇总 manifest
    stage6_manifest = {
        "principle": "Stage 6 基于 Stage 1-5 本地结果归档，不重新读取原始订单，不调用任何外部接口。",
        "root_dir": str(root),
        "out_dir": str(out),
        "inputs": {
            "stage2_manifest": str(stage2 / "stage2_manifest.json"),
            "stage3_manifest": str(stage3 / "stage3_manifest.json"),
            "stage3b_summary": str(stage3b / "stage3b_model_summary.json"),
            "stage4_manifest": str(stage4 / "stage4_manifest.json"),
            "stage5_manifest": str(stage5 / "stage5_manifest.json"),
        },
        "selected_dates": selected_dates,
        "distance_audit": distance_audit,
        "created_tables_dir": str(tables_dir),
        "created_figures_dir": str(figs_dir),
        "quality_checks": checks,
        "recommended_main_results": {
            "poi_mechanism_model": "Negative Binomial GLM",
            "demand_prediction_model": "random_forest_log 或 RMSE最低模型",
            "carbon_distance_baseline_km": stage5_manifest.get("default_trip_distance_km"),
            "carbon_interpretation": "情景化减碳潜力，不等同核证减排量或真实净减排量。",
        },
    }
    (out / "stage6_manifest.json").write_text(json.dumps(stage6_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 10. Markdown 摘要
    write_markdown_report(
        out / "final_results_digest.md",
        sample_overview,
        distance_audit,
        stage3_summary,
        stage4_summary,
        stage5_tables.get("stage5_scenario_summary"),
        stage5_tables.get("stage5_period_summary_with_intensity"),
        stage5_manifest,
    )

    print("\n========== Stage 6 完成 ==========")
    print(f"输出目录：{out.resolve()}")
    print("核心文件：stage6_manifest.json, stage6_quality_checks.csv, final_results_digest.md")
    print(f"表格目录：{tables_dir.resolve()}")
    print(f"图表目录：{figs_dir.resolve()}")


if __name__ == "__main__":
    main()
