r"""
Stage 4: 潜在需求预测与未满足需求代理测度

用途：
1. 读取 Stage 3 的订单-POI-时空特征面板；
2. 以相对供给平稳样本训练理论需求预测模型；
3. 对全样本网格-小时预测潜在需求；
4. 构造潜在未满足需求代理量：max(predicted_demand - actual_demand, 0)；
5. 输出 Stage 5 减碳情景估算所需表格与图像。

PowerShell 示例：
python .\bike_stage4_demand_gap.py --stage3-dir .\bike_output_bj_corrected\stage3_poi_mechanism

若想使用全样本训练，不限制模型训练行数：
python .\bike_stage4_demand_gap.py --stage3-dir .\bike_output_bj_corrected\stage3_poi_mechanism --max-train-rows 0
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


@dataclass
class Stage4Config:
    stage3_dir: str
    out_dir_name: str = "stage4_demand_gap"
    input_file: str = "model_data_with_poi_transformed.parquet"
    target_col: str = "orders_out"
    test_days: int = 2
    train_stable_only: bool = True
    use_active_grid_only: bool = True
    max_train_rows: int = 300000
    max_importance_rows: int = 30000
    random_state: int = 42
    gap_cap_quantile: float = 0.995
    min_prediction: float = 0.0
    make_figures: bool = True
    figure_dpi: int = 300


POI_FEATURES = [
    "log1p_poi_office_count",
    "log1p_poi_commercial_count",
    "log1p_poi_residential_count",
    "log1p_poi_metro_count",
    "log1p_poi_bus_count",
    "log1p_poi_education_count",
    "log1p_poi_recreation_count",
    "log1p_poi_public_service_count",
    "log1p_poi_medical_count",
    "log1p_poi_transport_other_count",
]

BASE_NUM_FEATURES = [
    "grid_center_lng",
    "grid_center_lat",
    "is_weekend",
    "log1p_orders_out_lag_1",
    "log1p_orders_out_lag_2",
    "log1p_orders_out_lag_24",
    "log1p_orders_out_rolling_3h",
    "log1p_orders_out_rolling_24h",
    "log1p_orders_in",
    "log1p_orders_in_lag_1",
    "log1p_net_in_abs",
]

CAT_FEATURES = ["hour_bj", "weekday", "period_24h"]

PERIOD_ORDER = [
    "夜间低谷(0-5)",
    "早高峰前(6)",
    "早高峰(7-9)",
    "日间平峰(10-16)",
    "晚高峰(17-19)",
    "夜间尾段(20-23)",
]


def ensure_stage4_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 时间字段
    if "time_bin_bj" in df.columns:
        df["time_bin_bj"] = pd.to_datetime(df["time_bin_bj"], errors="coerce")
    if "date_bj" not in df.columns and "time_bin_bj" in df.columns:
        df["date_bj"] = df["time_bin_bj"].dt.strftime("%Y-%m-%d")
    if "hour_bj" not in df.columns and "time_bin_bj" in df.columns:
        df["hour_bj"] = df["time_bin_bj"].dt.hour
    if "weekday" not in df.columns and "time_bin_bj" in df.columns:
        df["weekday"] = df["time_bin_bj"].dt.dayofweek
    if "is_weekend" not in df.columns and "weekday" in df.columns:
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    if "period_24h" not in df.columns:
        def period(h):
            h = int(h)
            if 0 <= h <= 5:
                return "夜间低谷(0-5)"
            if h == 6:
                return "早高峰前(6)"
            if 7 <= h <= 9:
                return "早高峰(7-9)"
            if 10 <= h <= 16:
                return "日间平峰(10-16)"
            if 17 <= h <= 19:
                return "晚高峰(17-19)"
            return "夜间尾段(20-23)"
        df["period_24h"] = df["hour_bj"].map(period)

    # 数值特征
    for c in ["orders_out", "orders_in", "net_flow"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 如果还没有 log lag，重新构造
    sort_cols = [c for c in ["grid_id", "time_bin_bj"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    if "orders_out_lag_1" not in df.columns and {"grid_id", "orders_out"}.issubset(df.columns):
        df["orders_out_lag_1"] = df.groupby("grid_id", observed=True)["orders_out"].shift(1)
    if "orders_out_lag_2" not in df.columns and {"grid_id", "orders_out"}.issubset(df.columns):
        df["orders_out_lag_2"] = df.groupby("grid_id", observed=True)["orders_out"].shift(2)
    if "orders_out_lag_24" not in df.columns and {"grid_id", "orders_out"}.issubset(df.columns):
        df["orders_out_lag_24"] = df.groupby("grid_id", observed=True)["orders_out"].shift(24)
    if "orders_out_rolling_3h" not in df.columns and {"grid_id", "orders_out"}.issubset(df.columns):
        df["orders_out_rolling_3h"] = (
            df.groupby("grid_id", observed=True)["orders_out"]
              .shift(1)
              .rolling(3, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if "orders_out_rolling_24h" not in df.columns and {"grid_id", "orders_out"}.issubset(df.columns):
        df["orders_out_rolling_24h"] = (
            df.groupby("grid_id", observed=True)["orders_out"]
              .shift(1)
              .rolling(24, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
    if "orders_in_lag_1" not in df.columns and {"grid_id", "orders_in"}.issubset(df.columns):
        df["orders_in_lag_1"] = df.groupby("grid_id", observed=True)["orders_in"].shift(1)

    log_sources = {
        "log1p_orders_out_lag_1": "orders_out_lag_1",
        "log1p_orders_out_lag_2": "orders_out_lag_2",
        "log1p_orders_out_lag_24": "orders_out_lag_24",
        "log1p_orders_out_rolling_3h": "orders_out_rolling_3h",
        "log1p_orders_out_rolling_24h": "orders_out_rolling_24h",
        "log1p_orders_in": "orders_in",
        "log1p_orders_in_lag_1": "orders_in_lag_1",
    }
    for out_c, src in log_sources.items():
        if out_c not in df.columns and src in df.columns:
            df[out_c] = np.log1p(pd.to_numeric(df[src], errors="coerce").fillna(0).clip(lower=0))

    if "log1p_net_in_abs" not in df.columns and "net_flow" in df.columns:
        df["log1p_net_in_abs"] = np.log1p(pd.to_numeric(df["net_flow"], errors="coerce").fillna(0).abs())

    # POI log 特征，缺则由原始 count 构造
    for c in POI_FEATURES:
        raw = c.replace("log1p_", "")
        if c not in df.columns and raw in df.columns:
            df[c] = np.log1p(pd.to_numeric(df[raw], errors="coerce").fillna(0).clip(lower=0))

    # 类型清理
    for c in BASE_NUM_FEATURES + POI_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in CAT_FEATURES:
        if c in df.columns:
            if c in ["hour_bj", "weekday"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).astype(str)
            else:
                df[c] = df[c].astype(str).fillna("unknown")

    return df


def select_model_rows(df: pd.DataFrame, cfg: Stage4Config) -> pd.DataFrame:
    data = df.copy()
    if cfg.use_active_grid_only and "is_active_grid" in data.columns:
        data = data[data["is_active_grid"].fillna(0).astype(int) == 1].copy()
    data[cfg.target_col] = pd.to_numeric(data[cfg.target_col], errors="coerce").fillna(0)
    # 不能用于模型的前24小时 lag 缺失行，如果缺失已填0可以保留；这里保留，但删除无时间行
    data = data.dropna(subset=["date_bj", "hour_bj"]).copy()
    return data


def make_train_test_split(data: pd.DataFrame, cfg: Stage4Config) -> Tuple[pd.Series, pd.Series, List[str], List[str]]:
    dates = sorted(data["date_bj"].astype(str).unique().tolist())
    if len(dates) <= cfg.test_days:
        raise ValueError(f"可用日期数 {len(dates)} 不足以划分 test_days={cfg.test_days}")
    test_dates = dates[-cfg.test_days:]
    train_dates = dates[:-cfg.test_days]
    train_mask = data["date_bj"].astype(str).isin(train_dates)
    test_mask = data["date_bj"].astype(str).isin(test_dates)

    # 理论需求训练样本：剔除供给紧张代理时段，降低“缺车导致订单偏低”的学习偏差
    if cfg.train_stable_only and "is_supply_tight_proxy" in data.columns:
        train_mask = train_mask & (data["is_supply_tight_proxy"].fillna(0).astype(int) == 0)
    if cfg.train_stable_only and "shortage_proxy" in data.columns:
        # 剔除短缺压力最高10%训练样本
        q = data.loc[train_mask, "shortage_proxy"].quantile(0.90)
        if pd.notna(q):
            train_mask = train_mask & (data["shortage_proxy"].fillna(0) <= q)
    return train_mask, test_mask, train_dates, test_dates


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """构建设计矩阵，并严格保留 df.index。

    修复点：上一版在拼接 dummy 变量时 reset_index(drop=True)，
    会让 X_all 与 train_mask/test_mask 的索引不一致，从而触发
    pandas.errors.IndexingError: Unalignable boolean Series。
    """
    use_num = [c for c in feature_cols if c in df.columns and c not in CAT_FEATURES]
    use_cat = [c for c in CAT_FEATURES if c in feature_cols and c in df.columns]

    X_num = df[use_num].copy()
    for c in use_num:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    X_num.index = df.index

    if use_cat:
        X_cat = pd.get_dummies(df[use_cat].astype(str), prefix=use_cat, dummy_na=False)
        X_cat.index = df.index
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num

    X.index = df.index
    return X.astype(np.float32)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred, name: str, split: str) -> Dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": name,
        "split": split,
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "mean_actual": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def baseline_grid_hour_mean(data: pd.DataFrame, train_mask: pd.Series) -> pd.Series:
    train = data.loc[train_mask].copy()
    # 按 grid_id + hour_bj 历史均值，缺失则用 hour 均值，再缺失用全局均值
    gh = train.groupby(["grid_id", "hour_bj"], observed=True)["orders_out"].mean().rename("pred_gh")
    h = train.groupby("hour_bj", observed=True)["orders_out"].mean().rename("pred_h")
    global_mean = float(train["orders_out"].mean())

    tmp = data[["grid_id", "hour_bj"]].copy()
    tmp = tmp.merge(gh.reset_index(), on=["grid_id", "hour_bj"], how="left")
    tmp = tmp.merge(h.reset_index(), on="hour_bj", how="left")
    pred = tmp["pred_gh"].fillna(tmp["pred_h"]).fillna(global_mean)
    return pred.clip(lower=0)


def fit_hgb(X_train, y_log_train, cfg: Stage4Config):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_log_train)
    return model


def fit_rf_optional(X_train, y_log_train, cfg: Stage4Config):
    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_log_train)
    return model


def plot_performance(perf: pd.DataFrame, fig_dir: Path, dpi: int):
    test = perf[perf["split"] == "test"].copy()
    if test.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = test["model"].tolist()
    vals = test["rmse"].to_numpy()
    ax.bar(labels, vals)
    ax.set_title("需求预测模型测试集 RMSE 对比")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage4_model_rmse.png", dpi=dpi)
    plt.close(fig)


def plot_pred_actual(df: pd.DataFrame, fig_dir: Path, dpi: int, sample_n: int = 80000):
    test = df[df["split"] == "test"].copy()
    if test.empty:
        return
    if len(test) > sample_n:
        test = test.sample(sample_n, random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(test["orders_out"], test["predicted_demand"], s=6, alpha=0.15)
    lim = np.nanpercentile(np.r_[test["orders_out"], test["predicted_demand"]], 99)
    lim = max(lim, 1)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("实际出发订单")
    ax.set_ylabel("预测理论需求")
    ax.set_title("测试集：实际订单与预测理论需求")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage4_predicted_vs_actual.png", dpi=dpi)
    plt.close(fig)


def plot_gap_by_hour(df: pd.DataFrame, fig_dir: Path, dpi: int):
    g = df.groupby("hour_bj", observed=True).agg(
        actual_orders=("orders_out", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
    ).reset_index()
    g["hour_num"] = pd.to_numeric(g["hour_bj"], errors="coerce")
    g = g.sort_values("hour_num")
    g.to_csv(fig_dir.parent / "stage4_hour_gap_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(g["hour_num"], g["actual_orders"], marker="o", label="实际订单")
    ax.plot(g["hour_num"], g["predicted_demand"], marker="o", label="预测理论需求")
    ax.bar(g["hour_num"], g["unmet_demand_proxy"], alpha=0.35, label="潜在未满足需求代理")
    ax.set_title("按小时汇总的预测需求与缺口代理")
    ax.set_xlabel("北京时间小时")
    ax.set_ylabel("订单/需求量")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage4_hourly_gap.png", dpi=dpi)
    plt.close(fig)


def plot_gap_spatial(grid_gap: pd.DataFrame, fig_dir: Path, dpi: int):
    if not {"grid_center_lng", "grid_center_lat", "unmet_demand_proxy"}.issubset(grid_gap.columns):
        return
    df = grid_gap.copy()
    v = df["unmet_demand_proxy"].fillna(0)
    cap = v.quantile(0.99) if len(v) else 0
    if cap <= 0:
        cap = max(v.max(), 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["grid_center_lng"], df["grid_center_lat"],
        c=np.clip(v, 0, cap), s=12, alpha=0.75,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.82)
    cb.set_label("潜在未满足需求代理量（裁剪至99分位）")
    ax.set_title("潜在未满足需求代理量空间分布")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage4_unmet_gap_spatial.png", dpi=dpi)
    plt.close(fig)


def run(cfg: Stage4Config) -> None:
    if not SKLEARN_AVAILABLE:
        raise ImportError("Stage 4 需要 scikit-learn。请先安装：pip install scikit-learn")

    stage3_dir = Path(cfg.stage3_dir)
    in_path = stage3_dir / cfg.input_file
    if not in_path.exists():
        alt = stage3_dir.parent / "stage2_research_ready" / "model_ready_grid_hour_panel_with_poi.parquet"
        raise FileNotFoundError(f"找不到输入文件：{in_path}\n请确认 Stage 3 已完成。")

    out_dir = stage3_dir.parent / cfg.out_dir_name if stage3_dir.name == "stage3_poi_mechanism" else stage3_dir / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures_stage4"
    if cfg.make_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df = ensure_stage4_features(df)
    data = select_model_rows(df, cfg).reset_index(drop=True)

    train_mask, test_mask, train_dates, test_dates = make_train_test_split(data, cfg)
    train_idx = data.index[train_mask].to_numpy()
    if cfg.max_train_rows and cfg.max_train_rows > 0 and len(train_idx) > cfg.max_train_rows:
        rng = np.random.default_rng(cfg.random_state)
        train_idx = rng.choice(train_idx, size=cfg.max_train_rows, replace=False)
        train_mask_fit = data.index.isin(train_idx)
    else:
        train_mask_fit = train_mask

    feature_cols = [c for c in (POI_FEATURES + BASE_NUM_FEATURES + CAT_FEATURES) if c in data.columns]
    X_all = build_feature_matrix(data, feature_cols)
    y = pd.to_numeric(data[cfg.target_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
    y_log = np.log1p(y)

    X_train = X_all.loc[train_mask_fit].copy()
    y_train_log = y_log[train_mask_fit]

    # baseline
    data["pred_baseline_grid_hour_mean"] = baseline_grid_hour_mean(data, train_mask)

    # HGB 主模型
    hgb = fit_hgb(X_train, y_train_log, cfg)
    pred_hgb_log = hgb.predict(X_all)
    pred_hgb = np.expm1(pred_hgb_log)
    pred_hgb = np.clip(pred_hgb, cfg.min_prediction, None)
    data["pred_hgb"] = pred_hgb

    # RF 对照模型：默认只在训练样本较小时启用，避免过慢
    rf_enabled = X_train.shape[0] <= 350000
    if rf_enabled:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = fit_rf_optional(X_train, y_train_log, cfg)
        pred_rf = np.clip(np.expm1(rf.predict(X_all)), cfg.min_prediction, None)
        data["pred_rf"] = pred_rf
    else:
        data["pred_rf"] = np.nan

    # 选择主预测：优先 HGB
    data["predicted_demand"] = data["pred_hgb"]
    data["actual_demand"] = data[cfg.target_col]

    gap = (data["predicted_demand"] - data["actual_demand"]).clip(lower=0)
    if cfg.gap_cap_quantile and 0 < cfg.gap_cap_quantile < 1:
        cap = gap.quantile(cfg.gap_cap_quantile)
        if pd.notna(cap) and cap > 0:
            data["unmet_demand_proxy_raw"] = gap
            gap = gap.clip(upper=cap)
    data["unmet_demand_proxy"] = gap
    data["gap_ratio_proxy"] = data["unmet_demand_proxy"] / (data["predicted_demand"] + 1e-6)
    data["split"] = np.where(train_mask, "train", np.where(test_mask, "test", "other"))

    # 评价
    perf_rows = []
    for split_name, mask in [("train_fit", train_mask_fit), ("test", test_mask)]:
        if mask.sum() == 0:
            continue
        perf_rows.append(evaluate(y[mask], data.loc[mask, "pred_baseline_grid_hour_mean"], "baseline_grid_hour_mean", split_name))
        perf_rows.append(evaluate(y[mask], data.loc[mask, "pred_hgb"], "hist_gradient_boosting_log", split_name))
        if rf_enabled:
            perf_rows.append(evaluate(y[mask], data.loc[mask, "pred_rf"], "random_forest_log", split_name))
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out_dir / "stage4_model_performance.csv", index=False, encoding="utf-8-sig")

    # 汇总表
    grid_gap = data.groupby("grid_id", observed=True).agg(
        actual_orders=("actual_demand", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        mean_gap_ratio_proxy=("gap_ratio_proxy", "mean"),
        grid_center_lng=("grid_center_lng", "first"),
        grid_center_lat=("grid_center_lat", "first"),
    ).reset_index()
    grid_gap["gap_rank_pct"] = grid_gap["unmet_demand_proxy"].rank(pct=True)
    grid_gap.to_csv(out_dir / "stage4_grid_gap_summary.csv", index=False, encoding="utf-8-sig")
    grid_gap.to_parquet(out_dir / "stage4_grid_gap_summary.parquet", index=False)

    period_gap = data.groupby("period_24h", observed=True).agg(
        actual_orders=("actual_demand", "sum"),
        predicted_demand=("predicted_demand", "sum"),
        unmet_demand_proxy=("unmet_demand_proxy", "sum"),
        rows=("grid_id", "size"),
    ).reset_index()
    period_gap["period_24h"] = pd.Categorical(period_gap["period_24h"], categories=PERIOD_ORDER, ordered=True)
    period_gap = period_gap.sort_values("period_24h")
    period_gap.to_csv(out_dir / "stage4_period_gap_summary.csv", index=False, encoding="utf-8-sig")

    high_gap = grid_gap[grid_gap["gap_rank_pct"] >= 0.90].sort_values("unmet_demand_proxy", ascending=False)
    high_gap.to_csv(out_dir / "stage4_high_gap_grids_top10pct.csv", index=False, encoding="utf-8-sig")

    # 保存主面板
    keep_cols = [
        "grid_id", "date_bj", "time_bin_bj", "hour_bj", "weekday", "is_weekend", "period_24h",
        "actual_demand", "orders_out", "orders_in", "net_flow",
        "pred_baseline_grid_hour_mean", "pred_hgb", "pred_rf", "predicted_demand",
        "unmet_demand_proxy", "unmet_demand_proxy_raw", "gap_ratio_proxy", "split",
        "grid_center_lng", "grid_center_lat",
        "is_supply_rich_proxy", "is_supply_tight_proxy", "supply_rich_score", "supply_tight_score",
    ]
    keep_cols = [c for c in keep_cols if c in data.columns]
    # 加 POI 和特征列，便于 Stage 5 使用
    extra_cols = [c for c in POI_FEATURES + [p.replace("log1p_", "") for p in POI_FEATURES] if c in data.columns]
    out_panel = data[keep_cols + [c for c in extra_cols if c not in keep_cols]].copy()
    out_panel.to_parquet(out_dir / "demand_prediction_panel.parquet", index=False)
    out_panel.head(20000).to_csv(out_dir / "demand_prediction_panel_sample.csv", index=False, encoding="utf-8-sig")

    # 特征重要性（置换重要性，仅测试集抽样）
    importance_status = "not_run"
    if cfg.max_importance_rows and cfg.max_importance_rows > 0 and test_mask.sum() > 1000:
        try:
            test_indices = data.index[test_mask].to_numpy()
            rng = np.random.default_rng(cfg.random_state)
            if len(test_indices) > cfg.max_importance_rows:
                test_indices = rng.choice(test_indices, cfg.max_importance_rows, replace=False)
            X_imp = X_all.loc[test_indices]
            y_imp = y_log[data.index.isin(test_indices)]
            imp = permutation_importance(hgb, X_imp, y_imp, n_repeats=5, random_state=cfg.random_state, n_jobs=-1)
            imp_df = pd.DataFrame({
                "feature": X_imp.columns,
                "importance_mean": imp.importances_mean,
                "importance_std": imp.importances_std,
            }).sort_values("importance_mean", ascending=False)
            imp_df.to_csv(out_dir / "stage4_hgb_permutation_importance.csv", index=False, encoding="utf-8-sig")
            importance_status = "ok"
        except Exception as exc:
            importance_status = f"error: {exc}"

    if cfg.make_figures:
        plot_performance(perf, fig_dir, cfg.figure_dpi)
        plot_pred_actual(data, fig_dir, cfg.figure_dpi)
        plot_gap_by_hour(data, fig_dir, cfg.figure_dpi)
        plot_gap_spatial(grid_gap, fig_dir, cfg.figure_dpi)

    manifest = {
        "principle": "Stage 4 基于 Stage 3 订单-POI面板，不重新读取原始订单，不调用高德接口。",
        "config": asdict(cfg),
        "input": str(in_path),
        "out_dir": str(out_dir),
        "rows_input": int(len(df)),
        "rows_model": int(len(data)),
        "train_dates": train_dates,
        "test_dates": test_dates,
        "train_stable_only": cfg.train_stable_only,
        "train_rows_before_sample": int(train_mask.sum()),
        "train_rows_fit": int(train_mask_fit.sum()),
        "test_rows": int(test_mask.sum()),
        "features_used_count": int(X_all.shape[1]),
        "base_feature_cols": feature_cols,
        "rf_enabled": bool(rf_enabled),
        "importance_status": importance_status,
        "core_outputs": [
            "demand_prediction_panel.parquet",
            "stage4_model_performance.csv",
            "stage4_grid_gap_summary.csv",
            "stage4_period_gap_summary.csv",
            "stage4_high_gap_grids_top10pct.csv",
        ],
        "figures": [p.name for p in sorted(fig_dir.glob("*.png"))] if fig_dir.exists() else [],
        "note": "unmet_demand_proxy 是预测理论需求高于实际订单的代理差额，不等同真实未成交需求。",
    }
    with open(out_dir / "stage4_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n========== Stage 4 完成 ==========")
    print(f"输入：{in_path}")
    print(f"输出目录：{out_dir.resolve()}")
    print(f"模型行数：{len(data):,}；训练日期：{train_dates}；测试日期：{test_dates}")
    print("核心文件：stage4_manifest.json, stage4_model_performance.csv, demand_prediction_panel.parquet")


def parse_args() -> Stage4Config:
    parser = argparse.ArgumentParser(description="Stage 4: 潜在需求预测与缺口代理测度")
    parser.add_argument("--stage3-dir", required=True, help="Stage 3 输出目录，如 .\\bike_output_bj_corrected\\stage3_poi_mechanism")
    parser.add_argument("--out-dir-name", default="stage4_demand_gap")
    parser.add_argument("--input-file", default="model_data_with_poi_transformed.parquet")
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=300000, help="0 表示不抽样，使用全部训练样本")
    parser.add_argument("--max-importance-rows", type=int, default=30000)
    parser.add_argument("--gap-cap-quantile", type=float, default=0.995)
    parser.add_argument("--no-train-stable-only", action="store_true", help="不剔除供给紧张代理样本，直接用全部训练集")
    parser.add_argument("--include-inactive-grid", action="store_true", help="不限制 is_active_grid==1")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    return Stage4Config(
        stage3_dir=args.stage3_dir,
        out_dir_name=args.out_dir_name,
        input_file=args.input_file,
        test_days=args.test_days,
        max_train_rows=args.max_train_rows,
        max_importance_rows=args.max_importance_rows,
        gap_cap_quantile=args.gap_cap_quantile,
        train_stable_only=not args.no_train_stable_only,
        use_active_grid_only=not args.include_inactive_grid,
        make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    run(parse_args())
