r"""
Stage 3B：POI机制模型修正与补充

目标：
1. 基于 Stage 3 的订单-POI合并面板，不重新读取原始订单、不调用高德接口；
2. 处理 Stage 3 中二项 Logit 出现 Singular matrix 的问题；
3. 补充更适合计数型订单数据的负二项模型（Negative Binomial）；
4. 对供给富裕/紧张代理区使用正则化 Logistic 回归，避免完全共线或准完全分离；
5. 输出可直接用于论文结果整理的模型表和图。

运行示例（PowerShell 一行）：
python .\bike_stage3b_refined_models.py --stage3-dir .\bike_output_bj_corrected\stage3_poi_mechanism

依赖：
pip install pandas numpy pyarrow statsmodels scikit-learn matplotlib
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:
    raise ImportError("请先安装 matplotlib：pip install matplotlib") from exc


def setup_matplotlib() -> None:
    mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["font.size"] = 10


setup_matplotlib()


@dataclass
class Config:
    stage3_dir: str
    out_dir_name: str = "stage3b_refined_models"
    max_model_rows: int = 300000  # 0 = 全样本；默认抽样以提高稳定性
    random_state: int = 42
    use_active_grid_only: bool = True
    logistic_test_size: float = 0.25
    logistic_c: float = 1.0
    negative_binomial_alpha: float = 1.0


POI_MAIN = [
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

POI_CN = {
    "log1p_poi_office_count": "办公/产业",
    "log1p_poi_commercial_count": "商业消费",
    "log1p_poi_residential_count": "居住",
    "log1p_poi_metro_count": "地铁",
    "log1p_poi_bus_count": "公交",
    "log1p_poi_education_count": "教育",
    "log1p_poi_recreation_count": "休闲游憩",
    "log1p_poi_public_service_count": "公共服务",
    "log1p_poi_medical_count": "医疗",
    "log1p_poi_transport_other_count": "其他交通",
}

CONTROL_VARS = [
    "log1p_orders_out_lag_1",
    "log1p_orders_out_lag_2",
    "log1p_orders_out_lag_24",
    "grid_center_lng",
    "grid_center_lat",
]


# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def load_panel(stage3_dir: Path) -> pd.DataFrame:
    candidates = [
        stage3_dir / "model_data_with_poi_transformed.parquet",
        stage3_dir.parent / "stage2_research_ready" / "model_ready_grid_hour_panel_with_poi.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            return df
    raise FileNotFoundError("未找到 model_data_with_poi_transformed.parquet 或 model_ready_grid_hour_panel_with_poi.parquet")


def prepare_data(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str], Dict]:
    data = df.copy()

    # 时间字段标准化
    if "date_bj" in data.columns:
        data["date_bj"] = data["date_bj"].astype(str)
    if "hour_bj" in data.columns:
        data["hour_bj"] = pd.to_numeric(data["hour_bj"], errors="coerce")
        data = data.dropna(subset=["hour_bj"]).copy()
        data["hour_bj"] = data["hour_bj"].astype(int).astype(str)
    if "period_24h" in data.columns:
        data["period_24h"] = data["period_24h"].astype(str)

    # 构造缺失的 log1p 变量
    for raw in [c.replace("log1p_", "") for c in POI_MAIN]:
        logc = "log1p_" + raw
        if logc not in data.columns and raw in data.columns:
            data[logc] = np.log1p(pd.to_numeric(data[raw], errors="coerce").fillna(0))

    for c in ["orders_out", "orders_in", "net_flow", "is_supply_rich_proxy", "is_supply_tight_proxy"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    if "log1p_orders_out" not in data.columns and "orders_out" in data.columns:
        data["log1p_orders_out"] = np.log1p(data["orders_out"].fillna(0))
    for c in ["orders_out_lag_1", "orders_out_lag_2", "orders_out_lag_24"]:
        logc = "log1p_" + c
        if logc not in data.columns and c in data.columns:
            data[logc] = np.log1p(pd.to_numeric(data[c], errors="coerce").fillna(0))

    # 使用可用主 POI 变量，不纳入 poi_total 和 other，降低共线性
    poi_vars = [c for c in POI_MAIN if c in data.columns]

    needed = ["orders_out", "log1p_orders_out", "date_bj", "hour_bj"] + poi_vars + [c for c in CONTROL_VARS if c in data.columns]
    needed = list(dict.fromkeys(needed))
    data = data.dropna(subset=[c for c in needed if c in data.columns]).copy()

    # 供给状态模型只在活跃网格上更合理。若字段不存在，则跳过。
    if cfg.use_active_grid_only and "is_active_grid" in data.columns:
        data = data[data["is_active_grid"].astype(int).eq(1)].copy()

    before_sample_rows = len(data)
    if cfg.max_model_rows and cfg.max_model_rows > 0 and len(data) > cfg.max_model_rows:
        data = data.sample(n=cfg.max_model_rows, random_state=cfg.random_state).copy()

    # statsmodels/patsy 对 pandas 扩展类型（Int64Dtype、boolean 等）兼容性较差；
    # 建模前统一转换，避免 Cannot interpret Int64Dtype。
    for c in data.columns:
        dtype_name = str(data[c].dtype)
        if dtype_name in {"Int64", "Int32", "Int16", "Int8", "UInt64", "UInt32", "UInt16", "UInt8", "boolean"}:
            data[c] = pd.to_numeric(data[c], errors="coerce").astype(float)
    if "hour_bj" in data.columns:
        data["hour_bj"] = pd.to_numeric(data["hour_bj"], errors="coerce").astype(int).astype(str)
    if "date_bj" in data.columns:
        data["date_bj"] = data["date_bj"].astype(str)

    meta = {
        "rows_after_filter_before_sample": int(before_sample_rows),
        "rows_used": int(len(data)),
        "sample_used": bool(before_sample_rows != len(data)),
        "poi_vars_used": poi_vars,
        "use_active_grid_only": cfg.use_active_grid_only,
    }
    return data, poi_vars, meta


def tidy_statsmodels_result(res, terms: List[str], model_name: str) -> pd.DataFrame:
    rows = []
    conf = res.conf_int() if hasattr(res, "conf_int") else None
    for term in terms:
        if term not in res.params.index:
            continue
        row = {
            "model": model_name,
            "term": term,
            "term_cn": POI_CN.get(term, term),
            "coef": float(res.params[term]),
            "std_err": float(res.bse[term]) if hasattr(res, "bse") and term in res.bse.index else np.nan,
            "p_value": float(res.pvalues[term]) if hasattr(res, "pvalues") and term in res.pvalues.index else np.nan,
        }
        if conf is not None and term in conf.index:
            row["ci_low"] = float(conf.loc[term, 0])
            row["ci_high"] = float(conf.loc[term, 1])
        else:
            row["ci_low"] = np.nan
            row["ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_coeff_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("coef").copy()
    labels = plot_df["term_cn"].fillna(plot_df["term"])
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.42)))
    xerr = None
    if {"ci_low", "ci_high"}.issubset(plot_df.columns) and plot_df[["ci_low", "ci_high"]].notna().all().all():
        xerr = np.vstack([plot_df["coef"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["coef"]])
    ax.errorbar(plot_df["coef"], y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("系数估计值及95%置信区间")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 模型
# ------------------------------------------------------------

def run_ols_reduced(data: pd.DataFrame, poi_vars: List[str], out_dir: Path) -> Dict:
    formula = (
        "log1p_orders_out ~ "
        + " + ".join(poi_vars)
        + " + C(date_bj) + C(hour_bj)"
        + " + " + " + ".join([c for c in CONTROL_VARS if c in data.columns])
    )
    res = smf.ols(formula, data=data).fit(cov_type="HC1")
    coef = tidy_statsmodels_result(res, poi_vars, "OLS_log_reduced_HC1")
    coef.to_csv(out_dir / "model_ols_log_orders_reduced_hc1.csv", index=False, encoding="utf-8-sig")
    save_coeff_plot(coef, out_dir / "fig_stage3b_ols_reduced_coefficients.png", "POI变量对出发订单强度的影响（OLS-log，稳健标准误）")
    return {
        "formula": formula,
        "nobs": int(res.nobs),
        "rsquared": float(res.rsquared),
        "aic": float(res.aic),
    }


def run_poisson_reduced(data: pd.DataFrame, poi_vars: List[str], out_dir: Path) -> Dict:
    formula = (
        "orders_out ~ "
        + " + ".join(poi_vars)
        + " + C(date_bj) + C(hour_bj)"
        + " + " + " + ".join([c for c in CONTROL_VARS if c in data.columns])
    )
    res = smf.glm(formula, data=data, family=sm.families.Poisson()).fit(cov_type="HC1")
    coef = tidy_statsmodels_result(res, poi_vars, "Poisson_reduced_HC1")
    coef.to_csv(out_dir / "model_poisson_orders_reduced_hc1.csv", index=False, encoding="utf-8-sig")
    save_coeff_plot(coef, out_dir / "fig_stage3b_poisson_reduced_coefficients.png", "POI变量对出发订单量的影响（Poisson，稳健标准误）")
    pearson = float(np.sum(res.resid_pearson ** 2))
    df_resid = float(res.df_resid) if res.df_resid else np.nan
    return {
        "formula": formula,
        "nobs": int(res.nobs),
        "aic": float(res.aic),
        "pearson_chi2_over_df": pearson / df_resid if df_resid else np.nan,
    }


def run_negative_binomial(data: pd.DataFrame, poi_vars: List[str], out_dir: Path, alpha: float) -> Dict:
    formula = (
        "orders_out ~ "
        + " + ".join(poi_vars)
        + " + C(date_bj) + C(hour_bj)"
        + " + " + " + ".join([c for c in CONTROL_VARS if c in data.columns])
    )
    res = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(alpha=alpha)).fit(cov_type="HC1", maxiter=100)
    coef = tidy_statsmodels_result(res, poi_vars, f"NegativeBinomial_alpha_{alpha:g}")
    coef.to_csv(out_dir / "model_negative_binomial_orders_main.csv", index=False, encoding="utf-8-sig")
    save_coeff_plot(coef, out_dir / "fig_stage3b_negative_binomial_coefficients.png", "POI变量对出发订单量的影响（负二项GLM）")
    return {
        "formula": formula,
        "nobs": int(res.nobs),
        "aic": float(res.aic),
        "alpha": float(alpha),
    }


def run_regularized_logistic(data: pd.DataFrame, poi_vars: List[str], target: str, out_dir: Path, cfg: Config) -> Dict:
    if target not in data.columns:
        return {"target": target, "status": "missing_target"}

    controls = [c for c in CONTROL_VARS if c in data.columns]
    # 加入小时和日期虚拟变量，使用 pandas get_dummies，正则化模型可承受共线风险。
    use_cols = poi_vars + controls + ["date_bj", "hour_bj"]
    model_df = data[use_cols + [target]].dropna().copy()
    model_df[target] = model_df[target].astype(int)
    class_counts = model_df[target].value_counts().to_dict()
    if len(class_counts) < 2:
        return {"target": target, "status": "single_class", "class_counts": {str(k): int(v) for k, v in class_counts.items()}}

    X = pd.get_dummies(model_df[use_cols], columns=["date_bj", "hour_bj"], drop_first=True)
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.logistic_test_size, random_state=cfg.random_state, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("logit", LogisticRegression(
            penalty="l2", C=cfg.logistic_c, solver="lbfgs", max_iter=1000, class_weight="balanced", n_jobs=None
        )),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    coef = pipe.named_steps["logit"].coef_.ravel()
    coef_df = pd.DataFrame({"term": X.columns, "coef_standardized": coef})
    coef_df = coef_df[coef_df["term"].isin(poi_vars)].copy()
    coef_df["term_cn"] = coef_df["term"].map(POI_CN).fillna(coef_df["term"])
    coef_df["target"] = target
    out_name = f"model_regularized_logit_{target}.csv"
    coef_df.to_csv(out_dir / out_name, index=False, encoding="utf-8-sig")

    # 作图
    plot_df = coef_df.sort_values("coef_standardized")
    if not plot_df.empty:
        y_pos = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(8, max(4, len(plot_df) * 0.42)))
        ax.scatter(plot_df["coef_standardized"], y_pos)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["term_cn"])
        ax.set_xlabel("标准化系数（L2正则化Logistic）")
        title_cn = "供给富裕代理" if "rich" in target else "供给紧张代理"
        ax.set_title(f"POI变量对{title_cn}的影响")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_stage3b_regularized_logit_{target}.png", bbox_inches="tight")
        plt.close(fig)

    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) == 2 else np.nan
    metrics = {
        "target": target,
        "status": "ok",
        "nobs": int(len(model_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "roc_auc": float(auc),
        "regularization": "L2",
        "C": float(cfg.logistic_c),
        "class_weight": "balanced",
        "note": "正则化Logistic用于替代传统Logit的Singular matrix问题，系数反映方向与相对强度，不提供传统p值。",
    }
    return metrics


def make_model_comparison(out_dir: Path, summaries: Dict) -> None:
    rows = []
    for name, meta in summaries.items():
        if isinstance(meta, dict) and "aic" in meta:
            rows.append({"model": name, "nobs": meta.get("nobs"), "aic": meta.get("aic"), "extra": meta.get("pearson_chi2_over_df", meta.get("alpha", np.nan))})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "model_count_comparison.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3B：补充负二项与正则化Logistic供给状态模型")
    parser.add_argument("--stage3-dir", required=True, help="Stage 3 输出目录，例如 .\\bike_output_bj_corrected\\stage3_poi_mechanism")
    parser.add_argument("--out-dir-name", default="stage3b_refined_models", help="Stage 3B 输出子目录名")
    parser.add_argument("--max-model-rows", type=int, default=300000, help="模型最大样本行数；0表示全样本")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-inactive-grids", action="store_true", help="默认只用活跃网格；加此参数则包含非活跃网格")
    parser.add_argument("--negative-binomial-alpha", type=float, default=1.0, help="负二项GLM alpha 参数")
    parser.add_argument("--logistic-c", type=float, default=1.0, help="正则化Logistic的C参数")
    args = parser.parse_args()

    cfg = Config(
        stage3_dir=args.stage3_dir,
        out_dir_name=args.out_dir_name,
        max_model_rows=args.max_model_rows,
        random_state=args.random_state,
        use_active_grid_only=not args.include_inactive_grids,
        negative_binomial_alpha=args.negative_binomial_alpha,
        logistic_c=args.logistic_c,
    )

    stage3_dir = Path(cfg.stage3_dir)
    out_dir = stage3_dir / cfg.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(stage3_dir)
    data, poi_vars, prep_meta = prepare_data(panel, cfg)

    # 固化本阶段模型数据小样，避免后续反复读取大表检查。
    data.head(10000).to_csv(out_dir / "stage3b_model_data_sample.csv", index=False, encoding="utf-8-sig")
    data.to_parquet(out_dir / "stage3b_model_data.parquet", index=False)

    summaries: Dict[str, object] = {
        "config": asdict(cfg),
        "preparation": prep_meta,
        "statsmodels_available": STATSMODELS_AVAILABLE,
        "sklearn_available": SKLEARN_AVAILABLE,
    }

    if STATSMODELS_AVAILABLE:
        summaries["ols_reduced_hc1"] = run_ols_reduced(data, poi_vars, out_dir)
        summaries["poisson_reduced_hc1"] = run_poisson_reduced(data, poi_vars, out_dir)
        try:
            summaries["negative_binomial_main"] = run_negative_binomial(data, poi_vars, out_dir, cfg.negative_binomial_alpha)
        except Exception as exc:
            summaries["negative_binomial_main_error"] = str(exc)
    else:
        summaries["statsmodels_error"] = "statsmodels not available"

    if SKLEARN_AVAILABLE:
        for target in ["is_supply_rich_proxy", "is_supply_tight_proxy"]:
            try:
                summaries[f"regularized_logit_{target}"] = run_regularized_logistic(data, poi_vars, target, out_dir, cfg)
            except Exception as exc:
                summaries[f"regularized_logit_{target}_error"] = str(exc)
    else:
        summaries["sklearn_error"] = "scikit-learn not available"

    make_model_comparison(out_dir, summaries)

    # 论文解释说明
    notes = """# Stage 3B 模型说明\n\n1. Stage 3 原始 Binomial Logit 出现 `Singular matrix`，通常由高度共线、完全分离或虚拟变量过多导致。\n2. 本阶段不再强行使用非正则化 Logit，而改用 L2 正则化 Logistic 回归，作为供给富裕/紧张代理的辅助机制模型。\n3. Poisson 模型若 Pearson chi2/df 明显大于 1，说明存在过度离散；因此本阶段补充 Negative Binomial GLM。\n4. 正则化 Logistic 系数没有传统 p 值，适合解释方向与相对强度，不建议作为唯一显著性证据。\n5. 主文建议以 OLS-log 与负二项模型解释订单强度，以正则化 Logistic 作为供给状态代理的稳健性/辅助分析。\n"""
    (out_dir / "stage3b_model_notes.md").write_text(notes, encoding="utf-8")

    with open(out_dir / "stage3b_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    manifest = {
        "principle": "Stage 3B 基于 Stage 3 订单-POI合并面板，不重新读取原始订单，也不调用高德接口。",
        "input_stage3_dir": str(stage3_dir),
        "out_dir": str(out_dir),
        "core_outputs": [
            "stage3b_model_summary.json",
            "stage3b_model_notes.md",
            "model_ols_log_orders_reduced_hc1.csv",
            "model_poisson_orders_reduced_hc1.csv",
            "model_negative_binomial_orders_main.csv",
            "model_regularized_logit_is_supply_rich_proxy.csv",
            "model_regularized_logit_is_supply_tight_proxy.csv",
            "model_count_comparison.csv",
        ],
        "figures": sorted([p.name for p in out_dir.glob("fig_stage3b_*.png")]),
    }
    with open(out_dir / "stage3b_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n========== Stage 3B 完成 ==========")
    print(f"输入目录：{stage3_dir}")
    print(f"输出目录：{out_dir.resolve()}")
    print(f"模型行数：{len(data):,}；POI变量数：{len(poi_vars)}")
    print("核心文件：stage3b_model_summary.json, model_negative_binomial_orders_main.csv, model_regularized_logit_*.csv")


if __name__ == "__main__":
    main()
