# 复现与输入说明

## 环境

建议在 Python 3.11 或更新版本的虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

提交时的原始订单来自[深圳市政府数据开放平台](https://opendata.sz.gov.cn/)“共享单车企业每日订单表”接口，POI 来自[高德开放平台](https://lbs.amap.com/) Web 服务。两处接口可能要求各自的有效密钥、配额和使用权限。仓库不含密钥、原始订单、POI 点表或大型中间面板，也未对当前在线接口重新发起请求。

以下命令在仓库根目录运行；示例使用 PowerShell 环境变量，不把密钥写进脚本或仓库文件。请先在本地设置 `SZ_OPEN_DATA_APP_KEY` 和 `AMAP_WEB_KEY`。

## 可从公开代码运行的前段

```powershell
python code/bike_pipeline_part1_stable_api.py --app-key $env:SZ_OPEN_DATA_APP_KEY --start-date 20210820 --end-date 20210830 --source-timezone bj --coord-system bd09 --out-dir work/bike_output_bj_corrected

python code/amap_poi_fetch_shenzhen_economy.py --amap-key $env:AMAP_WEB_KEY --grid-summary work/bike_output_bj_corrected/h3_grid_summary.csv --out-dir work/amap_poi_economy --estimate-only

# 核对 API 调用量、权限和坐标后，移去 --estimate-only 正式获取 POI。
python code/amap_poi_fetch_shenzhen_economy.py --amap-key $env:AMAP_WEB_KEY --grid-summary work/bike_output_bj_corrected/h3_grid_summary.csv --out-dir work/amap_poi_economy

python code/bike_stage2_research_ready_v3.py --out-dir work/bike_output_bj_corrected --analysis-start-date 2021-08-20 --analysis-end-date 2021-08-28 --poi-csv work/amap_poi_economy/poi_unique_minimal.csv
```

提交归档的订单配置记录 `source_timezone=bj`，因此示例显式指定该口径；源脚本默认值与归档记录不同。接口时间定义若发生变化，应先核查样例时间戳，再决定如何解析，不能直接假设现今重新下载会得到同样结果。

## 后段的额外输入

原支撑包没有生成 `stage3_poi_mechanism/model_data_with_poi_transformed.parquet` 的 Stage 3 脚本。以下脚本需要用户自行准备该提交版中间面板及其上游输出；缺失时不能仅靠本仓库完成端到端复算。

```powershell
python code/bike_stage3b_refined_models_v2.py --stage3-dir work/bike_output_bj_corrected/stage3_poi_mechanism --max-model-rows 0
python code/bike_stage4_demand_gap_v2.py --stage3-dir work/bike_output_bj_corrected/stage3_poi_mechanism --max-train-rows 0
python code/bike_stage5_carbon_scenarios_v2.py --stage4-dir work/bike_output_bj_corrected/stage4_demand_gap --default-trip-distance-km 1.1
python code/bike_stage6_robustness_archive.py --root-dir work/bike_output_bj_corrected
```

`results/` 是提交支撑材料中的汇总表，不是这次在新环境下重跑得到的结果。特别要按 `split=test` 核对预测 RMSE；原自动摘要曾误取训练集行，公开版归档脚本已修正。方法解释见[方法与结果核对](method_notes.md)。
