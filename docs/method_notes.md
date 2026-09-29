# 方法与结果核对

## 数值来源

README 以[匿名提交版论文](../paper/submitted_paper.pdf)和提交支撑材料中的原始结果表为准。`results/table_04b_stage4_model_performance.csv` 的 `split=test` 行分别给出历史均值 29.5186、HistGradientBoosting 17.1252、随机森林 14.8557，四舍五入后与论文表 8 一致。

原支撑包的 Stage 6 自动摘要把 `split=train_fit` 的随机森林 10.2379 和历史均值 21.4822 误标为“测试集”指标。公开版 `code/bike_stage6_robustness_archive.py` 已限定从 `split=test` 行生成测试摘要；旧摘要文件未收入仓库。此改动只修正摘要读取逻辑，不重新训练模型，也不改动论文数值。

## 预测与缺口的限制

Stage 4 按日期隔离训练和测试，但特征表中还含有同一网格—小时的 `orders_in` 以及 `net_flow` 的绝对值。`net_flow` 由同一小时到达量和出发量计算，而出发量正是预测目标。因此，日期隔离不足以保证这些特征在真实预测时可得，目标信息可能进入测试特征。论文报告的 RMSE 是**原流程在留出日期上的回顾性误差**，不宜宣称为可部署模型的未来预测精度。

`unmet_demand_proxy = max(predicted_demand - actual_demand, 0)` 是预测残差的代理量。它可能同时包含供给不足、模型误差、天气或事件冲击，以及上述特征问题，不能当作真实缺车订单。若要验证真实未满足需求，需要车辆库存、用户找车或开锁失败记录，并重新构建只使用决策时可得变量的前瞻预测实验。

## POI 与减碳的限制

负二项模型、正则化 Logistic 的结果描述条件关联，不识别 POI 建设对骑行需求的因果效应。情景碳量还假设一部分潜在缺口会转化为新增骑行，并为替代出行方式与车辆调度设定排放参数；`results/stage5_scenario_assumptions.csv` 保存了三组假设。1.10 km 是本样本的平均距离近似，不是真实轨迹长度。净减碳潜力不是核证减排量。

## 公开版改动

| 文件 | 改动 |
| --- | --- |
| `paper/submitted_paper.pdf` | 移除原 PDF 中的作者与制作软件等文件属性；37 页文本一致，抽查页面渲染一致 |
| `code/bike_pipeline_part1_stable_api.py` | 移除原脚本中固化的开放平台密钥，并在输出 `pipeline_config.json` 时遮盖密钥 |
| `code/bike_stage6_robustness_archive.py` | 仅从测试集行提取“测试集最优 RMSE”与历史均值基线 |

其他模型脚本沿用提交支撑材料。公开版未对原有预测流程做方法修正，也未重跑全量原始数据。
