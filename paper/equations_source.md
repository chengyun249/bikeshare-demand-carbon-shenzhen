# 公式源码索引

本文件按[论文正文](article_text.md)中的出现顺序列出公式源码。
代码块无需 GitHub 数学渲染器，适合在公式预览被挤压或显示异常时核对符号；正式排版请以原论文为准。

## 式（1）

净流入 = 到达订单数 − 出发订单数。

```tex
NF_{i,d,t}=O^{in}_{i,d,t}-O^{out}_{i,d,t}\tag{1}
```

## 式（2）

平均骑行距离 = 所有订单距离之和 ÷ 订单数。

```tex
\bar d=\frac{1}{N}\sum_{k=1}^{N}d_k\tag{2}
```

## 式（3）

隐含库存代理 = 基础供给与当日累计净流入之和，并以 0 为下界。

```tex
S_{i,d,t}=\max\left(B_i+\sum_{r=0}^{t}NF_{i,d,r},\,0\right)\tag{3}
```

## 式（4）

某网格的某类 POI 数量 = 该网格内属于该类的 POI 个数。

```tex
P_{im}=\sum_{p\in i}\mathbf 1(\mathrm{type}_p=m)\tag{4}
```

## 式（5）

对 POI 计数作 `log(1 + 数量)` 变换。

```tex
LP_{im}=\log(1+P_{im})\tag{5}
```

## 式（6）

潜在需求缺口代理 = `max(预测需求 − 实际出发订单数, 0)`。

```tex
Gap_{i,d,t}=\max\left(\widehat D_{i,d,t}-O^{out}_{i,d,t},\,0\right)\tag{6}
```

## 式（7）

情景净减碳潜力 = 各网格、日期和小时的新增骑行减碳量之和 − 调度排放。

```tex
CE_s^{net}=\sum_{i,d,t}\alpha_s Gap_{i,d,t}\,L\,EF-CE_s^{dispatch}\tag{7}
```

## 式（8）

订单数期望 = `exp(截距 + POI 变量效应 + 其他控制变量效应)`。

```tex
\mathbb E[Y_{i,d,t}]=\exp\left(\beta_0+\sum_m\beta_m LP_{im}+\gamma X_{i,d,t}\right)\tag{8}
```

## 式（9）

订单数方差 = 均值 + 过度离散参数 × 均值的平方。

```tex
\mathrm{Var}(Y_{i,d,t})=\mu_{i,d,t}+\alpha\mu_{i,d,t}^{2}\tag{9}
```

## 式（10）

与式（6）相同：潜在需求缺口代理 = `max(预测需求 − 实际出发订单数, 0)`。

```tex
Gap_{i,d,t}=\max\left(\widehat D_{i,d,t}-O^{out}_{i,d,t},\,0\right)\tag{10}
```

## 式（11）

情景新增骑行量 = 情景实现比例 × 潜在需求缺口代理。

```tex
A^s_{i,d,t}=\alpha_s Gap_{i,d,t}\tag{11}
```

## 式（12）

情景净减碳潜力 = 新增骑行量 × 平均距离 × 单位替代减排强度的总和 − 调度排放。

```tex
CE_s^{net}=\sum_{i,d,t}A^s_{i,d,t}\,L\,EF-CE_s^{dispatch}\tag{12}
```
