# Standard-Geometry Bias Field PPR: 问题与初步方案

本文档整理当前 PPR 早期分布问题中最关键的一层困难：我们希望构造一个“全空间绝对 bias 分布”，但现有 bias 观测强烈受到样本点几何结构影响。这里记录现阶段的问题理解、统计定义、类比解释、初步解决方案和后续需要继续思考的开放问题。

这份文档不是最终实现方案，而是为了把当前思路固定下来，方便继续推敲。

## 1. 当前目标

z2 当前要解决的是 Quijote residual-anchor emulator 的早期训练点设计问题。给定一组初始点

\[
D=\{\theta_i\}_{i=1}^{N},
\]

我们用它训练 residual/logdiff emulator：

\[
r_{\mathrm{Q}}(\theta,k)
=
\log P_{\mathrm{Q,nl}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

最终评价只看 Quijote fixed GP 在统一验证集上的 \(p68\)：

\[
J(D)
=
Q_{0.68,\theta,k}
\left(
\left|
\exp\left[
\hat r_D(\theta,k)-r_{\mathrm{Q}}(\theta,k)
\right]-1
\right|
\right).
\]

PPR 或 EED-PPR 的任务不是直接替代这个终评，而是生成一个更好的早期点分布 \(D_0\)，例如 \(N=64\)，让最终 \(J(D_0)\) 更低。

## 2. 已发现的问题

### 2.1 旧 PPR 的问题

旧 PPR 使用单个 GP 的 posterior variance 构造势能面：

\[
V_{\mathrm{GP}}(\theta)
=
V(\theta\mid D,\mathcal K,\alpha).
\]

这个量严重依赖训练点 \(D\)、kernel 和 GP 状态。它不是全空间真实困难度，而是“当前样本几何 + 当前模型假设”下的 uncertainty。

更严重的是，对于低噪声 GP：

\[
V_{\mathrm{GP}}(\theta_i)\approx 0,\qquad \theta_i\in D.
\]

训练点附近会天然变成低方差节点。这会导致 PPR 势能面出现由样本点决定的零点结构。

### 2.2 bias-PPR 的问题

我们随后考虑直接使用 bias：

\[
B(\theta;D)
=
Q_{0.68,k}
\left(
\left|
\exp\left[
\hat r_D(\theta,k)-r(\theta,k)
\right]-1
\right|
\right).
\]

这比 posterior variance 更接近最终指标，但仍然存在同一个核心问题：

\[
B(\theta;D)
\]

不是位置 \(\theta\) 本身的绝对 bias，而是在样本集 \(D\) 条件下的 bias。靠近训练点时 bias 会低，远离训练点时 bias 会高，因此它仍然强烈携带样本几何信息。

### 2.3 EED-PPR 的问题

EED-PPR 用多个 emulator 的 disagreement 构造 error field：

\[
E_{\mathrm{ens}}(\theta,k)
=
\operatorname{Var}_{s=1,\ldots,S}
\left[
\hat r_s(\theta,k)
\right].
\]

它能缓解单个 GP posterior variance 的问题，但不能彻底解决样本几何污染。原因有两个：

1. 如果多个 emulator 都在某个区域系统性错得一致，那么 \(E_{\mathrm{ens}}\) 可以很低，但真实 bias 很高。
2. 如果每个 emulator 的训练样本 \(D_s\) 不同，那么每个 \(D_s\) 会产生不同的低 bias 节点、空洞和 simplex 几何结构。多个图样叠加后，可能形成类似干涉条纹的结构。

因此 raw ensemble bias 或 raw ensemble disagreement 并不一定代表位置 \(\theta\) 的绝对困难度。

## 3. 问题的统计形式

当前观测到的 bias 可以写成：

\[
Y(\theta,D)
=
I(\theta)
+
G(\theta,D)
+
\epsilon.
\]

其中：

- \(Y(\theta,D)\)：我们实际测得的 bias；
- \(I(\theta)\)：位置 \(\theta\) 的内禀困难度，也就是我们真正想用于 PPR 的目标场；
- \(G(\theta,D)\)：由训练样本几何导致的误差项；
- \(\epsilon\)：模型训练噪声、数值误差和有限样本噪声。

目前最麻烦的是 \(G(\theta,D)\)。它包含：

1. \(\theta\) 到最近训练点的距离；
2. \(\theta\) 在 Delaunay simplex 中的相对位置；
3. simplex 的尺度；
4. simplex 的形状扭曲程度；
5. 是否靠近边界；
6. 训练点局部密度；
7. GP/PCA-GP kernel 对该几何结构的响应。

所以我们真正需要的不是

\[
Y(\theta,D),
\]

而是尽可能消除或控制 \(G(\theta,D)\) 后得到的 \(I(\theta)\)。

## 4. 为什么“估计几何影响再扣除”不够稳

一种自然思路是先拟合几何项：

\[
Y(\theta,D)
=
I(\theta)+g(\phi(\theta,D))+\epsilon,
\]

其中 \(\phi(\theta,D)\) 是几何特征，例如最近邻距离、simplex volume、boundary distance 等。然后扣除：

\[
\widehat I(\theta)
=
Y(\theta,D)-\hat g(\phi(\theta,D)).
\]

但这个方案有一个根本困难：不同样本点分布 \(D\) 产生的几何影响不是同一个标准波包。每一组 \(D\) 的节点、空洞、simplex 形状、边界结构都不同。

也就是说，

\[
G(\theta,D_s)
\]

不是一个简单、可统一扣除的函数。它更像是每个样本分布 \(D_s\) 自己产生的一套干涉图样。我们没有一个确定的“标准波包”作为扣除模板。

因此，“先估计几何影响再扣除”的方法容易变成：

\[
\text{用一个平均几何模型去解释许多不同相位、不同形状的几何图样。}
\]

这可能会过度平滑，也可能把真实的 \(I(\theta)\) 一起扣掉。

## 5. 现在更直接的思路：固定相对几何位置

新的思路不是扣除几何项，而是直接控制几何条件。

也就是说，不再比较任意几何位置下的 bias，而只比较同一类标准几何位置下的 bias。

定义一个标准几何条件：

\[
\mathcal G_0
=
\{
\text{位于 simplex 中心附近},
\text{simplex 尺度相近},
\text{simplex 形状不过度扭曲},
\text{边界影响可控}
\}.
\]

然后我们估计的不是抽象的绝对 \(I(\theta)\)，而是：

\[
I_{N,\mathcal G_0}(\theta)
=
\mathbb E_D
\left[
Y(\theta,D)
\mid
\theta \text{ 满足标准几何条件 } \mathcal G_0
\right].
\]

这个量的含义是：

\[
\boxed{
\text{在 }N\text{ 个均匀训练点、标准 simplex 几何位置下，位置 }\theta\text{ 的平均困难度。}
}
\]

这比“完全样本无关的绝对 bias”更可定义，也更符合 PPR64 的真实任务。

## 6. 干涉类比

可以把单个样本设计 \(D_s\) 理解为一个单缝或一个特定波包。它会产生自己的误差图样：

\[
D_s
\rightarrow
Y(\theta,D_s).
\]

这个图样中有由训练点造成的低 bias 节点，也有由样本空洞造成的高 bias 区域。

如果直接叠加很多组：

\[
\frac{1}{S}\sum_s Y(\theta,D_s),
\]

得到的结果未必是 \(I(\theta)\)，因为不同 \(D_s\) 的几何图样不同，叠加后可能产生干涉纹、条纹或伪结构。

因此关键不是简单平均，而是先把每次观测都放到相同的相对几何位置上：

\[
\theta \in \mathcal G_0(D_s).
\]

这样每个单缝的几何相位被规范到相近状态，再叠加时更有可能得到困难度包络：

\[
\text{标准几何条件下的平均 bias 包络}
\approx
I_{N,\mathcal G_0}(\theta).
\]

## 7. 为什么 simplex 中心是自然选择

在 \(d\) 维参数空间中，一个 simplex 有 \(d+1\) 个顶点。当前是 5D，所以每个 simplex 有 6 个顶点。

simplex 的 barycentric center 对应：

\[
\lambda_0
=
\left(
\frac{1}{d+1},
\frac{1}{d+1},
\ldots,
\frac{1}{d+1}
\right).
\]

对于 5D：

\[
\lambda_0
=
\left(
\frac{1}{6},
\frac{1}{6},
\frac{1}{6},
\frac{1}{6},
\frac{1}{6},
\frac{1}{6}
\right).
\]

选择 simplex 中心附近有几个好处：

1. 不靠近训练点，因此避开训练点处 bias/variance 接近 0 的节点。
2. 不靠近 simplex 边、面或顶点，相对位置更稳定。
3. 对不同 simplex 可以用同一个 barycentric 条件定义。
4. 对 PPR 的覆盖问题有直接意义：中心位置通常代表当前局部单元中最需要被解释的空隙。

但只固定 simplex 中心还不够，因为两个 simplex 的中心可以有完全不同的尺度和形状。因此标准几何条件必须同时控制：

\[
\text{relative position}
+
\text{scale}
+
\text{shape}
+
\text{boundary}.
\]

## 8. 两种可行估计器

### 8.1 估计器 A：simplex-center stacking

第一种直接方案是：

1. 生成很多组均匀训练集：

\[
D_1,D_2,\ldots,D_S,\qquad |D_s|=N.
\]

2. 对每个 \(D_s\) 做 Delaunay triangulation。
3. 对每个 simplex \(T_{s,m}\)，计算 barycentric center：

\[
c_{s,m}
=
\frac{1}{d+1}
\sum_{v\in T_{s,m}} v.
\]

4. 只在 \(c_{s,m}\) 上计算 bias：

\[
Y_{s,m}
=
B(c_{s,m};D_s).
\]

5. 把所有 \((c_{s,m},Y_{s,m})\) 叠加，再插值得到空间场：

\[
\widehat I(\theta)
\approx
\operatorname{Interp}
\left(
\{(c_{s,m},Y_{s,m})\}_{s,m}
\right).
\]

这个方法的优点是简单，且天然避免了训练点节点。缺点是：

1. simplex center 的空间分布不一定均匀；
2. 不同 simplex 的尺度和形状不同，仍然会带入几何差异；
3. 边界附近 simplex 可能异常大或畸形；
4. 插值结果可能受到 center density 的影响。

因此 simplex-center stacking 可以作为快速 pilot，但不建议作为最终严谨版本。

### 8.2 估计器 B：fixed-reference conditional estimator

更稳的方案是固定 reference grid：

\[
R=\{\theta_j\}_{j=1}^{M}.
\]

然后对很多组均匀训练集 \(D_s\) 重复以下操作。

对每个 reference 点 \(\theta_j\)，找到它落入的 simplex：

\[
T_{s,j}
=
\operatorname{Simplex}(\theta_j;D_s).
\]

计算它在这个 simplex 中的 barycentric 坐标：

\[
\lambda_{s,j}
=
\operatorname{Barycentric}(\theta_j;T_{s,j}).
\]

然后计算该 simplex 的几何特征：

\[
h_{s,j}
=
\text{scale}(T_{s,j}),
\]

\[
\kappa_{s,j}
=
\text{shape\ distortion}(T_{s,j}),
\]

\[
b_j
=
\text{boundary\ distance}(\theta_j).
\]

只保留满足标准几何条件的观测：

\[
\|\lambda_{s,j}-\lambda_0\| \le \tau_\lambda,
\]

\[
h_- \le h_{s,j} \le h_+,
\]

\[
\kappa_{s,j}\le \kappa_{\max},
\]

\[
b_j \ge b_{\min}.
\]

在这些通过筛选的情况下，计算 bias：

\[
Y_{s,j}
=
Q_{0.68,k}
\left(
\left|
\exp\left[
\hat r_{D_s}(\theta_j,k)-r(\theta_j,k)
\right]-1
\right|
\right).
\]

然后对同一个 reference 点 \(\theta_j\) 的多次合格观测做 robust aggregation：

\[
\widehat I_{N,\mathcal G_0}(\theta_j)
=
\operatorname{median}_{s\in A_j}
Y_{s,j},
\]

其中

\[
A_j
=
\{
s:
(\theta_j,D_s)\text{ 满足 }\mathcal G_0
\}.
\]

如果 \(A_j\) 太少，可以使用 trimmed mean，或者放宽 \(\tau_\lambda\)、\(h\)、\(\kappa\) 条件。

这个 estimator 的优点是：

1. 空间位置 \(\theta_j\) 是固定的，不会被 simplex center density 扭曲。
2. 只比较同一标准几何条件下的 bias。
3. 可以显式记录每个 \(\theta_j\) 的有效样本数 \(|A_j|\)，知道哪里估计可靠。
4. 可以逐步放宽几何条件，形成 bias-field 置信度。

因此它更适合作为正式方案。

## 9. 几何条件的具体定义

### 9.1 相对位置

使用 barycentric center 条件：

\[
\|\lambda_{s,j}-\lambda_0\|_2 \le \tau_\lambda.
\]

也可以用最大偏差：

\[
\max_a |\lambda_{s,j,a}-1/(d+1)| \le \tau_{\lambda,\infty}.
\]

第一版建议从较宽条件开始，例如只要求 reference 点不靠近顶点、边或面：

\[
\min_a \lambda_{s,j,a} \ge \lambda_{\min}.
\]

这表示点在 simplex 内部，不贴近边界面。

### 9.2 simplex 尺度

可以用顶点到中心的均方距离定义尺度：

\[
h(T)
=
\left[
\frac{1}{d+1}
\sum_{a=1}^{d+1}
\|v_a-c_T\|^2
\right]^{1/2}.
\]

然后保留：

\[
h_- \le h(T)\le h_+.
\]

这里的 \(h_-\)、\(h_+\) 可以用所有 simplex 尺度的分位数定义，例如：

\[
h_- = Q_{0.25}(h),
\qquad
h_+ = Q_{0.75}(h).
\]

第一版也可以分层估计：

\[
I_{N,\mathcal G_0,h\text{-bin}}(\theta),
\]

检查不同尺度层下的 bias field 是否一致。

### 9.3 simplex 形状

形状可以用 condition number 或 radius ratio 表示。一个实用定义是构造 edge matrix：

\[
E_T=[v_1-v_0,\ldots,v_d-v_0],
\]

然后用

\[
\kappa(T)
=
\operatorname{cond}(E_T).
\]

保留：

\[
\kappa(T)\le \kappa_{\max}.
\]

这个条件可以过滤非常扁、非常畸形的 simplex。

### 9.4 边界距离

在 unit space 中，边界距离可以写成：

\[
b(\theta)
=
\min_m
\left(
\theta_m,\ 1-\theta_m
\right).
\]

边界区域可以单独处理。原因是边界本身可能就是困难区域，但边界 simplex 的几何性质也更异常。建议分成两层：

1. 内部场：\(b(\theta)\ge b_{\min}\)，用于估计主 bias field；
2. 边界场：单独估计，或后续用 boundary regularization 处理。

不要简单把所有边界点过滤掉，因为当前我们已经知道边缘作用会影响最终 \(p68\)。

## 10. bias 的数据源选择

每次 \(Y_{s,j}\) 都需要 truth 和 emulator prediction。

### 10.1 Quijote truth

最理想的是直接使用 Quijote truth：

\[
r_{\mathrm{Q}}(\theta,k).
\]

但如果每轮需要大量设计 \(D_s\)、大量 reference 点和大量 truth 调用，成本会很高。

### 10.2 Quijote power-spectrum generator

当前 z2 已经有 Quijote 功率谱生成器。若其调用成本可控，可以先用它构造 \(Y_{s,j}\)。这和当前 fixed GP 测试更一致。

### 10.3 CSST fastmock

CSST fastmock 可以作为便宜 teacher，用于先估计几何条件化 bias 场：

\[
Y_{\mathrm{CSST}}(\theta,D_s).
\]

但它必须使用和 z2 一致的 residual-anchor/logdiff：

\[
r_{\mathrm{CSST}}(\theta,k)
=
\log P_{\mathrm{CSST,nl}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

CSST bias 不能直接当最终结论，只能作为 PPR 势能源或先验场。最终仍要用 Quijote \(p68\) 验证。

## 11. 从标准几何 bias 场到 PPR 设计

得到

\[
\widehat I_{N,\mathcal G_0}(\theta_j)
\]

后，构造目标密度：

\[
\rho(\theta_j)
\propto
\left(
\widehat I_{N,\mathcal G_0}(\theta_j)+\epsilon
\right)^\alpha.
\]

归一化：

\[
\rho(\theta_j)
=
\frac{
\left(
\widehat I_{N,\mathcal G_0}(\theta_j)+\epsilon
\right)^\alpha
}{
\sum_{\ell=1}^{M}
\left(
\widehat I_{N,\mathcal G_0}(\theta_\ell)+\epsilon
\right)^\alpha
}.
\]

然后不要直接取 top-64。应该做 weighted CVT / repulsive PPR：

\[
\min_D
\sum_{j=1}^{M}
\rho(\theta_j)
\min_{\theta_i\in D}
\|\theta_j-\theta_i\|^2
+
\lambda\mathcal R(D)
+
\mu\mathcal B(D).
\]

这里：

- 第一项让训练点覆盖高 bias density；
- \(\mathcal R(D)\) 防止聚集；
- \(\mathcal B(D)\) 控制边界过度集中；
- 最终设计仍然用 Quijote fixed GP \(p68\) 验证。

## 12. 推荐最小可行实验

### 12.1 Pilot 版本

先做一个较小的 pilot，验证这个想法是否有信号：

1. 固定当前 z2 参数盒子。
2. 固定 \(k\in[0.01,3.0]\)。
3. 生成 reference grid：

\[
M=1024 \text{ 或 } 2048.
\]

4. 生成多组均匀训练集：

\[
S=32,\qquad N=64.
\]

5. 每组训练一个 PCA-GP emulator。
6. 对每组 \(D_s\) 做 Delaunay triangulation。
7. 对 reference 点计算 \(\lambda_{s,j}\)、\(h_{s,j}\)、\(\kappa_{s,j}\)、\(b_j\)。
8. 保留标准几何观测。
9. 对每个 \(\theta_j\) 估计：

\[
\widehat I_{64,\mathcal G_0}(\theta_j).
\]

10. 画出：

- \(\widehat I_{64,\mathcal G_0}(\theta)\) 的二维投影；
- 每个 reference 点的 accepted count \(|A_j|\)；
- \(\widehat I\) 与 raw bias average 的对比；
- \(\widehat I\) 与 nearest-neighbor distance 的相关性。

如果 \(\widehat I\) 仍然和 nearest-neighbor distance 高度相关，说明几何条件没有控制住。

### 12.2 正式 MVP

在 pilot 成立后扩大：

\[
M=4096,\qquad S=128,\qquad N=64.
\]

输出：

```text
data/standard_geometry_bias/<run_id>/
  config_resolved.yaml
  reference_theta.npz
  training_designs.npz
  geometry_features.npz
  accepted_observations.npz
  intrinsic_bias_field.npz
  density_field.npz
  sg_ppr64_design.npz
  diagnostics.json
  plots/
```

最终比较：

\[
J(D_{\mathrm{Sobol64}}),
\quad
J(D_{\mathrm{oldPPR64}}),
\quad
J(D_{\mathrm{EED-PPR64}}),
\quad
J(D_{\mathrm{SG-PPR64}}).
\]

其中 \(D_{\mathrm{SG-PPR64}}\) 表示 standard-geometry PPR 得到的 64 点设计。

## 13. 关键诊断指标

### 13.1 几何残留相关性

检查估计出的 bias field 是否仍然与几何量强相关：

\[
\operatorname{corr}
\left(
\widehat I_{64,\mathcal G_0}(\theta_j),
d_{\mathrm{nn}}(\theta_j,D_s)
\right).
\]

如果相关性仍然很高，说明 \(\mathcal G_0\) 控制不够。

### 13.2 accepted count

每个 reference 点必须记录：

\[
n_j=|A_j|.
\]

如果某些区域 \(n_j\) 太低，该区域的 \(\widehat I\) 不可靠，需要：

1. 增加 \(S\)；
2. 放宽几何条件；
3. 用插值补齐；
4. 单独标记为低置信区域。

### 13.3 尺度分层稳定性

按 simplex scale 分层：

\[
h\in \text{low/mid/high bins}.
\]

分别估计：

\[
\widehat I^{(h\text{-bin})}(\theta).
\]

如果不同 scale bin 的空间结构差异很大，说明 bias field 仍然没有摆脱几何尺度。

### 13.4 与最终 Quijote \(p68\) 的关系

最终还是要看：

\[
\widehat I_{64,\mathcal G_0}
\rightarrow
D_{\mathrm{SG-PPR64}}
\rightarrow
J(D_{\mathrm{SG-PPR64}}).
\]

如果标准几何 bias 场看起来合理，但 \(J\) 没有改善，需要判断问题出在：

1. bias field 数据源，例如 CSST 和 Quijote 不一致；
2. 几何条件过于理想化，不能代表真实 PPR64 训练分布；
3. density-to-design 的 weighted CVT/PPR 过程有问题；
4. 当前 emulator 架构的主要误差不由空间点分布决定。

## 14. 当前方案的优点

相对于 raw bias averaging 或 raw ensemble disagreement，这个方案的优势是：

1. 它承认 bias 依赖样本几何，而不是假设可以直接平均掉。
2. 它不强行拟合一个统一几何扣除项 \(g(\phi)\)。
3. 它通过条件化把不同样本设计放在相同相对几何位置下比较。
4. 它可以显式检查 accepted count、scale、shape、boundary 的影响。
5. 它给出了一个可操作、可验证的目标：

\[
I_{64,\mathcal G_0}(\theta)
=
\text{64 点均匀设计下标准几何位置处的平均困难度。}
\]

这个目标比“绝对 bias”更保守，也更工程化。

## 15. 仍未解决的问题

### 15.1 标准几何是否过于理想化

真实 PPR64 设计不是完全均匀随机设计，它会根据 \(\rho(\theta)\) 聚集到困难区域。因此用“均匀训练集 + 标准 simplex 中心”估计出的 \(I_{64,\mathcal G_0}\)，可能不能完全代表最终 SG-PPR64 训练时的几何。

一个可能的后续方案是迭代：

\[
\rho^{(0)}
\rightarrow
D^{(1)}
\rightarrow
I^{(1)}
\rightarrow
\rho^{(1)}
\rightarrow
D^{(2)}.
\]

但第一版不要直接做迭代，先验证静态场是否有信号。

### 15.2 simplex center 是否是最佳相对位置

simplex center 避开了训练节点，但它也只代表一种相对几何位置。也许更合理的是采样多个标准 barycentric shells，例如：

\[
\lambda \in \mathcal S_1,\mathcal S_2,\mathcal S_3.
\]

这样可以估计 bias 随相对位置变化的曲线，再判断中心位置是否足够代表整体。

### 15.3 边界区域如何处理

边界既是几何异常区，也可能是真实困难区。简单过滤边界会让 PPR 忽略重要区域；不过把边界混入主场又会污染估计。

可能需要边界单独建模：

\[
I_{\mathrm{interior}}(\theta),
\qquad
I_{\mathrm{boundary}}(\theta).
\]

### 15.4 高维 Delaunay 的稳定性

当前是 5D，Delaunay 仍然可做，但当 \(S\)、\(N\)、\(M\) 较大时成本和数值稳定性需要测试。可能要考虑：

1. 使用 scipy Delaunay；
2. 预先在 unit space 工作；
3. 对畸形 simplex 做过滤；
4. 对失败点用 nearest-simplex 或 kNN fallback。

### 15.5 bias 数据源的可信度

如果使用 CSST fastmock，它提供的是 proxy：

\[
I_{\mathrm{CSST},64,\mathcal G_0}(\theta),
\]

而最终目标是 Quijote：

\[
I_{\mathrm{Q},64,\mathcal G_0}(\theta).
\]

二者未必完全一致。因此需要小规模 Quijote 校准或最终 fixed GP 验证。

## 16. 暂定结论

目前更合理的方向不是：

\[
\text{raw ensemble bias}
\rightarrow
\text{average}
\rightarrow
\text{density}
\]

也不是：

\[
\text{raw bias}
\rightarrow
\text{fit geometry term}
\rightarrow
\text{subtract}
\rightarrow
\text{density}.
\]

更稳的方向是：

\[
\boxed{
\text{大量均匀训练集}
\rightarrow
\text{Delaunay simplex}
\rightarrow
\text{固定 reference grid}
\rightarrow
\text{筛选标准相对几何位置}
\rightarrow
\text{robust aggregation}
\rightarrow
I_{64,\mathcal G_0}(\theta)
\rightarrow
\rho(\theta)
\rightarrow
\text{weighted CVT/PPR}
\rightarrow
\text{Quijote }p68\text{ 终评}
}
\]

这个方案的核心不是“扣除几何影响”，而是“在相同几何条件下比较 bias”。它把原先不可控的样本几何波包，转化为一个可定义、可筛选、可诊断的条件化统计量。

## 17. 推荐下一步

下一步不建议直接实现完整 SG-PPR64，而是先做一个诊断型 pilot：

1. \(S=32\)、\(N=64\)、\(M=1024\)。
2. 使用当前 z2 参数盒子和 \(k\in[0.01,3.0]\)。
3. 每个 \(D_s\) 训练一个 fixed PCA-GP。
4. 在 reference grid 上计算 raw bias 和标准几何条件。
5. 得到 \(\widehat I_{64,\mathcal G_0}(\theta)\)。
6. 检查它是否明显弱化了 nearest-neighbor distance、simplex scale 等几何相关性。
7. 如果有效，再用它构造 \(\rho(\theta)\) 并生成 SG-PPR64。

只要 pilot 能证明标准几何条件化后的 bias field 比 raw bias average 更少受样本几何支配，这条路线就值得继续推进。
