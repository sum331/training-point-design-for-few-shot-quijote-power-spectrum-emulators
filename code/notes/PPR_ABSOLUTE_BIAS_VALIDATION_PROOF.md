# PPR Standard-Geometry Absolute Bias Field: Theory Proof

本文档只给出理论证明，不包含实验验证脚本或实验流程。这里要证明的是：在 PPR 早期分布阶段，我们构造的“绝对 bias 场”在数学上是一个可识别、可估计、无偏的条件误差场；在物理上，它是对 residual power spectrum 学习困难度的合理抽象，而不是对 Quijote 真实误差的无条件宣称。

一句话概括：

\[
\boxed{
\text{我们估计的不是无条件 } I(\theta)，
\text{而是标准几何条件下的平均学习困难度 }
B_{\mathrm{SG},F}(\theta).
}
}
\]

其中 \(F\) 表示 fastmock source，当前工程中为 CSST residual teacher。

## 1. 物理与工程对象

z2 当前学习对象是 Quijote CDM 非线性功率谱相对 CAMB/CDM 非线性 anchor 的 logdiff residual：

\[
r_{\mathrm Q}(\theta,k)
=
\log P_{\mathrm{Q,nl}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

这里

\[
\theta=(\Omega_m,\Omega_b,h,n_s,A)\in\Theta\subset\mathbb R^5.
\]

当前盒子为

\[
\Omega_m\in[0.24,0.40],\quad
\Omega_b\in[0.04,0.06],\quad
h\in[0.60,0.80],
\]

\[
n_s\in[0.92,1.00],\quad
A\in[1.7,2.5].
\]

最终评价指标是 Quijote 验证集上的

\[
J(D)
=
Q_{0.68,\theta,k}
\left(
\left|
\exp\left[
\hat r_D(\theta,k)-r_{\mathrm Q}(\theta,k)
\right]-1
\right|
\right),
\]

其中 \(D=\{\theta_i\}_{i=1}^{N}\) 是训练设计，\(\hat r_D\) 是用 \(D\) 训练出的 PCA-GP emulator。

PPR 不是最终评价器。PPR 的任务是生成早期训练点分布 \(D_0\)，让后续 emulator 在 \(J(D)\) 上更低。也就是说，PPR 需要一个势能场来告诉粒子“哪些区域应更密集”，但这个势能场本身不等价于最终 Quijote \(p68\)。

## 2. 原始问题为何不可直接解

对任意训练设计 \(D\)，我们能观测或计算的 bias proxy 可以写成

\[
Y(\theta,D).
\]

直觉上，我们想把它拆成“位置本身的困难度”和“训练点几何造成的影响”：

\[
Y(\theta,D)
=
I(\theta)+G(\theta,D)+\epsilon.
\]

但这个分解在数学上不可识别。

### 命题 1：无条件 intrinsic field 不可识别

若只观测 \(Y(\theta,D)\)，且不对 \(I\) 与 \(G\) 加额外约束，则 \(I(\theta)\) 不能唯一确定。

证明如下。任取一个只依赖 \(\theta\) 的函数 \(q(\theta)\)，定义

\[
I'(\theta)=I(\theta)+q(\theta),
\]

\[
G'(\theta,D)=G(\theta,D)-q(\theta).
\]

则

\[
I'(\theta)+G'(\theta,D)+\epsilon
=
I(\theta)+G(\theta,D)+\epsilon
=
Y(\theta,D).
\]

所以同一个观测 \(Y\) 可以对应无穷多个 \(I'\)。因此，在没有进一步规范化条件时，所谓“完全脱离几何的绝对 \(I(\theta)\)”不是可识别对象。

同理，乘性形式

\[
Y(\theta,D)=I(\theta)G(\theta,D)+\epsilon
\]

也不可唯一识别。任取正函数 \(q(\theta)>0\)，令

\[
I'(\theta)=I(\theta)q(\theta),
\qquad
G'(\theta,D)=\frac{G(\theta,D)}{q(\theta)},
\]

则乘积不变。故不能通过假设加性或乘性结构直接恢复 \(I(\theta)\)。

这一步非常关键：我们的方案不能声称“从 \(Y\) 中扣除了几何项，恢复了真实 \(I\)”。正确路线必须换成定义一个可识别的条件对象。

## 3. 标准几何条件

训练设计 \(D\) 由 \(N=64\) 个点组成。对固定位置 \(\theta\)，考虑它在 \(D\) 诱导的 Delaunay triangulation 中落入的 simplex。设该 simplex 为

\[
T(\theta,D)=\operatorname{conv}(v_0,\ldots,v_d),
\qquad d=5.
\]

于是该 simplex 有 \(d+1=6\) 个顶点。令 \(\lambda(\theta,D)\) 为 \(\theta\) 在该 simplex 中的 barycentric 坐标：

\[
\lambda(\theta,D)
=
(\lambda_0,\ldots,\lambda_d),
\qquad
\sum_{a=0}^{d}\lambda_a=1.
\]

simplex 中心的 barycentric 坐标为

\[
\lambda_\star
=
\left(
\frac{1}{d+1},\ldots,\frac{1}{d+1}
\right)
=
\left(
\frac16,\frac16,\frac16,\frac16,\frac16,\frac16
\right).
\]

定义局部尺度

\[
h(T)
=
\left[
\frac{1}{d+1}
\sum_{a=0}^{d}
\|v_a-c_T\|^2
\right]^{1/2},
\qquad
c_T=\frac{1}{d+1}\sum_{a=0}^{d}v_a.
\]

定义形状条件数

\[
\kappa(T)
=
\operatorname{cond}
\left(
[v_1-v_0,\ldots,v_d-v_0]
\right).
\]

定义 unit space 中的边界距离

\[
b(\theta)
=
\min_m\{\theta_m,1-\theta_m\}.
\]

标准几何接受事件定义为

\[
A_{\mathrm{SG}}(\theta,D)=1
\]

当且仅当

\[
\theta\in\operatorname{Hull}(D),
\]

\[
\|\lambda(\theta,D)-\lambda_\star\|_\infty\le \tau_\lambda,
\]

\[
h_{\min}\le h(T(\theta,D))\le h_{\max},
\]

\[
\kappa(T(\theta,D))\le \kappa_{\max},
\]

\[
b(\theta)\ge b_{\min}.
\]

这个条件的含义是：只比较同一类局部几何位置上的误差。它不是估计几何影响后再扣除，而是通过条件化把几何自由度限制在一个标准区域内。

## 4. 可识别的目标量

令 \(\Pi_N\) 表示外部给定的 \(N\) 点训练设计分布，例如 uniform LHS、scrambled Sobol 或其混合。令

\[
D_1,\ldots,D_S\sim \Pi_N
\]

表示多组独立或随机化低差异训练设计。

对 fastmock source \(F\)，定义真实 residual 函数

\[
r_F(\theta,k).
\]

用训练设计 \(D\) 训练 emulator，得到

\[
\hat r_{F,D}(\theta,k).
\]

定义单点 spectral bias 为

\[
Y_F(\theta,D)
=
Q_{0.68,k}
\left(
\left|
\exp[
\hat r_{F,D}(\theta,k)-r_F(\theta,k)
]-1
\right|
\right).
\]

于是我们定义标准几何条件 bias 场：

\[
\boxed{
B_{\mathrm{SG},F}(\theta)
=
\mathbb E_{D\sim\Pi_N}
\left[
Y_F(\theta,D)
\mid
A_{\mathrm{SG}}(\theta,D)=1
\right].
}
\]

这就是本文证明的核心对象。

它有三个性质：

1. 它是可识别的，因为右侧完全由 \(\Pi_N\)、\(Y_F\) 和 \(A_{\mathrm{SG}}\) 定义。
2. 它是几何规范化后的量，因为只在 \(A_{\mathrm{SG}}=1\) 的局部几何状态下比较。
3. 它与 PPR 任务对齐，因为 PPR 关心的是哪些区域在通常 \(N=64\) 采样几何下更难被 emulator 学好。

所以更准确的名称不是“无条件绝对 bias”，而是：

\[
\boxed{
\text{标准几何条件下的平均 fastmock 学习困难度。}
}
\]

## 5. 无偏估计证明

固定一个 reference 点 \(\theta_j\)。为简化记号，令

\[
A_s=A_{\mathrm{SG}}(\theta_j,D_s),
\]

\[
Y_s=Y_F(\theta_j,D_s).
\]

假设

\[
p_j=\mathbb P(A_s=1)>0.
\]

目标量为

\[
B_j
=
B_{\mathrm{SG},F}(\theta_j)
=
\mathbb E[Y_s\mid A_s=1].
\]

实际估计量为

\[
\widehat B_j
=
\frac{\sum_{s=1}^{S}A_sY_s}{\sum_{s=1}^{S}A_s},
\qquad
n_j=\sum_{s=1}^{S}A_s.
\]

当 \(n_j=0\) 时，该点没有标准几何观测，不能直接估计，应标记为无观测或低置信区域。下面讨论 \(n_j>0\)。

### 命题 2：条件于 \(n_j>0\) 时，\(\widehat B_j\) 是无偏估计

对任意 \(n\ge 1\)，条件在 \(n_j=n\) 下，被接受的 \(n\) 个样本来自条件分布

\[
(Y_s\mid A_s=1).
\]

因此

\[
\mathbb E[\widehat B_j\mid n_j=n]
=
\mathbb E
\left[
\frac{1}{n}
\sum_{\ell=1}^{n}
Y_{\ell}^{\mathrm{acc}}
\right],
\]

其中 \(Y_{\ell}^{\mathrm{acc}}\) 表示第 \(\ell\) 个被接受样本的 bias。由于这些样本同分布于 \(Y_s\mid A_s=1\)，有

\[
\mathbb E[Y_{\ell}^{\mathrm{acc}}]
=
\mathbb E[Y_s\mid A_s=1]
=
B_j.
\]

所以

\[
\mathbb E[\widehat B_j\mid n_j=n]
=
\frac{1}{n}
\sum_{\ell=1}^{n}
B_j
=
B_j.
\]

进一步对 \(n_j>0\) 的所有可能取值取全期望：

\[
\mathbb E[\widehat B_j\mid n_j>0]
=
\sum_{n=1}^{S}
\mathbb E[\widehat B_j\mid n_j=n]
\mathbb P(n_j=n\mid n_j>0)
=
B_j.
\]

因此，在有至少一个 accepted observation 的点上，\(\widehat B_j\) 对 \(B_j\) 无偏。

## 6. 一致性证明

仍固定 \(\theta_j\)。定义

\[
U_s=A_sY_s,\qquad V_s=A_s.
\]

则

\[
\widehat B_j
=
\frac{\frac1S\sum_{s=1}^{S}U_s}
{\frac1S\sum_{s=1}^{S}V_s}.
\]

若 \(D_s\) 独立同分布于 \(\Pi_N\)，且 \(\mathbb E[|Y_s|]<\infty\)，由大数定律：

\[
\frac1S\sum_{s=1}^{S}U_s
\xrightarrow{a.s.}
\mathbb E[A_sY_s],
\]

\[
\frac1S\sum_{s=1}^{S}V_s
\xrightarrow{a.s.}
\mathbb E[A_s]
=p_j.
\]

因为 \(p_j>0\)，连续映射定理给出

\[
\widehat B_j
\xrightarrow{a.s.}
\frac{\mathbb E[A_sY_s]}{\mathbb E[A_s]}.
\]

又因为

\[
\mathbb E[A_sY_s]
=
\mathbb P(A_s=1)\,
\mathbb E[Y_s\mid A_s=1]
=
p_jB_j,
\]

所以

\[
\widehat B_j
\xrightarrow{a.s.}
B_j.
\]

这说明估计量不仅无偏，而且随设计组数 \(S\) 增加而一致收敛。

## 7. 有限靶区版本的证明

如果严格要求每次都在完全相同的 \(\theta_j\) 上观测，accepted count 可能不足。一个自然扩展是使用小靶区：

\[
\Theta_j(\rho)
=
\{\theta:\|\theta-\theta_j\|\le \rho\}.
\]

令 \(\Theta\) 是从靶区中按某个对称核 \(K_\rho(\theta-\theta_j)\) 抽取的位置。定义靶区条件目标：

\[
B_{\mathrm{SG},F}^{(\rho)}(\theta_j)
=
\mathbb E
\left[
Y_F(\Theta,D)
\mid
\Theta\in\Theta_j(\rho),
A_{\mathrm{SG}}(\Theta,D)=1
\right].
\]

对应估计量是所有落入靶区且满足标准几何条件的观测平均：

\[
\widehat B_j^{(\rho)}
=
\frac{
\sum_{s,m}A_{\mathrm{SG}}(\theta_{s,m},D_s)
\mathbf 1\{\theta_{s,m}\in\Theta_j(\rho)\}
Y_F(\theta_{s,m},D_s)
}{
\sum_{s,m}A_{\mathrm{SG}}(\theta_{s,m},D_s)
\mathbf 1\{\theta_{s,m}\in\Theta_j(\rho)\}
}.
\]

与上面的证明完全相同，只要分母非零，就有

\[
\mathbb E[
\widehat B_j^{(\rho)}
\mid n_j^{(\rho)}>0
]
=
B_{\mathrm{SG},F}^{(\rho)}(\theta_j).
\]

也就是说，小靶区估计对靶区平均目标仍然无偏。

接下来说明它与点值目标的关系。若 \(B_{\mathrm{SG},F}(\theta)\) 在 \(\theta_j\) 附近 Lipschitz 连续，即存在 \(L\)，使得

\[
|B_{\mathrm{SG},F}(\theta)-B_{\mathrm{SG},F}(\theta_j)|
\le
L\|\theta-\theta_j\|,
\]

则

\[
|B_{\mathrm{SG},F}^{(\rho)}(\theta_j)-B_{\mathrm{SG},F}(\theta_j)|
\le
L\rho.
\]

如果核 \(K_\rho\) 关于 \(\theta_j\) 对称，且 \(B_{\mathrm{SG},F}\) 二阶连续，则一阶项抵消，有

\[
B_{\mathrm{SG},F}^{(\rho)}(\theta_j)
=
B_{\mathrm{SG},F}(\theta_j)
O(\rho^2).
\]

因此，小靶区方法不是对点值目标严格无偏，而是对靶区平均目标无偏；当 \(\rho\) 足够小且场足够平滑时，它对点值目标的偏差可控。

## 8. 插值后的场是什么意思

标准几何观测只存在于有限 reference support 上。对无观测或低观测区域，插值得到的不是新的无偏观测，而是一个空间平滑外推：

\[
\widetilde B_{\mathrm{SG},F}(\theta)
=
\mathcal I
\left(
\{\theta_j,\widehat B_j,n_j\}
\right).
\]

这里 \(\mathcal I\) 是 reliability-weighted local interpolation。它的理论角色是把离散估计场转换为 PPR 势能源，而不是创造新的真实观测。

所以必须区分：

\[
\widehat B_j
\quad
\text{是有 accepted observations 的统计估计，}
\]

\[
\widetilde B(\theta)
\quad
\text{是用于 PPR 松弛的连续势能源。}
\]

前者有无偏性证明，后者的正确性依赖平滑性、support 覆盖和插值稳定性。

## 9. 物理合理性证明

数学证明解决的是“这个统计量是否定义清楚、是否能无偏估计”。物理证明要说明：为什么这个统计量和我们真正关心的 Quijote emulator 精度有关。

### 9.1 residual/logdiff 是正确学习对象

非线性功率谱可以粗略看作

\[
\log P_{\mathrm{nl}}(\theta,k)
=
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k)
+r(\theta,k).
\]

anchor 捕获主要平滑宇宙学趋势，残差 \(r\) 捕获模拟真值与 anchor 之间的剩余结构。学习 \(r\) 比直接学习 \(\log P_{\mathrm{nl}}\) 更接近 emulator 的真实负担。

因此，bias 场中的困难度应理解为：

\[
\text{在当前 anchor 已解释主趋势后，残差函数 } r(\theta,k)
\text{ 对 PCA-GP 的剩余学习难度。}
\]

### 9.2 误差同时由物理曲率和采样几何决定

对局部光滑函数 \(r(\theta,k)\)，插值误差通常由两部分共同控制：

1. 函数本身的局部复杂度，例如
   \[
   \|\nabla_\theta^2 r(\theta,k)\|,
   \quad
   \|\nabla_\theta^3 r(\theta,k)\|.
   \]
2. 训练点在 \(\theta\) 附近的几何结构，例如局部空洞大小、simplex 形状、边界截断和训练点距离。

旧 posterior variance 势能源高度依赖第二项，因此会把“训练点附近 GP 自信”误读成“真实学习难度低”。标准几何方案的物理作用是固定第二项的主导自由度，使剩余平均误差更接近第一项，即 residual 函数本身的学习困难度。

### 9.3 simplex 中心是局部空洞的代表点

在 Delaunay simplex 中，顶点是训练点。靠近顶点时，GP 误差会因为插值条件被压低。simplex 中心附近远离所有顶点，是局部空洞最自然的代表位置。

因此标准几何条件选择

\[
\lambda(\theta,D)\approx \lambda_\star
\]

的物理含义是：测量一个训练设计在局部空洞中心处的学习失败程度。PPR 的任务正是通过调整点密度来减少这些局部空洞带来的误差，所以该对象与 PPR 势能构造天然对齐。

### 9.4 为什么 CSST 可以作为 fastmock proxy

当前 fastmock teacher 是 CSST residual：

\[
r_{\mathrm{CSST}}(\theta,k)
=
\log P_{\mathrm{CSST,nl}}(\theta,k)
-
\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k).
\]

通过固定

\[
w=-1,\qquad w_a=0,\qquad m_\nu=0,
\]

CSST 的 8D 参数空间被限制到与 Quijote 当前任务相容的 5D CDM-like 子空间。两者共享相同的主要宇宙学参数方向、相同的 residual-anchor 思想、相近的 \(k\)-range 和同类 nonlinear matter power 物理。

因此，如果某些区域在 CSST residual 上表现出高学习困难度，合理推断是：这些区域可能具有较高的非线性响应曲率、参数耦合或谱形变化复杂度。这些因素同样会影响 Quijote residual emulator。

需要强调的是：

\[
B_{\mathrm{SG},\mathrm{CSST}}(\theta)
\neq
B_{\mathrm{SG},\mathrm Q}(\theta)
\]

作为数学恒等式并不成立。我们只能说，在物理机制相近的前提下，

\[
B_{\mathrm{SG},\mathrm{CSST}}(\theta)
\]

是

\[
B_{\mathrm{SG},\mathrm Q}(\theta)
\]

的学习困难度 proxy，而不是 Quijote 真值本身。

## 10. 从 bias 场到 PPR 势能

PPR 不直接取 top-bias 点，而是把 bias 场转成目标密度：

\[
\rho(\theta)
\propto
\left(
\widetilde B_{\mathrm{SG},F}(\theta)+\epsilon
\right)^\alpha.
\]

然后通过势能吸引、粒子排斥和边界约束生成设计：

\[
D_{\mathrm{PPR}}
=
\operatorname{Relax}
\left[
\rho(\theta),
\mathcal R(D),
\mathcal B(D)
\right].
\]

这里 \(\rho\) 的作用是提高高学习困难区域的采样密度；\(\mathcal R(D)\) 防止粒子塌缩；\(\mathcal B(D)\) 控制边界聚集。

这一步的理论含义是：

\[
\text{用条件 bias 场定义采样密度，而不是用单个 GP posterior variance 定义采样密度。}
\]

它解决的是旧 PPR 的核心错位：旧方法优化的是模型自认为的不确定性，新方法优化的是在标准几何下实际可观测的学习误差 proxy。

## 11. 最终可证明与不可证明的边界

可以严格证明的内容：

1. 无条件 \(I(\theta)\) 在原问题下不可识别。
2. 标准几何条件目标 \(B_{\mathrm{SG},F}(\theta)\) 是可识别对象。
3. 在随机设计族 \(\Pi_N\) 下，accepted-sample mean 对 \(B_{\mathrm{SG},F}(\theta)\) 无偏。
4. 随 \(S\to\infty\)，该估计量一致收敛到 \(B_{\mathrm{SG},F}(\theta)\)。
5. 小靶区估计对靶区平均目标无偏，并在平滑条件下以 \(O(\rho)\) 或 \(O(\rho^2)\) 接近点值目标。

不能仅靠数学证明的内容：

1. \(B_{\mathrm{SG},\mathrm{CSST}}(\theta)\) 与 \(B_{\mathrm{SG},\mathrm Q}(\theta)\) 完全相同。
2. 插值后的 \(\widetilde B(\theta)\) 在低 support 区域仍是无偏观测。
3. 用该场生成的 PPR 设计必然优于 Sobol。
4. 当前参数 \(\tau_\lambda,h_{\min},h_{\max},\kappa_{\max},b_{\min}\) 是唯一最优选择。

这些属于物理迁移假设和工程性能问题，而不是无偏估计定理本身。

## 12. 最终表述

建议以后用下面这段作为严格表述：

> 我们并不假设 emulator bias 可以被唯一分解为 intrinsic term 与 geometry term，因为该分解在数学上不可识别。我们改为定义一个可识别的条件对象：在 \(N=64\) 外部均匀训练设计、Delaunay simplex 中心附近、尺度正常、形状正常、边界可控的标准几何条件下，位置 \(\theta\) 的平均 fastmock residual GP 误差。对该条件对象，accepted-sample mean 是无偏且一致的估计。物理上，该对象测量的是采样几何被规范化后 residual 功率谱本身对 PCA-GP 的学习困难度。CSST 版本是 Quijote 早期采样密度的 fastmock proxy，而不是最终 Quijote 真值。

因此当前方法的理论结论是：

\[
\boxed{
\widehat B_{\mathrm{SG},\mathrm{CSST}}(\theta)
\text{ 是 }
B_{\mathrm{SG},\mathrm{CSST}}(\theta)
\text{ 的无偏一致估计。}
}
\]

物理结论是：

\[
\boxed{
B_{\mathrm{SG},\mathrm{CSST}}(\theta)
\text{ 是构造 Quijote residual PPR 初始采样密度的合理 fastmock proxy。}
}
\]
