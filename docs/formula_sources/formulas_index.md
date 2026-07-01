# Rendered LaTeX formulas

## residual

Source: `tex/residual.tex`

Rendered PNG: `png/residual.png`

```latex
\[
r_Q(\theta,k)=\log P_Q^{\mathrm{nl}}(\theta,k)-\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k)
\]
```

## reconstruction

Source: `tex/reconstruction.tex`

Rendered PNG: `png/reconstruction.png`

```latex
\[
\log \widehat P_Q^{\mathrm{nl}}(\theta,k)=\widehat r_Q(\theta,k)+\log P_{\mathrm{anchor}}^{\mathrm{CDM,nl}}(\theta,k)
\]
```

## p68k

Source: `tex/p68k.tex`

Rendered PNG: `png/p68k.png`

```latex
\[
p68(k_m)=Q_{0.68}\!\left(e_{j,m}\mid j=1,\ldots,256\right)
\]
```

## p68all

Source: `tex/p68all.tex`

Rendered PNG: `png/p68all.png`

```latex
\[
p68_{\mathrm{overall}}=Q_{0.68}\!\left(e_{j,m}\mid j=1,\ldots,256,\;m=1,\ldots,475\right)
\]
```

## biasproxy

Source: `tex/biasproxy.tex`

Rendered PNG: `png/biasproxy.png`

```latex
\[
Y_F(x,D)=Q_{0.68,k}\!\left(\left|\exp\!\left[\widehat r_{F,D}(x,k)-r_F(x,k)\right]-1\right|\right)
\]
```

## sgfield

Source: `tex/sgfield.tex`

Rendered PNG: `png/sgfield.png`

```latex
\[
B_{\mathrm{SG},F}(x)=\mathbb{E}_{D\sim\Pi_N}\!\left[Y_F(x,D)\mid A_{\mathrm{SG}}(x,D)=1\right]
\]
```

## sgset

Source: `tex/sgset.tex`

Rendered PNG: `png/sgset.png`

```latex
\[
\mathcal A_x=\left\{s:\,A_{\mathrm{SG}}(x,D_s)=1\right\},\qquad c_x=\left|\mathcal A_x\right|
\]
```

## sgest

Source: `tex/sgest.tex`

Rendered PNG: `png/sgest.png`

```latex
\[
\widehat B_{\mathrm{SG},F}(x)=\frac{1}{c_x}\sum_{s\in\mathcal A_x}Y_F(x,D_s)
\]
```

## potential

Source: `tex/potential.tex`

Rendered PNG: `png/potential.png`

```latex
\[
V_B(x)=-g_B\,T\!\left(\widehat B(x)\right)
\]
```

## repulsion

Source: `tex/repulsion.tex`

Rendered PNG: `png/repulsion.png`

```latex
\[
F_i^{\mathrm{rep}}=\eta_r\sum_{j\ne i}\frac{\exp\!\left(-d_{ij}^2/2\ell_r^2\right)}{d_{ij}^2+\delta^2}\left(x_i-x_j\right)
\]
```

## score

Source: `tex/score.tex`

Rendered PNG: `png/score.png`

```latex
\[
S_t(x)=\left[\left(\frac{U_t(x)}{s_{U,t}}\right)^2+\left(\lambda\frac{B_t(x)}{s_{B,t}}\right)^2\right]^{1/2}
\]
```
