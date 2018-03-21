## Machine Learning: Week 2

2018-03-14

kernel alignment

$max_A  tr(AB)$

$A = M^T P M$

$\sum \limits_{t-h \leq l \leq t}J_l^2= IV_{t,h} + JV_{t,h}$



哲学：Occam's razor

计算机： MDL (minimum description length)

数学：regularization/正则化



$||u||_0=Card(u)$

## 矩阵

references: _Matrix Cookbook_

书: (A.25) — (A.29)

### SVD

Application: 

- PCA

$C=cov(XX^T)$ eigenvalue decompostion  time comlexity $O(d^3)$

$CP=P\Lambda \rightarrow XX^TP=P\Lambda $ 

$U\Sigma V^TV\Sigma U^TP=U\Sigma ^2U^TP= P\Lambda$

可以将特征值分解转换成SVD分解 — $O(d^2n)$ when $d<n$ or $O(n^2d)$ when $n<d$.