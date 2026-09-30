# 公式源码索引

本文件按[论文正文](article_text.md)中的出现顺序列出公式源码。
代码块无需 GitHub 数学渲染器，适合在公式预览被挤压或显示异常时核对符号；正式排版请以原论文为准。

## 式（1）

```tex
\begin{aligned}
\min\quad & F=\sum_{t=1}^{144}p_tg_t,\\
\mathrm{s.t.}\quad
&g_t+r_t-w_t+d_t=\ell_t+c_t,\\
&S_t=S_{t-1}+0.9c_t-\frac{d_t}{0.9},\\
&1200\le S_t\le10800,\qquad 0\le c_t,d_t\le M,\\
&g_t\ge0,\qquad 0\le w_t\le r_t,\\
&S_0=S_{144}=6000.
\end{aligned}\tag{1}
```

## 未编号公式（第 2 个公式块）

```tex
0\le c_t\le Mz_t,\qquad 0\le d_t\le M(1-z_t),\qquad z_t\in\{0,1\}.
```

## 式（2）

```tex
\min_{\boldsymbol g_k,\mu_k}\quad
 \sum_{t=1}^{T}p_tg_{k,t}+
 \mathbb E\!\left[5\sum_{t=1}^{T}p_tu_{k,t}^{\mu_k}-vS_{k,T}^{\mu_k}
 \,\middle|\,\mathcal F_{k-1}\right].\tag{2}
```

## 式（3）

```tex
\widehat y_{k,t}^{(1)}=
 \left[b_{k,t}^{(1)}+f_k(\boldsymbol x_{k,t})\right]_{+},\qquad
 f_k(\boldsymbol x)=\gamma_{k,0}+\sum_{j=1}^{J_k}\gamma_{k,j}\mathcal T_{k,j}(\boldsymbol x).\tag{3}
```

## 式（4）

```tex
(\widehat a,\widehat{\boldsymbol\beta})=
 \arg\min_{a,\boldsymbol\beta}
 \|\boldsymbol r-a\boldsymbol1-X\boldsymbol\beta\|_2^2+\lambda_{\mathrm R}\|\boldsymbol\beta\|_2^2,
 \qquad
 \widehat y_{k,t}^{(2)}=\left[b_{k,t}^{(2)}+\widehat a+\widetilde{\boldsymbol x}_{k,t}^{\mathsf T}\widehat{\boldsymbol\beta}\right]_{+}.\tag{4}
```

## 式（5）

```tex
\begin{aligned}
 B_{k,t}^{L}&=0.65\,\mathcal G_2(L_{k-7,\cdot})_t+0.35\,\mathcal G_2(L_{k-1,\cdot})_t,\\
 B_{k,t}^{P}&=0.65\,\mathcal G_2(P^{\mathrm{PV}}_{k-1,\cdot})_t+
 0.35\,\mathcal G_2\!\left(\frac13\sum_{j=1}^{3}P^{\mathrm{PV}}_{k-j,\cdot}\right)_t.
 \end{aligned}\tag{5}
```

## 式（6）

```tex
\widehat y_{k,t}^{(3)}=\left[B_{k,t}+f_k^{\mathrm{res}}(\boldsymbol z_{k,t})\right]_{+}.\tag{6}
```

## 式（7）

```tex
\min_{A,\boldsymbol b}\ \sum_{j\lt k}\omega_{k,j}
 \|\boldsymbol y_j-A^{\mathsf T}\boldsymbol c_j-\boldsymbol b\|_2^2
 +\lambda_{\mathrm C}\|A\|_{\mathrm F}^2,
 \qquad \omega_{k,j}=2^{-(k-1-j)/84}.\tag{7}
```

## 式（8）

```tex
\widehat y_{k,t}=\sum_{m=1}^{4}\alpha_{k,m}\widehat y_{k,t}^{(m)},
 \qquad \alpha_{k,m}\ge0,\quad\sum_{m=1}^{4}\alpha_{k,m}=1.\tag{8}
```

## 式（9）

```tex
\widehat{\boldsymbol\alpha}_k=
 \arg\min_{\boldsymbol\alpha\ge0,\,\boldsymbol1^{\mathsf T}\boldsymbol\alpha=1}
 \frac{\|X_k\boldsymbol\alpha-\boldsymbol y_k^{H}\|_2^2}{m_k}
 +\lambda_k\left\|\boldsymbol\alpha-\tfrac14\boldsymbol1\right\|_2^2\tag{9}
```

## 式（10）

```tex
\widehat n_{k,t}=(\widehat L_{k,t}-\widehat P^{\mathrm{PV}}_{k,t})\Delta t.\tag{10}
```

## 式（11）

```tex
n_{k,t}^{(i)}=\widehat n_{k,t}+e_{i,t},\qquad
 \pi_{k,i}=\frac{2^{-(k-i)/14}}{\sum_{j\in\mathcal I_k}2^{-(k-j)/14}},\quad i\in\mathcal I_k.\tag{11}
```

## 式（12）

```tex
\begin{aligned}
 &g_t+d_t+u_t=n_t+c_t+w_t,\qquad S_t=S_{t-1}+\eta c_t-d_t/\eta,\\
 &S_{\min}\le S_t\le S_{\max},\qquad 0\le c_t,d_t\le M,\qquad c_td_t=0,\\
 &g_t,u_t,w_t\ge0,\qquad S_{k+1,0}=S_{k,T}.
 \end{aligned}\tag{12}
```

## 式（13）

```tex
\Psi_t(g)=p_tg+5p_t\,\mathbb E[(N_t-g)_+].\tag{13}
```

## 式（14）

```tex
\Psi_t'(g)=p_t-5p_t\Pr(N_t>g),\qquad
 \Psi_t'(g)=0\ \Longrightarrow\ F_{N_t}(g)=1-\frac15=0.8.\tag{14}
```

## 式（15）

```tex
q_t=\inf\left\{x:\sum_{i\in\mathcal I}\pi_i\,\boldsymbol1(n_t^{(i)}\le x)\ge0.8\right\}.\tag{15}
```

## 式（16）

```tex
\begin{aligned}
 \min\quad &Z_0=\sum_{t=1}^{T}p_t(g_t+5u_t)-vS_T
             +\varepsilon\sum_{t=1}^{T}(c_t+d_t),\\
 \mathrm{s.t.}\quad
 &g_t+d_t+u_t=q_t+c_t+w_t,\\
 &S_t=S_{t-1}+\eta c_t-d_t/\eta,\qquad S_0=S^{\mathrm{ini}},\\
 &S_{\min}\le S_t\le S_{\max},\\
 &0\le c_t\le M\delta_t,\qquad 0\le d_t\le M(1-\delta_t),\\
 &g_t,u_t,w_t\ge0,\qquad \delta_t\in\{0,1\},\quad t=1,\ldots,T.
 \end{aligned}\tag{16}
```

## 式（17）

```tex
v=\frac{\min_{1\le t\le36}p_t}{\eta},\qquad \varepsilon=10^{-7}.\tag{17}
```

## 式（18）

```tex
\begin{cases}
 d_t=\min\{h_t,M,\eta[S_{t-1}-R_t]_+\},\quad
 u_t=h_t-d_t,\quad c_t=w_t=0, & h_t>0,\\[3pt]
 c_t=\min\{-h_t,M,(S_{\max}-S_{t-1})/\eta\},\quad
 w_t=-h_t-c_t,\quad d_t=u_t=0, & h_t\le0.
 \end{cases}\tag{18}
```

## 式（19）

```tex
\begin{aligned}
 R_t&\in\arg\min_{r\in\mathcal S}\{V_{t+1}(r)+5p_t\eta r\},\\
 V_t(s)&=\sum_{i\in\mathcal I}\pi_i\left[
 5p_tu_t^{(i)}(s;R_t)+\widetilde V_{t+1}\!\left(S_t^{(i)}(s;R_t)\right)\right].
 \end{aligned}\tag{19}
```

## 式（20）

```tex
\begin{aligned}
 g_t(\boldsymbol z)&=[g_t^{(0)}+z_{b(t)}\sigma_t]_+,
 \qquad b(t)=1+\left\lfloor\frac{t-1}{24}\right\rfloor,\quad -2\le z_b\le2,\\
 \sigma_t&=\max\left\{20,\sqrt{\sum_i\pi_i(n_t^{(i)}-\bar n_t)^2}\right\},
 \qquad \bar n_t=\sum_i\pi_i n_t^{(i)}.
 \end{aligned}\tag{20}
```

## 式（21）

```tex
\min_{\boldsymbol z\in[-2,2]^6}J(\boldsymbol z;\boldsymbol R)=
 \sum_t p_tg_t(\boldsymbol z)+
 \sum_i\pi_i\left[5\sum_t p_tu_t^{(i)}(\boldsymbol z)-vS_T^{(i)}(\boldsymbol z)\right].\tag{21}
```

## 式（22）

```tex
F=\sum_{k\in\mathcal D}\sum_{t=1}^{T}p_t(g_{k,t}+5u_{k,t})
 =\boxed{13\,475\,274.43\ \text{元}}\tag{22}
```

## 式（23）

```tex
\widehat P^{(r)}_{k,t}=(1-\alpha_{k,r})\widehat P^B_{k,t}
                    +\alpha_{k,r}\widetilde P^{(r)}_{k,t},\qquad t>b_r.\tag{23}
```

## 式（24）

```tex
\alpha_{k,r}=\Pi_{[0,1]}\!\left(
 \frac{\displaystyle\sum_{i\in\mathcal H_k}\rho_{k,i}
       \sum_{t>b_r}\delta^{(r)}_{i,t}(P^{\mathrm{PV}}_{i,t}-\widehat P^B_{i,t})}
      {\displaystyle\sum_{i\in\mathcal H_k}\rho_{k,i}
       \sum_{t>b_r}(\delta^{(r)}_{i,t})^2}\right).\tag{24}
```

## 式（25）

```tex
x_{k,r}=\left(\frac{1}{6}\sum_{t=b_r-5}^{b_r}e_{k,t},\quad
                 \frac{1}{b_r}\sum_{t=1}^{b_r}e_{k,t}\right)^{\!\mathsf T}.\tag{25}
```

## 式（26）

```tex
\widehat\beta_{k,r,j}=\arg\min_{\beta\in\mathbb R^3}
 \left\{\sum_{i\in\mathcal H_k}\rho_{k,i}
 (y_{i,j}-\widetilde x_{i,r}^{\mathsf T}\beta)^2+10\|\beta\|_2^2\right\}.\tag{26}
```

## 式（27）

```tex
\widehat L^{(r)}_{k,t}=\left[\widehat L^B_{k,t}
       +\widetilde x_{k,r}^{\mathsf T}\widehat\beta_{k,r,j}\right]_{+},
 \qquad 36j<t\leq36(j+1).\tag{27}
```

## 式（28）

```tex
n^{(i,r)}_{k,t}=\widehat n^{(r)}_{k,t}
       +n_{i,t}-\widehat n^{(r)}_{i,t},\qquad i\in\mathcal J_k,\quad t>b_r.\tag{28}
```

## 式（29）

```tex
\begin{aligned}
 D_{k,i,r}&=\frac13\sum_{\ell=1}^{3}
       \left(\frac{f_{i,r,\ell}-f_{k,r,\ell}}{s_\ell}\right)^2,\\
 q_{k,i,r}&=\frac{\pi_{k,i}\exp[-\tfrac12\min(D_{k,i,r},50)]}
 {\sum_{h\in\mathcal J_k}\pi_{k,h}\exp[-\tfrac12\min(D_{k,h,r},50)]},\\
 \omega_{k,i,r}&=0.25\pi_{k,i}+0.75q_{k,i,r}.
 \end{aligned}\tag{29}
```

## 式（30）

```tex
\begin{aligned}
 \phi_t(a;g^0)&=p_tg^0+1.5p_t\left[a-g^0\right]_{+}-0.5p_t\left[g^0-a\right]_{+},\\
 F_k&=\sum_{t=1}^{T}\left[\phi_t(a_{k,t};g^0_{k,t})+5p_tu_{k,t}\right].
\end{aligned}\tag{30}
```

## 式（31）

```tex
\begin{aligned}
 \min_{a^{(r)},\mu^{(r)}}\ J_{k,r}
 ={}&\sum_{t\in\mathcal T_r}\phi_t(a^{(r)}_{k,t};g^0_{k,t})\\
 &+\sum_{i\in\mathcal J_k}\omega_{k,i,r}
 \left[5\sum_{t\in\mathcal T_r}p_tu^{(i,\mu^{(r)})}_{k,t}
       -vS^{(i,\mu^{(r)})}_{k,T}\right].
 \end{aligned}\tag{31}
```

## 式（32）

```tex
a^{(r)}_{k,t}=\left[\overline a^{(r)}_{k,t}+z_t\sigma_{k,r,t}\right]_{+},
 \qquad -2\leq z_t\leq2,\qquad t\in\mathcal T_r.\tag{32}
```

## 式（33）

```tex
\widehat p^{(1)}_{q|k,t}=p_{i^*,t},\qquad
 \widehat p^{(2)}_{q|k,t}=\sum_{i\in\mathcal W_{k,q}}\nu_{k,i}p_{i,t}.\tag{33}
```

## 式（34）

```tex
\begin{aligned}
 \widehat\gamma_k&=\arg\min_\gamma
 \left\{\sum_{i\in\mathcal H_k}2^{-(k-i)/28}
       (\overline p_i-z_{i;k}^{\mathsf T}\gamma)^2+\gamma^{\mathsf T}\Lambda\gamma\right\},\\
 \widehat p^{(3)}_{q|k,t}&=\max\left\{\varepsilon_p,\ z_{q;k}^{\mathsf T}\widehat\gamma_k
       +\sum_{i\in\mathcal W_{k,q}}\nu_{k,i}(p_{i,t}-\overline p_i)\right\}.
\end{aligned}\tag{34}
```

## 式（35）

```tex
\widehat p^{(r)}_{k,t}=\max\{\varepsilon_p,\widehat p^{(0)}_{k,t}
                    +\widetilde h_{k,r}^{\mathsf T}\widehat\beta^p_{k,r,j}\},
 \qquad 36j<t\leq36(j+1),\quad j\geq r.\tag{35}
```

## 式（36）

```tex
v_k=\frac{Q_{0.25}\bigl(\{\widehat p_{k+1|k,t}:1\leq t\leq36\}\bigr)}{0.9},
 \qquad V_{T+1}(s)=-v_ks.\tag{36}
```

## 式（37）

```tex
\epsilon_{i,t}=p_{i,t}-\widehat p^{(r)}_{i,t},\qquad
 \epsilon^{c}_{i,t}=\epsilon_{i,t}-\sum_{j\in\mathcal J_k}\omega_j\epsilon_{j,t}.\tag{37}
```

## 式（38）

```tex
\begin{aligned}
 \kappa_t&=\min\left\{1,\frac{0.95\widehat p^{(r)}_{k,t}}
 {\max\{\max_i(-\epsilon^{c}_{i,t}),10^{-12}\}}\right\},\\
 p^{(i,r)}_{k,t}&=\widehat p^{(r)}_{k,t}+\kappa_t\epsilon^{c}_{i,t}.
 \end{aligned}\tag{38}
```

## 式（39）

```tex
\min_{g^0,\mu}\ \sum_{i\in\mathcal J_k}\omega_i
 \left[\sum_{t=1}^{T}p^{(i,0)}_{k,t}
              (g^0_{k,t}+5u^{(i,\mu)}_{k,t})-v_kS^{(i,\mu)}_{k,T}\right].\tag{39}
```

## 式（40）

```tex
\begin{aligned}
 \widetilde n^{(i,r)}_{k,t}
   &=\widehat n^{(0)}_{k,t}+\widehat n^{(r)}_{i,t}-\widehat n^{(0)}_{i,t},\\
 \widetilde p^{(i,r)}_{k,t}
   &=\max\{\varepsilon_p,\widehat p^{(0)}_{k,t}
             +\widehat p^{(r)}_{i,t}-\widehat p^{(0)}_{i,t}\},\qquad t>b_r.
 \end{aligned}\tag{40}
```

## 式（41）

```tex
p\bigl[G+1.5(A-G)_+-0.5(G-A)_+\bigr]
      =pA+0.5p|G-A|.\tag{41}
```

## 式（42）

```tex
g^0_{k,t}=\begin{cases}
 g^{\mathrm b}_{k,t},&1\leq t\leq36,\\
 0.5g^{\mathrm b}_{k,t}+0.5\widetilde g_{k,t},&36\lt t\leq T.
 \end{cases}\tag{42}
```
