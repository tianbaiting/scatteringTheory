#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch08_born_convergence.py
========================

第 8 章数值实验: 部分波 Lippmann-Schwinger 方程的 Born 级数收敛性.

迭代 T_l = V_l + V_l G_0 T_l (取 l = 0), 比较截断 Born 级数
    T^(N) = sum_{n=0}^N V (G_0 V)^n
与精确解
    T_exact = (1 - V G_0)^(-1) V
在动量空间网格上的 on-shell 元素 T(p0, p0).

物理设置 (hbar = m = 1):
    Yukawa 势        V(r) = - gamma * exp(- mu r) / r
    三维 Fourier     tildeV(q) = - 4 pi gamma / (q^2 + mu^2)
    动量基矩阵元      <p'|V|p> = tildeV(|p'-p|) / (2 pi)^3
    s 波投影
        V_0(p', p) = - gamma / (4 pi^2 p' p)
                     * ln[((p'+p)^2 + mu^2) / ((p'-p)^2 + mu^2)]
    自由 Green 函数  G_0(p; E) = 1 / (E - p^2/2 + i epsilon),  E = p0^2 / 2
    on-shell 极点用小虚部 +i epsilon 正则化 (与 ch07 约定一致).

数值方法: 用 Gauss-Legendre 节点离散化 [0, p_max], 把
    T(p_i, p_j) = V(p_i, p_j) + sum_k w_k p_k^2 V(p_i,p_k) G_0(p_k) T(p_k,p_j)
化为有限维线性方程; on-shell 元素取距 p0 最近的网格点.

输出: figures/ch08_born_convergence.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

N_GRID = 200
P_MAX  = 10.0
MU     = 1.0
EPS    = 1e-3   # +i epsilon 正则化 (有限-N 等价于主值 + 留数)


def make_grid(n=N_GRID, p_max=P_MAX):
    """[0, p_max] 上 n 个 Gauss-Legendre 节点 / 权重."""
    x, w = np.polynomial.legendre.leggauss(n)
    p = 0.5 * (x + 1.0) * p_max
    w = 0.5 * w * p_max
    return p, w


def yukawa_swave(pp, p, gamma, mu=MU):
    """Yukawa s 波动量空间矩阵元 V_0(p', p)."""
    safe = pp * p + 1e-14
    num  = (pp + p) ** 2 + mu ** 2
    den  = (pp - p) ** 2 + mu ** 2
    return -gamma / (4.0 * np.pi ** 2 * safe) * np.log(num / den)


def build_V(p, gamma):
    P1, P2 = np.meshgrid(p, p, indexing="ij")
    return yukawa_swave(P1, P2, gamma)


def build_G0_diag(p, w, p0):
    """K_j = w_j * p_j^2 * G_0(p_j; E = p0^2 / 2 + i eps), 离散化对角."""
    E = 0.5 * p0 ** 2
    return w * p ** 2 / (E - 0.5 * p ** 2 + 1j * EPS)


def born_series_onshell(V, K, p, p0, N_max):
    """累积 Born 级数与精确解, 取 j0 ~ p0 的 on-shell 元素."""
    j0  = int(np.argmin(np.abs(p - p0)))
    VG  = V * K[np.newaxis, :]                   # (V G_0) 矩阵
    cur = V.copy()                                # (V G_0)^k V, 初值 k=0
    series = [cur[j0, j0]]
    for _ in range(1, N_max + 1):
        cur = VG @ cur
        series.append(series[-1] + cur[j0, j0])
    T_full = np.linalg.solve(np.eye(V.shape[0]) - VG, V)
    return np.asarray(series), T_full[j0, j0]


def spectral_radius(V, K):
    """rho(V G_0) — Born 级数全程收敛要求 < 1."""
    return np.max(np.abs(np.linalg.eigvals(V * K[np.newaxis, :])))


def run_case(gamma, p0=1.0, N_max=40):
    p, w = make_grid()
    V    = build_V(p, gamma)
    K    = build_G0_diag(p, w, p0)
    s, T = born_series_onshell(V, K, p, p0, N_max)
    return np.arange(N_max + 1), np.abs(s / T - 1.0), T


def run_rho_curve(gamma, p0_arr):
    p, w = make_grid()
    V    = build_V(p, gamma)
    return np.array([spectral_radius(V, build_G0_diag(p, w, p0)) for p0 in p0_arr])


def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.normpath(os.path.join(here, "..", "figures"))
    os.makedirs(fig_dir, exist_ok=True)

    gamma_weak, gamma_strong = 0.5, 15.0
    p0_fixed, N_max          = 1.0, 40

    n_w, rel_w, Tw = run_case(gamma_weak,   p0=p0_fixed, N_max=N_max)
    n_s, rel_s, Ts = run_case(gamma_strong, p0=p0_fixed, N_max=N_max)

    p0_arr = np.linspace(0.2, 4.0, 24)
    rho_w  = run_rho_curve(gamma_weak,   p0_arr)
    rho_s  = run_rho_curve(gamma_strong, p0_arr)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].semilogy(n_w, rel_w + 1e-16, "o-", color="#1f77b4",
                   label=fr"weak $\gamma={gamma_weak}$")
    ax[0].semilogy(n_s, rel_s + 1e-16, "s-", color="#d62728",
                   label=fr"strong $\gamma={gamma_strong}$")
    ax[0].set_xlabel("Born order $N$")
    ax[0].set_ylabel(r"$|T^{(N)}_0(p_0,p_0)/T_0^{\rm exact} - 1|$")
    ax[0].set_title(fr"Born series convergence at $p_0={p0_fixed}$")
    ax[0].grid(True, which="both", ls=":", alpha=0.5)
    ax[0].legend()

    E_arr = 0.5 * p0_arr ** 2
    ax[1].plot(E_arr, rho_w, "o-", color="#1f77b4",
               label=fr"weak $\gamma={gamma_weak}$")
    ax[1].plot(E_arr, rho_s, "s-", color="#d62728",
               label=fr"strong $\gamma={gamma_strong}$")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\rho=1$ threshold")
    ax[1].set_xlabel(r"energy $E = p_0^2/2$")
    ax[1].set_ylabel(r"spectral radius $\rho(V G_0)$")
    ax[1].set_title(r"Convergence radius criterion")
    ax[1].grid(True, ls=":", alpha=0.5)
    ax[1].legend()

    plt.tight_layout()
    out = os.path.join(fig_dir, "ch08_born_convergence.pdf")
    fig.savefig(out)
    print(f"[ch08] wrote {out}")
    print(f"[ch08] weak   T_exact(p0,p0) = {Tw:.4e}")
    print(f"[ch08] strong T_exact(p0,p0) = {Ts:.4e}")
    i1 = int(np.argmin(np.abs(p0_arr - 1.0)))
    print(f"[ch08] weak   rho(VG_0) at p0~=1 = {rho_w[i1]:.3f}")
    print(f"[ch08] strong rho(VG_0) at p0~=1 = {rho_s[i1]:.3f}")


if __name__ == "__main__":
    main()
