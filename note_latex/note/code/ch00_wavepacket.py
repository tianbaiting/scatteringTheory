#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch00_wavepacket.py
==================

第 0 章 数值实验: 一维自由 Gauss 波包在 H_0 = p^2/(2m) 下的演化.

自然单位: hbar = m = 1.

算法: 在动量空间使用解析传播子
    tilde_psi(p, t) = tilde_psi(p, 0) * exp(-i p^2 t / 2),
然后通过逆 FFT 回到坐标空间.

输出:
    figures/ch00_wavepacket.pdf       —— |psi(x,t)|^2 在 t=0,5,10,15 的快照
    figures/ch00_uncertainty.pdf      —— Delta x, Delta p, Delta x * Delta p 随时间的变化
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False  # 仅使用 mathtext, 避免外部 LaTeX 依赖
import matplotlib.pyplot as plt

# ---------- 数值参数 ----------
N  = 2**13                     # 网格点数 (足够稠密以分辨快振荡)
L  = 200.0                     # 坐标域 [-L/2, L/2]
dx = L / N
x  = np.linspace(-L/2, L/2 - dx, N)

# 与 numpy.fft 约定匹配的动量网格
p  = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)   # 单位: hbar=1 下 k=p
dp = p[1] - p[0]                              # 注: fftfreq 已排序为 [0,...,+,-,...]

# ---------- 初始波包 ----------
x0    = -10.0
p0    =   2.0
sigma =   1.0

def psi0(x):
    """初始 Gauss 波包 psi(x,0)."""
    norm = (np.pi * sigma**2) ** (-0.25)
    return norm * np.exp(-(x - x0)**2 / (2.0 * sigma**2) + 1j * p0 * x)

psi_x0 = psi0(x)

# ---------- 动量空间传播子 ----------
def evolve(psi_x, t):
    """用动量空间相位演化 psi 至时间 t."""
    psi_p = np.fft.fft(psi_x) * dx                  # 离散 -> 连续 Fourier 近似
    psi_p_t = psi_p * np.exp(-1j * p**2 * t / 2.0)  # H_0 = p^2/2
    psi_x_t = np.fft.ifft(psi_p_t) / dx
    return psi_x_t

# ---------- 期望值 / 方差 (在坐标与动量空间) ----------
def moments(psi_x):
    rho_x = np.abs(psi_x)**2
    norm  = np.sum(rho_x) * dx
    rho_x /= norm
    mean_x = np.sum(x * rho_x) * dx
    var_x  = np.sum((x - mean_x)**2 * rho_x) * dx

    psi_p = np.fft.fft(psi_x) * dx
    rho_p = np.abs(psi_p)**2 / (2.0 * np.pi)        # 连续动量分布密度
    norm_p = np.sum(rho_p) * dp
    rho_p /= norm_p
    mean_p = np.sum(p * rho_p) * dp
    var_p  = np.sum((p - mean_p)**2 * rho_p) * dp
    return mean_x, np.sqrt(var_x), mean_p, np.sqrt(var_p)

# ---------- 演化并作图 ----------
times = [0.0, 5.0, 10.0, 15.0]
colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

fig, ax = plt.subplots(figsize=(7.0, 4.2))
records = []
for t, c in zip(times, colors):
    psi_t = evolve(psi_x0, t)
    rho   = np.abs(psi_t)**2
    ax.plot(x, rho, color=c, lw=1.6, label=fr"$t={t:g}$")
    mx, dxs, mp, dps = moments(psi_t)
    records.append((t, mx, dxs, mp, dps))
    print(f"t = {t:5.2f}  <x> = {mx:+8.4f}  Dx = {dxs:.4f}"
          f"  <p> = {mp:+.4f}  Dp = {dps:.4f}  DxDp = {dxs*dps:.4f}")

ax.set_xlim(-15, 30)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.set_title(r"Free Gaussian wave-packet evolution  ($x_0=-10,\ p_0=2,\ \sigma=1$)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.3)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ch00_wavepacket.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- Delta x * Delta p 随时间 ----------
ts_fine = np.linspace(0.0, 20.0, 41)
prods, dxs_arr, dps_arr = [], [], []
for t in ts_fine:
    _, dxs, _, dps = moments(evolve(psi_x0, t))
    dxs_arr.append(dxs); dps_arr.append(dps); prods.append(dxs*dps)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(ts_fine, dxs_arr, label=r"$\Delta x$", lw=1.6)
ax.plot(ts_fine, dps_arr, label=r"$\Delta p$", lw=1.6)
ax.plot(ts_fine, prods,   label=r"$\Delta x\,\Delta p$", lw=1.8, color="k")
ax.axhline(0.5, ls="--", color="gray", lw=1.0, label=r"Heisenberg bound $1/2$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"uncertainty")
ax.set_title(r"Numerical check of $\Delta x\,\Delta p \geq 1/2$")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
fig.savefig("figures/ch00_uncertainty.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- 范数守恒检查 ----------
norms = [np.sum(np.abs(evolve(psi_x0, t))**2) * dx for t in times]
print("Norm at each t:", ["%.6f" % n for n in norms])
print("Done. Figures saved to figures/.")
