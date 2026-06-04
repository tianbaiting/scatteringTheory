#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch06_wignerD.py
===============

第 6 章 数值实验: Wigner d^j_{m'm}(theta) 函数, 正交性验证, 螺度振幅 phi 因子.

(a) 计算 j ∈ {1/2, 1, 3/2, 2} 的所有 d^j_{m'm}(theta).
(b) 验证正交关系
        ∫_0^pi d^j_{m'm}(theta) d^j_{m''m}(theta) sin(theta) dtheta
            = 2/(2j+1) * delta_{m'm''}.
(c) 螺度振幅 phi 因子: 验证 <lc ld|S|la lb> 在玩具模型下确实呈
        ~ exp(i Δλ phi),   Δλ = la - lb - lc + ld.

输出:
    figures/ch06_wignerD.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from math import factorial, sqrt
from scipy.integrate import quad


# ---------- Wigner small-d via the explicit sum (real-valued) ----------
def wigner_d(j, mp, m, theta):
    """d^j_{m' m}(theta) -- Wigner small-d, j 可半整数."""
    # j, mp, m 用 2*整数 表示以避免半整数浮点失误
    two_j  = int(round(2 * j))
    two_mp = int(round(2 * mp))
    two_m  = int(round(2 * m))
    # 整数化的阶乘参数
    a = (two_j + two_mp) // 2
    b = (two_j - two_mp) // 2
    c = (two_j + two_m)  // 2
    d = (two_j - two_m)  // 2
    pref = sqrt(factorial(a) * factorial(b) * factorial(c) * factorial(d))
    c2 = np.cos(theta / 2.0)
    s2 = np.sin(theta / 2.0)
    # k 范围: max(0, m-m') <= k <= min(j+m, j-m')
    k_min = max(0, (two_m - two_mp) // 2)
    k_max = min((two_j + two_m) // 2, (two_j - two_mp) // 2)
    out = np.zeros_like(np.asarray(theta, dtype=float))
    for k in range(k_min, k_max + 1):
        denom = (factorial(k) * factorial((two_j + two_m) // 2 - k)
                 * factorial((two_mp - two_m) // 2 + k)
                 * factorial((two_j - two_mp) // 2 - k))
        sign = (-1) ** ((two_mp - two_m) // 2 + k)
        p1 = (2 * two_j + two_m - two_mp - 4 * k) // 2  # cos(θ/2) 幂: 2j+m-m'-2k
        p2 = ((two_mp - two_m) // 2 + 2 * k)             # sin(θ/2) 幂: m'-m+2k
        out = out + sign / denom * c2**p1 * s2**p2
    return pref * out


# ---------- (a) 绘 d^{1/2}, d^1 ----------
theta = np.linspace(0, np.pi, 401)

fig = plt.figure(figsize=(11, 8))
ax1 = fig.add_subplot(2, 2, 1)
for mp, m, ls in [(0.5, 0.5, "-"), (0.5, -0.5, "--"),
                  (-0.5, 0.5, ":"), (-0.5, -0.5, "-.")]:
    ax1.plot(theta, wigner_d(0.5, mp, m, theta), ls,
             lw=1.6, label=fr"$d^{{1/2}}_{{{mp:+.1f},{m:+.1f}}}$")
ax1.set_xlabel(r"$\theta$"); ax1.set_ylabel(r"$d^{1/2}_{m'm}(\theta)$")
ax1.set_title(r"Wigner $d^{1/2}$ 函数 ($\cos(\theta/2),\ \pm\sin(\theta/2)$)")
ax1.axhline(0, color="gray", lw=0.6); ax1.grid(alpha=0.3)
ax1.legend(frameon=False, fontsize=8, ncol=2)

ax2 = fig.add_subplot(2, 2, 2)
mvals1 = [1, 0, -1]
for mp in mvals1:
    for m in mvals1:
        ax2.plot(theta, wigner_d(1, mp, m, theta),
                 lw=1.3, label=fr"$d^{{1}}_{{{mp:+d},{m:+d}}}$")
ax2.set_xlabel(r"$\theta$"); ax2.set_ylabel(r"$d^{1}_{m'm}(\theta)$")
ax2.set_title(r"Wigner $d^{1}$ 函数 (9 个分量)")
ax2.axhline(0, color="gray", lw=0.6); ax2.grid(alpha=0.3)
ax2.legend(frameon=False, fontsize=7, ncol=3, loc="lower center")


# ---------- (b) 正交性验证 ----------
# 标准正交关系 (固定 m', m, 在 j 上正交):
#     ∫_0^pi d^{j1}_{m'm}(θ) d^{j2}_{m'm}(θ) sin θ dθ = 2/(2 j1 + 1) δ_{j1 j2}.
# 我们枚举 (m', m) 公共允许的 j ∈ {1/2, 1, 3/2, 2}, 计算 (j1, j2) 矩阵.
def orthogonality_matrix_j(mp, m, jlist):
    """返回 (j1, j2) inner-product 矩阵, 期望对角为 2/(2j+1) * δ_{j1j2}."""
    n = len(jlist)
    M = np.zeros((n, n))
    for i, j1 in enumerate(jlist):
        for k, j2 in enumerate(jlist):
            if j1 < max(abs(mp), abs(m)) or j2 < max(abs(mp), abs(m)):
                continue
            f = lambda th, j1=j1, j2=j2: (wigner_d(j1, mp, m, th)
                                          * wigner_d(j2, mp, m, th)
                                          * np.sin(th))
            M[i, k], _ = quad(f, 0, np.pi, limit=200)
    expect = np.diag([2.0 / (2 * j + 1) if j >= max(abs(mp), abs(m))
                      else 0.0 for j in jlist])
    return M, expect


# 整数 j 取 (m', m) = (1, 0); 半整数 j 取 (m', m) = (1/2, 1/2)
jlist_int  = [1.0, 2.0, 3.0, 4.0]
jlist_half = [0.5, 1.5, 2.5, 3.5]
M_int, E_int = orthogonality_matrix_j(1, 0, jlist_int)
M_h,   E_h   = orthogonality_matrix_j(0.5, 0.5, jlist_half)
err_int  = np.max(np.abs(M_int - E_int))
err_half = np.max(np.abs(M_h   - E_h))
print(f"[orthogonality int  j (m'=1, m=0)] max |num - exp| = {err_int:.3e}")
print(f"[orthogonality half j (m'=m=1/2)]  max |num - exp| = {err_half:.3e}")

# 把整数 j 矩阵作热图; 在格子里同时标注 数值 / 解析
ax3 = fig.add_subplot(2, 2, 3)
diag_max = max(2.0/(2*j+1) for j in jlist_int)
im = ax3.imshow(M_int, cmap="RdBu_r", vmin=-diag_max, vmax=diag_max)
labels_j = [f"{int(j)}" for j in jlist_int]
ax3.set_xticks(range(len(jlist_int))); ax3.set_xticklabels(labels_j)
ax3.set_yticks(range(len(jlist_int))); ax3.set_yticklabels(labels_j)
ax3.set_xlabel(r"$j_2$"); ax3.set_ylabel(r"$j_1$")
ax3.set_title(r"$\int d^{j_1}_{1,0} d^{j_2}_{1,0}\sin\theta\,d\theta$"
              r"  vs  $\frac{2}{2j+1}\delta_{j_1 j_2}$")
for i in range(len(jlist_int)):
    for k in range(len(jlist_int)):
        ax3.text(k, i, f"{M_int[i,k]:+.3f}", ha="center", va="center",
                 fontsize=8, color="black")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)


# ---------- (c) 螺度振幅 phi 因子玩具模型 ----------
# 玩具 S 矩阵: S(la, lb -> lc, ld) = f(theta) * exp(i (la-lb-lc+ld) phi)
# 我们用 d^j 函数构造 f(theta) (没有物理模型, 只测 phi 行为)
def toy_helicity_S(la, lb, lc, ld, theta, phi, j_max=2):
    dlam = la - lb
    dlam_p = lc - ld
    # f(theta) = sum_j (2j+1)/(4pi) * d^j_{dlam_p, dlam}(theta)
    f = 0.0 + 0j
    for j2 in range(int(2*max(abs(dlam), abs(dlam_p))), 2*j_max + 1, 1):
        j = j2 / 2.0
        if (2*j) % 2 != (2*abs(dlam)) % 2:
            continue
        f = f + (2*j + 1) / (4*np.pi) * wigner_d(j, dlam_p, dlam, theta)
    Delta_lambda = la - lb - lc + ld
    return f * np.exp(1j * Delta_lambda * phi)


phis = np.linspace(0, 2*np.pi, 200)
theta0 = np.pi / 3   # 任选一个 θ

ax4 = fig.add_subplot(2, 2, 4)

# no-flip:  la=1/2, lb=1/2, lc=1/2, ld=1/2  =>  Δλ = 1/2 - 1/2 - 1/2 + 1/2 = 0
S_noflip = np.array([toy_helicity_S(0.5, 0.5, 0.5, 0.5, theta0, ph) for ph in phis])
# single-flip: la=1/2, lb=1/2, lc=-1/2, ld=1/2 => Δλ = 1/2 - 1/2 - (-1/2) + 1/2 = 1
S_flip1  = np.array([toy_helicity_S(0.5, 0.5, -0.5, 0.5, theta0, ph) for ph in phis])
# double-flip: la=1/2, lb=-1/2, lc=-1/2, ld=1/2 => Δλ = 1/2 - (-1/2) - (-1/2) + 1/2 = 2
S_flip2  = np.array([toy_helicity_S(0.5, -0.5, -0.5, 0.5, theta0, ph) for ph in phis])

ax4.plot(phis, np.real(S_noflip)/np.abs(S_noflip[0]), label=r"$\Delta\lambda=0$ Re/|S|",
         lw=1.5)
ax4.plot(phis, np.real(S_flip1)/np.abs(S_flip1[0]),  label=r"$\Delta\lambda=1$ Re/|S|",
         lw=1.5)
ax4.plot(phis, np.real(S_flip2)/np.abs(S_flip2[0]),  label=r"$\Delta\lambda=2$ Re/|S|",
         lw=1.5)
# 解析期望: cos(Δλ φ)
ax4.plot(phis, np.cos(0*phis), 'k:', lw=0.8)
ax4.plot(phis, np.cos(1*phis), 'k:', lw=0.8)
ax4.plot(phis, np.cos(2*phis), 'k:', lw=0.8, label=r"$\cos(\Delta\lambda\,\varphi)$")
ax4.set_xlabel(r"$\varphi$"); ax4.set_ylabel(r"$\mathrm{Re}\,S/|S|$")
ax4.set_title(rf"螺度振幅的 $\varphi$ 依赖 ($\theta={theta0:.2f}$)")
ax4.legend(frameon=False, fontsize=8); ax4.grid(alpha=0.3)

# 数值验证: 提取相位斜率
def fit_phase_slope(S, phis):
    ang = np.unwrap(np.angle(S))
    p = np.polyfit(phis, ang, 1)
    return p[0]
slope0 = fit_phase_slope(S_noflip, phis)
slope1 = fit_phase_slope(S_flip1, phis)
slope2 = fit_phase_slope(S_flip2, phis)
print(f"[phi factor] slopes: Δλ=0 -> {slope0:+.4f},  "
      f"Δλ=1 -> {slope1:+.4f},  Δλ=2 -> {slope2:+.4f}")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ch06_wignerD.pdf", bbox_inches="tight")
plt.close(fig)
print("Done. Figure saved to figures/ch06_wignerD.pdf")
