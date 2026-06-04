#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ch09_lippmann_schwinger.py
==========================

第 9 章 数值实验：求解 s 波 Lippmann-Schwinger 方程
    u^+(r) = sin(kr) + \int_0^\infty dr' G^0_0(r,r';E) [2 m V(r')] u^+(r'),
其中实轴 (主值/对称) 形式的自由 s 波 Green 函数为
    G^0_0(r,r';E) = - sin(k r_<) cos(k r_>) / k.
势取为吸引方势阱 V(r) = -V0 (r<a), 0 (r>a).

两种方法:
  (A) 离散化 LS 方程为 (I - K) u = u0, 数值线性代数求逆.
  (B) 直接积分径向 Schroedinger 方程, 大 r 处与自由解 sin(kr+delta_0)/k 拟合.

输出:
  figures/ch09_lippmann_schwinger.pdf   —— 三联图
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- 物理与数值参数 (自然单位 hbar = m = 1) ----------
V0   = 4.0          # 阱深 (吸引)
a    = 1.0          # 阱半径
Rmax = 30.0         # 计算区域
N    = 2400         # 网格点数
r    = np.linspace(Rmax / N, Rmax, N)       # 不含 r=0 (sin(kr)|_0 = 0)
dr   = r[1] - r[0]

def V_well(rr):
    """吸引方势阱."""
    return np.where(rr < a, -V0, 0.0)

# ---------- 方法 A: LS 矩阵反演 ----------
def G0_swave(rr, rrp, k):
    """自由 s 波 (实) Green 函数 G^0_0(r,r';E) = -sin(k r_<) cos(k r_>) / k."""
    r_lt = np.minimum(rr, rrp)
    r_gt = np.maximum(rr, rrp)
    return -np.sin(k * r_lt) * np.cos(k * r_gt) / k

def solve_LS_matrix(k):
    """构造矩阵方程 (I - K) u = u0 并求解."""
    R, Rp = np.meshgrid(r, r, indexing="ij")
    G  = G0_swave(R, Rp, k)
    Vp = V_well(r)
    K  = G * (2.0 * Vp)[None, :] * dr        # 梯形/中点积分权
    u0 = np.sin(k * r)
    u  = np.linalg.solve(np.eye(N) - K, u0)
    return u

# ---------- 方法 B: 直接积分径向 Schroedinger ----------
def solve_radial_ODE(k):
    """对 s 波 Schroedinger u'' + (k^2 - 2 m V) u = 0 从 r=0 向外积分."""
    def rhs(rr, y):
        u, up = y
        return [up, -(k**2 - 2.0 * V_well(np.array([rr]))[0]) * u]
    # 阱内 r<a 时 K^2 = k^2 + 2 V0, 起始斜率取使 r->0 时 u ~ sin(K r) 的归一
    Kin = np.sqrt(k**2 + 2.0 * V0)
    y0 = [0.0, 1.0]                          # u(0)=0, u'(0)=1 (任意整体常数, 后归一)
    sol = solve_ivp(rhs, (0.0, Rmax), y0, t_eval=r, rtol=1e-10, atol=1e-12)
    return sol.y[0]

def extract_delta0(u, k, r_fit_min=10.0):
    """在大 r 处用 u(r) ~ A sin(k r + delta_0) 拟合提取 delta_0."""
    mask = r > r_fit_min
    rr = r[mask]; uu = u[mask]
    # A sin(k r + delta) = A cos(delta) sin(kr) + A sin(delta) cos(kr)
    M = np.column_stack([np.sin(k * rr), np.cos(k * rr)])
    c, *_ = np.linalg.lstsq(M, uu, rcond=None)
    A_cos, A_sin = c
    delta = np.arctan2(A_sin, A_cos)
    A = np.sqrt(A_cos**2 + A_sin**2)
    # 折叠到 (-pi/2, pi/2] 区间 (s 波相移按 Levinson 定理常约定)
    while delta >  np.pi / 2: delta -= np.pi
    while delta <= -np.pi / 2: delta += np.pi
    return delta, A

def delta0_analytic(k):
    """方势阱 s 波相移解析式: tan(K a)/K = tan(k a + delta)/k."""
    Kin = np.sqrt(k**2 + 2.0 * V0)
    # k cot(k a + delta) = K cot(K a)  =>  delta = atan(k tan(K a)/K) - k a
    delta = np.arctan(k * np.tan(Kin * a) / Kin) - k * a
    while delta >  np.pi / 2: delta -= np.pi
    while delta <= -np.pi / 2: delta += np.pi
    return delta

# ---------- 主流程 ----------
energies = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
ks = np.sqrt(2.0 * energies)                  # E = k^2/2

deltas_LS, deltas_ODE, deltas_AN = [], [], []
psi_LS_store = {}

for E, k in zip(energies, ks):
    uA = solve_LS_matrix(k)
    uB = solve_radial_ODE(k)
    dA, _ = extract_delta0(uA, k)
    dB, _ = extract_delta0(uB, k)
    dC = delta0_analytic(k)
    deltas_LS.append(dA); deltas_ODE.append(dB); deltas_AN.append(dC)
    psi_LS_store[E] = (uA, uB)
    print(f"E={E:5.2f}  k={k:5.3f}  delta(LS)={dA:+.4f}  "
          f"delta(ODE)={dB:+.4f}  delta(analytic)={dC:+.4f}")

deltas_LS  = np.array(deltas_LS)
deltas_ODE = np.array(deltas_ODE)
deltas_AN  = np.array(deltas_AN)

# ---------- 光学定理验证 (s 波贡献) ----------
# s 波散射振幅 f_0 = (e^{i delta_0} sin delta_0) / k
# 仅 s 波时 sigma_tot = (4 pi / k^2) sin^2 delta_0
# Im f_0(0) = sin^2(delta_0) / k = (k / (4 pi)) * sigma_tot --> 自动成立 (恒等式)
E_check = 1.0
k_check = np.sqrt(2.0 * E_check)
delta_check, _ = extract_delta0(solve_LS_matrix(k_check), k_check)
Im_f0      = np.sin(delta_check)**2 / k_check
sigma_tot  = (4.0 * np.pi / k_check**2) * np.sin(delta_check)**2
RHS        = (k_check / (4.0 * np.pi)) * sigma_tot
print(f"\nOptical theorem check at E={E_check}:")
print(f"  delta_0 = {delta_check:+.6f}")
print(f"  Im f_0(0)            = {Im_f0:.6f}")
print(f"  (k/4pi) sigma_tot    = {RHS:.6f}")
print(f"  relative error       = {abs(Im_f0 - RHS) / abs(Im_f0):.2e}")

# ---------- 作图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

# (1) |psi^+(r)|^2 = |u^+(r)/r|^2 (与 sin(kr)/(kr) 比较)
ax = axes[0]
plot_Es = [0.5, 2.0, 6.0]
colors = ["#1f77b4", "#d62728", "#2ca02c"]
for E, c in zip(plot_Es, colors):
    k = np.sqrt(2.0 * E)
    uA, _ = psi_LS_store[E]
    psi2 = (uA / r) ** 2
    free = (np.sin(k * r) / (k * r)) ** 2
    # 归一化: 大 r 处振幅匹配 1/k
    scale = 1.0 / (np.max(np.abs(uA[r > 5])) ** 2)
    ax.plot(r, psi2 * scale, color=c, lw=1.6, label=f"$E={E:g}$ (LS)")
    ax.plot(r, free * scale * (np.max(np.abs(uA[r > 5]))**2),
            color=c, lw=1.0, ls="--", alpha=0.6)
ax.axvline(a, color="gray", ls=":", lw=0.8)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$|\psi^+(r)|^2$  (arb. units)")
ax.set_title(r"solid: full $|u^+/r|^2$;  dashed: free $|\sin(kr)/(kr)|^2$")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.3)

# (2) delta_0(E)
ax = axes[1]
E_dense = np.linspace(0.05, 10.0, 200)
k_dense = np.sqrt(2.0 * E_dense)
delta_dense = np.array([delta0_analytic(kk) for kk in k_dense])
ax.plot(E_dense, delta_dense, "k-", lw=1.4, label="analytic")
ax.plot(energies, deltas_LS,  "o", ms=7, mfc="none", color="#1f77b4",
        label="LS matrix inv.")
ax.plot(energies, deltas_ODE, "x", ms=8, color="#d62728",
        label="ODE integration")
ax.set_xlabel(r"$E = k^2/2$")
ax.set_ylabel(r"$\delta_0(E)$")
ax.set_title(r"s-wave phase shift vs energy  ($V_0=4,\ a=1$)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# (3) 光学定理
ax = axes[2]
sin2_LS  = np.sin(deltas_LS) ** 2
sin2_AN  = np.sin(deltas_AN) ** 2
Im_f0_arr  = sin2_LS / ks
RHS_arr    = (ks / (4.0 * np.pi)) * (4.0 * np.pi / ks**2) * sin2_LS  # = sin^2 / k
ax.plot(energies, Im_f0_arr, "o-",  color="#1f77b4", label=r"$\mathrm{Im}\,f_0(0)$")
ax.plot(energies, RHS_arr,   "s--", color="#d62728",
        label=r"$(k/4\pi)\sigma_{\rm tot}$")
ax.plot(energies, np.sin(deltas_AN)**2 / ks, "k:", lw=1.0, label="analytic")
ax.set_xlabel(r"$E$")
ax.set_ylabel(r"both sides of optical theorem")
ax.set_title(rf"optical theorem (E={E_check}: rel. err. ${abs(Im_f0-RHS)/abs(Im_f0):.1e}$)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

fig.suptitle(r"Numerical solution of the s-wave Lippmann-Schwinger eq. "
             r"(attractive square well)", fontsize=12, y=1.02)
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ch09_lippmann_schwinger.pdf", bbox_inches="tight")
plt.close(fig)

print("\nDone. Figure saved to figures/ch09_lippmann_schwinger.pdf")
