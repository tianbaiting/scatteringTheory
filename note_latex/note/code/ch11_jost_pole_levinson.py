#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch11_jost_pole_levinson.py
==========================

第 11 章 数值实验: 吸引方势阱 V(r) = -V0 * theta(a-r) 的 s 波 Jost 函数,
S 矩阵极点搜索, 以及 Levinson 定理的数值验证.

自然单位: hbar = 2m = 1, a = 1.

s 波 Jost 函数 (Taylor §11):
    F0(p) = exp(i p a) [ cos(K a) - i (p/K) sin(K a) ],   K = sqrt(p^2 + V0).

S 矩阵: S0(p) = F0(-p) / F0(p);
束缚态 <-> F0(p) 在上半平面虚轴上的零点 p = i*kappa_n;
虚态     <-> F0(p) 在下半平面虚轴上的零点 p = -i*kappa_n.

输出: figures/ch11_jost_pole_levinson.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.optimize import newton

a = 1.0  # 阱宽

# ---------- Jost 函数 ----------
def F0(p, V0):
    """s 波 Jost 函数 F0(p; V0). 支持复 p."""
    p = np.asarray(p, dtype=complex)
    K = np.sqrt(p * p + V0)
    return np.exp(1j * p * a) * (np.cos(K * a) - 1j * (p / K) * np.sin(K * a))

def dF0(p, V0, h=1e-6):
    """复数差分得到 F0' 用于 Newton 法."""
    return (F0(p + h, V0) - F0(p - h, V0)) / (2.0 * h)

def S0(p, V0):
    """S 矩阵 S0(p) = F0(-p)/F0(p)."""
    return F0(-p, V0) / F0(p, V0)

# ---------- 解析束缚态条件 ----------
def bound_state_kappas(V0):
    """求解 kappa * tan(K a) = ... 等价的 Jost 零点条件.
       束缚态: p = i kappa, kappa>0.
       F0(i kappa) = 0  <=>  K cot(K a) = -kappa, K = sqrt(V0 - kappa^2).
       直接对 kappa in (0, sqrt(V0)) 找根."""
    def g(kappa):
        if kappa <= 0 or kappa * kappa >= V0:
            return np.nan
        K = np.sqrt(V0 - kappa * kappa)
        return K / np.tan(K * a) + kappa
    kappas = []
    grid = np.linspace(1e-6, np.sqrt(V0) - 1e-6, 4000)
    vals = np.array([g(k) for k in grid])
    for i in range(len(grid) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if np.isnan(v1) or np.isnan(v2):
            continue
        if v1 * v2 < 0 and abs(v1) < 1e3 and abs(v2) < 1e3:
            try:
                root = newton(g, 0.5 * (grid[i] + grid[i + 1]), tol=1e-12, maxiter=80)
                if 0 < root < np.sqrt(V0) and all(abs(root - r) > 1e-6 for r in kappas):
                    kappas.append(root)
            except RuntimeError:
                pass
    return sorted(kappas)

# ---------- 复平面零点搜索 (Newton) ----------
def find_jost_zero(p0, V0, tol=1e-12, maxiter=80, max_step=None):
    """从复平面种子 p0 出发, 用 Newton 法寻找 F0 的零点.
       max_step 若给定, 则限制单步位移幅度 (用于轨迹追踪防止跳到他根)."""
    p = complex(p0)
    for _ in range(maxiter):
        f  = F0(p, V0)
        fp = dF0(p, V0)
        if abs(fp) < 1e-30:
            return None
        dp = f / fp
        if max_step is not None and abs(dp) > max_step:
            dp = dp * (max_step / abs(dp))
        p -= dp
        if abs(dp) < tol:
            return p
    return None

# ---------- 相移 ----------
def delta0(p, V0):
    """物理相移, p>0 实数. 利用 S0 = exp(2 i delta0)."""
    s = S0(p, V0)
    return 0.5 * np.angle(s)

def delta0_unwrapped(p_arr, V0):
    """对实数 p>0 的相移做 unwrap, 选择 Levinson 约定:
       delta0(infty) = 0, 故 delta0(0) = n0 * pi >= 0.
       p_arr 必须升序, 且 p_arr[-1] 应足够大使 delta 已收敛."""
    d = delta0(p_arr, V0)
    d = np.unwrap(2.0 * d) / 2.0
    # 把高动量端归到 (-pi/2, pi/2] 即可视为 delta(infty) = 0
    n_hi = round(d[-1] / np.pi)
    d -= n_hi * np.pi
    return d

# =====================================================================
#  主程序
# =====================================================================
os.makedirs("figures", exist_ok=True)

V0_main = 30.0   # 主例: 较深势阱, 多束缚态

# 1) 解析束缚态动量
kap_an = bound_state_kappas(V0_main)
print(f"V0 = {V0_main}: 解析束缚态 kappa_n = {kap_an}")

# 2) 用 Newton 复平面搜索 (从 i*sqrt(V0 - (n+1/2)^2 pi^2) 种子)
seeds = []
n = 0
while True:
    arg = V0_main - ((n + 0.5) * np.pi / a) ** 2
    if arg <= 0:
        break
    seeds.append(1j * np.sqrt(arg))
    n += 1
poles = []
for s in seeds:
    z = find_jost_zero(s, V0_main)
    if z is not None and z.imag > 0 and all(abs(z - q) > 1e-6 for q in poles):
        poles.append(z)
poles.sort(key=lambda z: -z.imag)
print(f"Newton 找到的上半平面 Jost 零点: {poles}")

# =====================================================================
#  作图
# =====================================================================
fig = plt.figure(figsize=(11.5, 9.0))
gs  = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

# ---------- 面板 (1): log|S0| 复 p 平面热图 ----------
ax1 = fig.add_subplot(gs[0, 0])
pr = np.linspace(-6.0, 6.0, 401)
pi = np.linspace(-3.0, 6.5, 401)
PR, PI = np.meshgrid(pr, pi)
P = PR + 1j * PI
with np.errstate(divide="ignore", invalid="ignore"):
    LOG = np.log10(np.abs(S0(P, V0_main)) + 1e-300)
LOG = np.clip(LOG, -3.0, 3.0)
im = ax1.imshow(LOG, extent=[pr[0], pr[-1], pi[0], pi[-1]],
                origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=-3, vmax=3)
plt.colorbar(im, ax=ax1, label=r"$\log_{10}|S_0(p)|$")
# 标记上半平面极点 (S 矩阵的极点 = F0 的零点)
for q in poles:
    ax1.plot(q.real, q.imag, "k*", ms=14, mfc="yellow", mec="k", mew=1.0)
# 也标记下半平面 (S 的零点对应 F0(-p) 的零点 = -conj(上半零点))
for q in poles:
    qm = -q  # 由 F0(-p)=0 得到 S 的零点
    ax1.plot(qm.real, qm.imag, "ko", ms=8, mfc="white", mec="k", mew=1.0)
ax1.axhline(0.0, color="k", lw=0.6)
ax1.axvline(0.0, color="k", lw=0.6)
ax1.set_xlabel(r"Re $p$")
ax1.set_ylabel(r"Im $p$")
ax1.set_title(rf"$\log_{{10}}|S_0(p)|$, $V_0={V0_main}$  "
              r"(★ poles, $\circ$ zeros)")

# ---------- 面板 (2): 数值 vs 解析相移 ----------
ax2 = fig.add_subplot(gs[0, 1])
# 在足够长的区间 [1e-3, 2000] 上 unwrap, 然后只画 [0, 6] 部分
p_long = np.linspace(1e-3, 2000.0, 40000)
d_long = delta0_unwrapped(p_long, V0_main)
# 解析公式: tan(delta + p a) = (p/K) tan(K a)
K_long = np.sqrt(p_long ** 2 + V0_main)
d_an_raw = np.arctan((p_long / K_long) * np.tan(K_long * a)) - p_long * a
d_an = np.unwrap(2.0 * d_an_raw) / 2.0
n_hi = round(d_an[-1] / np.pi)
d_an -= n_hi * np.pi
mask = p_long <= 6.0
ax2.plot(p_long[mask], d_long[mask] / np.pi, "b-",  lw=2.0,
         label=r"numerical $\delta_0/\pi$")
ax2.plot(p_long[mask], d_an[mask]   / np.pi, "r--", lw=1.4,
         label=r"analytic $\delta_0/\pi$")
ax2.axhline(len(kap_an), color="gray", ls=":", lw=1.0,
            label=rf"$\delta_0(0)/\pi=n_0={len(kap_an)}$")
ax2.set_xlabel(r"$p$")
ax2.set_ylabel(r"$\delta_0(p)/\pi$")
ax2.set_title(r"s-wave phase shift, $V_0=$"+f"{V0_main}")
ax2.legend(frameon=False, loc="best")
ax2.grid(alpha=0.3)

# ---------- 面板 (3): Levinson 定理柱状对比 ----------
ax3 = fig.add_subplot(gs[1, 0])
V0_list = [2.0, 6.0, 12.0, 25.0, 40.0, 60.0]
n_bound = []
delta_diff = []
for V0 in V0_list:
    kk = bound_state_kappas(V0)
    n_bound.append(len(kk))
    # 用足够大的 p_max 让 delta(p_max) 收敛到渐近值
    p_grid = np.linspace(1e-3, 2000.0, 40000)
    d = delta0_unwrapped(p_grid, V0)
    delta_diff.append(d[0] - d[-1])  # delta0(0) - delta0(infty)
xs = np.arange(len(V0_list))
w = 0.38
ax3.bar(xs - w/2, [n*np.pi for n in n_bound], width=w, label=r"$n_0\,\pi$",
        color="#1f77b4")
ax3.bar(xs + w/2, delta_diff, width=w, label=r"$\delta_0(0)-\delta_0(\infty)$",
        color="#d62728")
ax3.set_xticks(xs)
ax3.set_xticklabels([f"{V0:g}" for V0 in V0_list])
ax3.set_xlabel(r"$V_0$")
ax3.set_ylabel("phase / radians")
ax3.set_title("Levinson check:  $n_0\\pi$  vs  $\\delta_0(0)-\\delta_0(\\infty)$")
ax3.legend(frameon=False)
ax3.grid(alpha=0.3, axis="y")
print("Levinson:", list(zip(V0_list, n_bound,
                            [round(dd / np.pi, 3) for dd in delta_diff])))

# ---------- 面板 (4): 极点轨迹 (V0 扫过阈值) ----------
ax4 = fig.add_subplot(gs[1, 1])
# 第 n 个束缚态阈值为 V0_n = ((n+1/2) pi / a)^2.
# 我们追踪 n=0 模式: 阈值 V0_0 = (pi/2)^2 ~ 2.467.
# V0 > V0_thr: 上半虚轴 (束缚态);  V0 < V0_thr: 下半虚轴 (虚态).
n_track = 0
V0_thr = ((n_track + 0.5) * np.pi / a) ** 2  # ~ 2.467
# 自高 V0 向下扫, 让束缚态平滑迁移成虚态
V0_scan = np.linspace(5.0, 0.5, 110)
trajec = []
prev = None
for V0 in V0_scan:
    if prev is None:
        # 首个种子 (V0=5): 用解析估计 kappa = sqrt(V0 - (pi/2)^2)
        arg = V0 - V0_thr
        seed = 1j * np.sqrt(max(arg, 1e-3))
    else:
        # 用上一步结果作种子, 限制步长以防跳到他根
        seed = prev
    z = find_jost_zero(seed, V0, max_step=0.1)
    if z is None or abs(z) > 5.0:
        trajec.append((V0, np.nan, np.nan))
        prev = None
    else:
        trajec.append((V0, z.real, z.imag))
        prev = z
trajec = np.array(trajec)
ax4.plot(trajec[:, 0], trajec[:, 2], "b-o", ms=3, lw=1.4,
         label=r"Im $p$  (bound $\to$ virtual)")
ax4.plot(trajec[:, 0], trajec[:, 1], "g-s", ms=3, lw=1.4,
         label=r"Re $p$")
ax4.axvline(V0_thr, color="k", ls="--", lw=1.0,
            label=rf"$V_0^{{\rm thr}}=(\pi/2)^2$")
ax4.axhline(0.0, color="gray", lw=0.6)
ax4.set_xlabel(r"$V_0$")
ax4.set_ylabel(r"pole position $p$")
ax4.set_title(rf"Pole trajectory: lowest $s$-wave pole ($n={n_track}$)")
ax4.legend(frameon=False, loc="best")
ax4.grid(alpha=0.3)

fig.suptitle("Square-well s-wave: Jost zeros, S-matrix poles, Levinson",
             fontsize=13)
fig.savefig("figures/ch11_jost_pole_levinson.pdf", bbox_inches="tight")
plt.close(fig)

print("Done. Figure saved to figures/ch11_jost_pole_levinson.pdf")
