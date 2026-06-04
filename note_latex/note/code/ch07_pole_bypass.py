#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ch07_pole_bypass.py
===================

第 7 章 数值实验: Green 算符 / T-矩阵 / 色散积分中, 处理积分核

    1 / (E - E' + i*epsilon)

奇异性的四种典型方法的对比.

被积函数模型
------------
    I(E) = \int_0^\infty f(E') / (E - E' + i*epsilon) dE'

其中 f(E') = exp(-E'/E0), E0 = 5, 在 E = 1 处求值.
精确极限 (epsilon -> 0+):

    Re I = P.V. \int_0^\infty f(E') / (E - E') dE'
    Im I = -pi * f(E)

本脚本对四种方法做数值对比, 并作出收敛 / 等价性 / 围道的可视化.

输出: figures/ch07_pole_bypass.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.special import exp1   # E1(z) = \int_z^\infty e^{-t}/t dt

# ============== 物理 / 数值参数 ==============
E    = 1.0          # 求值能量 (落在积分区间内, 是奇异点)
E0   = 5.0          # 指数衰减尺度
Emax = 200.0        # 积分上限截断 (足够远以使尾部贡献 << 1e-8)
N    = 200001       # 实轴均匀网格点数 (奇数, 保证 E 落在网格中点附近)

def f(Ep):
    """光滑权重函数 f(E')."""
    return np.exp(-Ep / E0)

# ---------- "解析" 参考解 ----------
# Im 部分严格为 -pi * f(E).
Im_exact = -np.pi * f(E)

# Re 部分用高精度参考: PV 积分写成
#     P.V. \int_0^Emax f(E')/(E-E') dE'
#  = \int_0^Emax [f(E')-f(E)] / (E-E') dE'  (光滑)
#    + f(E) * P.V. \int_0^Emax 1/(E-E') dE'
#    + (尾部 \int_Emax^\infty f(E')/(E-E') dE')
# P.V. 主值: \int_0^Emax 1/(E-E') dE' = ln|E/(Emax-E)|  (E < Emax)
def reference_real():
    # 光滑部分: 用 scipy quad
    from scipy.integrate import quad
    def smooth(Ep):
        if abs(Ep - E) < 1e-12:
            # 极限: f'(E) * (-1)
            return (-1.0/E0) * np.exp(-E/E0) * (-1.0)
        return (f(Ep) - f(E)) / (E - Ep)
    val_smooth, _ = quad(smooth, 0.0, Emax, limit=400)
    val_log = f(E) * np.log(E / (Emax - E))
    # Emax 之外的尾部: f(E')/(E-E'), E'>>E, 近似 -f(E')/E', 极小
    def tail(Ep):
        return f(Ep) / (E - Ep)
    val_tail, _ = quad(tail, Emax, np.inf, limit=400)
    return val_smooth + val_log + val_tail

Re_exact = reference_real()
print(f"参考值: Re I = {Re_exact:+.10f}, Im I = {Im_exact:+.10f}")

# ============== 方法 (a): +i*epsilon, 有限 epsilon, 实轴均匀积分 ==============
def method_a(eps):
    Ep = np.linspace(0.0, Emax, N)
    integrand = f(Ep) / (E - Ep + 1j * eps)
    # 复合 Simpson
    return _simpson(integrand, Ep)

def _simpson(y, x):
    # 等距 Simpson 复合规则; len(x) 必须为奇数
    h = x[1] - x[0]
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
    return s * h / 3.0

# ============== 方法 (b): Sokhotski-Plemelj / 主值 + delta 拆分 ==============
def method_b():
    # Re: 在奇异点附近做 f(E')-f(E) 减法 (光滑后用 Simpson)
    Ep = np.linspace(0.0, Emax, N)
    # 减法被积函数 g(E') = (f(E')-f(E)) / (E-E'), 在 E'=E 处取极限 -f'(E)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (f(Ep) - f(E)) / (E - Ep)
    # 找到最接近 E 的网格点, 用 L'Hopital 极限替换
    j = np.argmin(np.abs(Ep - E))
    # f'(E') = -1/E0 * exp(-E'/E0), 极限 g(E) = -f'(E) = (1/E0)*exp(-E/E0)
    g[j] = (1.0/E0) * np.exp(-E/E0)
    val_smooth = _simpson(g, Ep)
    # 主值的解析对数部分:  P.V. \int_0^Emax 1/(E-E') dE' = ln(E/(Emax-E))
    val_log = f(E) * np.log(E / (Emax - E))
    Re = val_smooth + val_log
    # Im: 解析  -pi f(E)
    Im = -np.pi * f(E)
    return Re + 1j * Im

# ============== 方法 (c): 围道形变 (小半圆绕开极点) ==============
# 路径: 0 -> E - r  (实轴),  E - r -> E + r (下半平面半圆),  E + r -> Emax (实轴).
# 半圆参数化: E' = E + r * exp(i*theta), theta: pi -> 2*pi (即下方半圆).
# 下半圆使得极点 E'=E 落在闭合围道 *外* 的方式与 +i*epsilon 等价.
def method_c(r=1e-3, n_arc=4001):
    from scipy.integrate import quad
    # 实轴段使用自适应积分 (端点处 f(E')/(E-E') 接近奇异, Simpson 需要极密网格)
    re_seg = lambda Ep: f(Ep) / (E - Ep)
    val1, _ = quad(re_seg, 0.0, E - r, limit=400)
    val2, _ = quad(re_seg, E + r, Emax, limit=400)
    # 半圆: theta: pi -> 2*pi, E' = E + r e^{i*theta} (下半平面)
    # 在此约定下, 对 1/(E-E'+i0+) 与原始 +i*eps 等价 (极点位于上半).
    theta = np.linspace(np.pi, 2.0 * np.pi, n_arc)
    Ep_arc = E + r * np.exp(1j * theta)
    dEp = 1j * r * np.exp(1j * theta)            # dE'/d theta
    integrand_arc = f(Ep_arc) / (E - Ep_arc) * dEp
    h = theta[1] - theta[0]
    arc = (integrand_arc[0] + integrand_arc[-1]
           + 4.0 * np.sum(integrand_arc[1:-1:2])
           + 2.0 * np.sum(integrand_arc[2:-1:2])) * h / 3.0
    return val1 + val2 + arc

# ============== 方法 (d): 减法 + 解析修正 ==============
# I = \int [f(E')-f(E)] / (E-E'+i*eps) dE'  +  f(E) \int dE' / (E-E'+i*eps)
# 第一项被积函数光滑 (E'->E 时极限存在), 直接 Simpson, eps 可取 0.
# 第二项解析:
#   \int_0^Emax 1/(E-E'+i*eps) dE'   eps->0+
#     = ln(E + i*eps) - ln(E - Emax + i*eps)
#     = ln|E/(Emax-E)| - i*pi              ( E < Emax,  Emax-E>0 故 E-Emax<0,
#                                            ln(- |Emax-E| + i 0+) = ln|Emax-E| + i*pi
#                                            合并后取 -i*pi 给 1/(E-E'+i0+) )
def method_d():
    Ep = np.linspace(0.0, Emax, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (f(Ep) - f(E)) / (E - Ep)
    j = np.argmin(np.abs(Ep - E))
    g[j] = (1.0/E0) * np.exp(-E/E0)
    val_smooth = _simpson(g, Ep)
    # 解析部分
    val_anal = f(E) * (np.log(E / (Emax - E)) - 1j * np.pi)
    return val_smooth + val_anal

# ============== 运行实验 ==============
# --- (a) epsilon 扫描 ---
eps_list = np.logspace(-1, -4, 13)
results_a = np.array([method_a(eps) for eps in eps_list])
err_re_a  = np.abs(results_a.real - Re_exact)
err_im_a  = np.abs(results_a.imag - Im_exact)

# --- (b), (c), (d) 单次结果 ---
res_b = method_b()
res_c = method_c(r=1e-3)
res_d = method_d()

# 选 (a) 在 eps = 1e-3 (典型值) 作为 "代表收敛值"
res_a_repr = results_a[np.argmin(np.abs(eps_list - 1e-3))]

print("\n=== 四种方法结果 ===")
for name, val in [("(a) +i*eps, eps=1e-3", res_a_repr),
                  ("(b) Sokhotski-Plemelj", res_b),
                  ("(c) Contour deform.  ", res_c),
                  ("(d) Subtract + anal. ", res_d),
                  ("    Reference        ", Re_exact + 1j * Im_exact)]:
    print(f"{name}:  Re = {val.real:+.8f}, Im = {val.imag:+.8f}")

# ============== 作图 ==============
os.makedirs("figures", exist_ok=True)
fig = plt.figure(figsize=(13.5, 4.4))

# Panel 1: (a) 收敛性
ax1 = fig.add_subplot(1, 3, 1)
ax1.loglog(eps_list, err_re_a, 'o-', lw=1.6, label=r"|Re error|")
ax1.loglog(eps_list, err_im_a, 's-', lw=1.6, label=r"|Im error|")
ax1.loglog(eps_list, eps_list, '--', color='gray', lw=1.0,
           label=r"$\propto\epsilon$")
ax1.set_xlabel(r"$\epsilon$")
ax1.set_ylabel("absolute error")
ax1.set_title(r"(a) Convergence of +i$\epsilon$ method")
ax1.legend(frameon=False, fontsize=9)
ax1.grid(which='both', alpha=0.3)
ax1.invert_xaxis()

# Panel 2: 四种方法的 Re / Im 对比 (柱状图)
ax2 = fig.add_subplot(1, 3, 2)
labels = ['(a)\n$\\epsilon=10^{-3}$', '(b)\nPV+$\\delta$',
          '(c)\nContour', '(d)\nSubtract']
re_vals = [res_a_repr.real, res_b.real, res_c.real, res_d.real]
im_vals = [res_a_repr.imag, res_b.imag, res_c.imag, res_d.imag]
xpos = np.arange(len(labels))
w = 0.35
ax2.bar(xpos - w/2, re_vals, w, color='#1f77b4', label='Re', alpha=0.85)
ax2.bar(xpos + w/2, im_vals, w, color='#d62728', label='Im', alpha=0.85)
ax2.axhline(Re_exact, ls='--', color='#1f77b4', lw=1.0)
ax2.axhline(Im_exact, ls='--', color='#d62728', lw=1.0)
ax2.text(3.4, Re_exact, f"Re$_{{\\rm ref}}$={Re_exact:+.4f}",
         color='#1f77b4', va='bottom', ha='right', fontsize=8)
ax2.text(3.4, Im_exact, f"Im$_{{\\rm ref}}$={Im_exact:+.4f}",
         color='#d62728', va='top', ha='right', fontsize=8)
ax2.set_xticks(xpos)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("value")
ax2.set_title("Comparison of four methods at convergence")
ax2.legend(frameon=False, fontsize=9, loc='center right')
ax2.grid(axis='y', alpha=0.3)

# Panel 3: 围道可视化
ax3 = fig.add_subplot(1, 3, 3)
ax3.axhline(0.0, color='gray', lw=0.6)
ax3.axvline(0.0, color='gray', lw=0.6)
# 实轴段
ax3.plot([0.0, E - 0.15], [0.0, 0.0], color='#1f77b4', lw=2.2)
ax3.plot([E + 0.15, 2.5], [0.0, 0.0], color='#1f77b4', lw=2.2)
# 半圆 (下半平面)
th = np.linspace(np.pi, 2*np.pi, 200)
ax3.plot(E + 0.15*np.cos(th), 0.15*np.sin(th), color='#d62728', lw=2.2,
         label='small semicircle (lower half)')
# 极点
ax3.plot(E, 0.0, 'kx', ms=10, mew=2.0, label='pole $E^\\prime = E$')
# +i*epsilon 等价图示
ax3.plot(E, 0.05, 'b+', ms=10, mew=1.6,
         label='+i$\\epsilon$ shifted pole')
# 箭头方向
for xa in [0.3, 0.7]:
    ax3.annotate('', xy=(xa+0.1, 0), xytext=(xa, 0),
                 arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4))
for xa in [1.4, 2.0]:
    ax3.annotate('', xy=(xa+0.1, 0), xytext=(xa, 0),
                 arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4))
# 半圆方向
ax3.annotate('', xy=(E, -0.15+0.005), xytext=(E - 0.1, -0.11),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.4))
ax3.set_xlim(-0.2, 2.6)
ax3.set_ylim(-0.4, 0.4)
ax3.set_xlabel(r"Re $E^\prime$")
ax3.set_ylabel(r"Im $E^\prime$")
ax3.set_title("(c) Contour deformation (lower-half semicircle)")
ax3.set_aspect('equal')
ax3.legend(frameon=False, loc='lower right', fontsize=8)
ax3.grid(alpha=0.3)

fig.tight_layout()
out = "figures/ch07_pole_bypass.pdf"
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved -> {out}")
