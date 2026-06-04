#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch12_coulomb.py
===============

第 12 章 (Coulomb 散射) 数值实验.

四个面板:
  (1) 正则 Coulomb 波函数 F_0(eta, rho), 不同 eta, 与自由 sin(rho) 比较,
      展示长程 1/r 尾部诱导的对数相位畸变 ~ -eta * ln(2 rho).
  (2) Rutherford 微分截面 d sigma / d Omega = eta^2 / [4 k^2 sin^4(theta/2)]
      与 Coulomb 振幅幅值 |f_C(theta)| (对数标度).
  (3) ``截断'' 总截面 sigma(theta_min) = int_{theta_min}^pi 2 pi sin th |f_C|^2 d th
      在 theta_min -> 0 时发散.
  (4) Coulomb 相移 sigma_ell = arg Gamma(ell + 1 + i eta), 不同 eta vs ell.

输出: figures/ch12_coulomb.pdf
"""

import os
import numpy as np
import mpmath
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.special import loggamma


# ---------------------------------------------------------------------------
# 工具: 把 mpmath 标量函数向量化为 numpy float 数组
# ---------------------------------------------------------------------------
def coulomb_F(ell, eta, rho_arr):
    """正则 Coulomb 函数 F_ell(eta, rho), 输入为 numpy 数组, 返回 numpy 数组."""
    out = np.empty_like(rho_arr, dtype=float)
    for i, r in enumerate(rho_arr):
        if r <= 0.0:
            out[i] = 0.0
        else:
            out[i] = float(mpmath.coulombf(ell, eta, r))
    return out


def sigma_ell(ell, eta):
    """Coulomb 相移 sigma_ell = arg Gamma(ell + 1 + i eta), 用 scipy.loggamma 高效计算."""
    return np.imag(loggamma(ell + 1.0 + 1j * eta))


# ---------------------------------------------------------------------------
# 面板 (1): F_0(eta, rho) 与对数相位畸变
# ---------------------------------------------------------------------------
def panel_F_logdistortion(ax):
    rho = np.linspace(0.05, 30.0, 600)
    etas = [0.0, 1.0, 2.0, -1.0]
    colors = ["k", "#1f77b4", "#d62728", "#2ca02c"]
    labels = [r"$\eta=0$ (free $\sin\rho$)", r"$\eta=+1$ (repulsive)",
              r"$\eta=+2$ (strong rep.)", r"$\eta=-1$ (attractive)"]
    for eta, c, lab in zip(etas, colors, labels):
        F = coulomb_F(0, eta, rho)
        ax.plot(rho, F, color=c, lw=1.3, label=lab)
    # 对数相位标记: 渐近 sin(rho - eta ln 2 rho + sigma_0) 的 eta=2 参考曲线
    eta_ref = 2.0
    sig0 = sigma_ell(0, eta_ref)
    ax.plot(rho, np.sin(rho - eta_ref * np.log(2.0 * rho) + sig0),
            color="#d62728", lw=0.8, ls="--",
            label=r"asymp. $\sin(\rho-\eta\ln 2\rho+\sigma_0),\ \eta=2$")
    ax.set_xlabel(r"$\rho = k r$")
    ax.set_ylabel(r"$F_0(\eta,\rho)$")
    ax.set_title(r"(1) Regular Coulomb wave $F_0$: log phase distortion")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.4, 1.4)


# ---------------------------------------------------------------------------
# 面板 (2): Rutherford 角分布 |f_C|^2
# ---------------------------------------------------------------------------
def panel_rutherford(ax):
    theta = np.linspace(np.deg2rad(1.0), np.pi - 1e-3, 400)
    # 取无量纲 k=1, 不同 eta
    for eta, c in zip([0.5, 1.0, 2.0], ["#1f77b4", "#d62728", "#2ca02c"]):
        dsigma = eta**2 / (4.0 * np.sin(theta / 2.0)**4)
        ax.semilogy(np.rad2deg(theta), dsigma, color=c, lw=1.4,
                    label=fr"$\eta={eta}$  Rutherford")
    # 也叠加 Coulomb 振幅 (与 Rutherford 完全一致, 仅相位不同)
    eta = 1.0
    sig0 = sigma_ell(0, eta)
    fC = -eta / (2.0 * np.sin(theta / 2.0)**2) * np.exp(
        -1j * eta * np.log(np.sin(theta / 2.0)**2) + 2j * sig0)
    ax.semilogy(np.rad2deg(theta), np.abs(fC)**2,
                color="k", lw=0.7, ls=":",
                label=r"$|f_C|^2,\ \eta=1$ (overlaps Rutherford)")
    ax.set_xlabel(r"$\theta$  (deg)")
    ax.set_ylabel(r"$d\sigma/d\Omega$  (units $1/k^2$)")
    ax.set_title(r"(2) Rutherford differential cross section $\propto \sin^{-4}(\theta/2)$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0, 180)


# ---------------------------------------------------------------------------
# 面板 (3): 截断总截面 sigma(theta_min) -> infinity
# ---------------------------------------------------------------------------
def panel_total_divergence(ax):
    theta_min = np.logspace(-3, np.log10(np.pi / 2), 200)  # 从 ~0.001 rad
    # sigma(theta_min) = 2 pi int_{theta_min}^{pi} sin th * eta^2/(4 k^2 sin^4(th/2)) d th
    # 解析: 令 u = sin^2(th/2), du = sin(th/2) cos(th/2) d th = (1/2) sin th d th
    # => integrand = eta^2 / (4 k^2 u^2) * 2 du
    # int = (eta^2 / (2 k^2)) [-1/u]_{sin^2(theta_min/2)}^{1}  * 2pi (azimuthal)
    #     = (pi eta^2 / k^2) [ 1/sin^2(theta_min/2) - 1 ]  (因为 sin^2(pi/2)=1)
    for eta, c in zip([0.5, 1.0, 2.0], ["#1f77b4", "#d62728", "#2ca02c"]):
        sig = (np.pi * eta**2) * (1.0 / np.sin(theta_min / 2.0)**2 - 1.0)
        ax.loglog(theta_min, sig, color=c, lw=1.5, label=fr"$\eta={eta}$")
    # 渐近: sigma ~ pi eta^2 * (2/theta_min)^2 = 4pi eta^2 / theta_min^2
    ax.loglog(theta_min, 4.0 * np.pi * 1.0**2 / theta_min**2,
              color="k", lw=0.7, ls="--",
               label=r"$\sim 4\pi\eta^2/\theta_{\min}^2,\ \eta=1$")
    ax.set_xlabel(r"$\theta_{\min}$  (rad)")
    ax.set_ylabel(r"$\sigma(\theta_{\min})$  (units $1/k^2$)")
    ax.set_title(r"(3) Truncated total cross section diverges as $\theta_{\min}\to 0$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")


# ---------------------------------------------------------------------------
# 面板 (4): Coulomb 相移 sigma_ell vs ell
# ---------------------------------------------------------------------------
def panel_sigmal(ax):
    ells = np.arange(0, 21)
    for eta, c in zip([0.5, 1.0, 2.0], ["#1f77b4", "#d62728", "#2ca02c"]):
        sig = np.array([sigma_ell(int(L), eta) for L in ells])
        ax.plot(ells, sig, "o-", color=c, lw=1.2, ms=4,
                label=fr"$\sigma_\ell,\ \eta={eta}$")
        # 大 ell 渐近: arg Gamma(l + 1 + i eta) ~ eta * ln(l) (Stirling)
        ax.plot(ells[2:], eta * np.log(ells[2:].astype(float)),
                color=c, lw=0.7, ls="--",
                label=fr"$\eta\ln\ell,\ \eta={eta}$" if eta == 1.0 else None)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma_\ell = \arg\Gamma(\ell+1+i\eta)$")
    ax.set_title(r"(4) Coulomb phase shifts grow $\sim\eta\ln\ell$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    panel_F_logdistortion(axes[0, 0])
    panel_rutherford(axes[0, 1])
    panel_total_divergence(axes[1, 0])
    panel_sigmal(axes[1, 1])
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/ch12_coulomb.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

    # 数值诊断
    print("\n--- 数值诊断 ---")
    for eta in [0.5, 1.0, 2.0]:
        s0 = sigma_ell(0, eta)
        s5 = sigma_ell(5, eta)
        print(f"eta={eta}:  sigma_0 = {s0:+.6f},  sigma_5 = {s5:+.6f}")
    # 验证 F_0 在 rho=20, eta=1 的渐近行为
    rho_t = 20.0
    eta_t = 1.0
    F = float(mpmath.coulombf(0, eta_t, rho_t))
    F_asymp = np.sin(rho_t - eta_t * np.log(2.0 * rho_t) + sigma_ell(0, eta_t))
    print(f"F_0(1, 20)         = {F:+.6f}")
    print(f"sin(rho - eta ln 2rho + sigma_0) = {F_asymp:+.6f}")
    print(f"差值 = {F - F_asymp:+.2e}  (应远小于 1)")


if __name__ == "__main__":
    main()
