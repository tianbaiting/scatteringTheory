#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ch14_resonance_breitwigner.py
=============================

Chapter 14 numerical experiment: l=1 attractive square-well resonance.

Goal:
    For an l=1 attractive square well that supports a (Gamow) resonance
    just above threshold, demonstrate the full set of "resonance fingerprints":
        (a) phase shift delta_1(k) rising rapidly through pi/2,
        (b) Breit-Wigner cross section sigma_1(E) and chi^2 fit,
        (c) Wigner time delay tau(E) = 2 hbar d delta_1/dE peaking at E_R,
        (d) complex S-pole on the second sheet at E_R - i Gamma/2,
        (e) survival probability P(t) of a quasi-stationary state showing
            exponential decay (Breit-Wigner regime) followed by an
            algebraic (Khalfin) tail.

Units: hbar = 2m = a = 1, so the radial reduced equation is
    u'' + (E - V(r) - l(l+1)/r^2) u = 0,
with k = sqrt(E), inside-well wavenumber kappa = sqrt(E + V0).
Square well: V(r) = -V0 for r < 1, 0 outside.

The l=1 attractive square well has an analytic phase shift formula via
matching of spherical Bessel functions; we use it as the "exact" curve.
For panel (e) we discretise the radial Hamiltonian on [0, R_box] with
Dirichlet boundary conditions (u(0) = u(R_box) = 0), diagonalise H,
and propagate via spectral expansion.

Output: figures/ch14_resonance_breitwigner.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ----------------------------- physics ---------------------------------------
V0 = 9.0                          # well depth, in units hbar=2m=a=1
A  = 1.0                          # well radius
# Choice rationale: at V0 = 9 the l = 1 attractive square well has the
# lowest "shape resonance" (a quasi-bound state trapped behind the
# centrifugal barrier) sitting at E_R ~ 0.35 with a moderate width
# Gamma ~ 0.3.  A deeper well such as V0 = (3 pi / 2)^2 ~ 22 produces a
# proper bound state instead, washing out the resonance fingerprint.


def j1(x):
    """Regular spherical Bessel j_1, safe at x=0."""
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small]  = x[small] / 3.0 - x[small] ** 3 / 30.0
    big         = ~small
    xb = x[big]
    out[big] = np.sin(xb) / xb ** 2 - np.cos(xb) / xb
    return out


def j1p(x):
    """Derivative d j_1 / dx, safe at x=0."""
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small] = 1.0 / 3.0 - x[small] ** 2 / 10.0
    big = ~small
    xb = x[big]
    # j_1'(x) = j_0(x) - 2 j_1(x)/x = sin x / x - 2/x*(sin x/x^2 - cos x/x)
    out[big] = (np.sin(xb) / xb
                - 2.0 * (np.sin(xb) / xb ** 2 - np.cos(xb) / xb) / xb)
    return out


def n1(x):
    """Irregular spherical Bessel n_1 (Neumann)."""
    x = np.asarray(x, dtype=complex)
    return -np.cos(x) / x ** 2 - np.sin(x) / x


def n1p(x):
    """d n_1 / dx."""
    x = np.asarray(x, dtype=complex)
    # n_1'(x) = n_0(x) - 2 n_1(x)/x, n_0(x) = -cos x / x
    return (-np.cos(x) / x
            - 2.0 * (-np.cos(x) / x ** 2 - np.sin(x) / x) / x)


def tan_delta1(k, V0=V0, a=A):
    """Analytic l=1 phase shift for attractive square well of depth V0, radius a.

    Matching log-derivative of u_1 = k r j_1(k r) inside (with k -> kappa)
    to a + b*tan delta combination outside gives
        tan delta_1 = [kappa j_1'(kappa a) j_1(k a) - k j_1(kappa a) j_1'(k a)]
                    / [kappa j_1'(kappa a) n_1(k a) - k j_1(kappa a) n_1'(k a)].
    Valid for both real and complex k (analytic continuation).
    """
    k     = np.asarray(k, dtype=complex)
    kappa = np.sqrt(k ** 2 + V0)
    num =  kappa * j1p(kappa * a) * j1(k * a) - k * j1(kappa * a) * j1p(k * a)
    den =  kappa * j1p(kappa * a) * n1(k * a) - k * j1(kappa * a) * n1p(k * a)
    return num / den


def delta1_real(k):
    """Real-axis phase shift, unwrapped onto the rising-through-pi/2 branch.

    arctan returns delta in (-pi/2, pi/2); near a resonance we want the
    physically continuous delta(E) that rises by ~pi.  We add pi to every
    point past the first crossing where tan(delta) jumps from + infinity to
    - infinity (i.e. cot delta passes through 0 with negative slope)."""
    td = np.real(tan_delta1(k))
    d  = np.arctan(td)
    # detect where tan delta jumps from large + to large - (rising through pi/2)
    cd = 1.0 / td
    bumped = np.zeros_like(d)
    n_bump = 0
    for i in range(1, len(d)):
        if cd[i - 1] > 0 and cd[i] < 0 and (cd[i - 1] - cd[i]) > 0:
            n_bump += 1
        bumped[i] = n_bump * np.pi
    return d + bumped


def S1(k):
    """S-matrix element for l=1: S_1 = (1 + i tan delta)/(1 - i tan delta)."""
    td = tan_delta1(k)
    return (1.0 + 1j * td) / (1.0 - 1j * td)


# ---------- complex pole search: zero of denominator (1 - i tan delta) -------
def cot_delta1(k):
    """cot delta_1 (= 1/tan delta_1); pole of S occurs at cot delta = i."""
    return 1.0 / tan_delta1(k)


def _k_second_sheet(E):
    """Pick k(E) on the resonance sheet (Re k > 0, Im k < 0)."""
    k = np.sqrt(complex(E))
    if k.imag > 0:
        k = -k
    if k.real < 0:
        k = -k
    return k


def find_pole_E(E_guess):
    """Newton iteration in complex E plane for S_1-pole: cot delta_1(k) = i,
    on the second sheet (Im k < 0).  Equivalently: zero of (1 - i tan delta).
    """
    E = complex(E_guess)
    for _ in range(120):
        k  = _k_second_sheet(E)
        f  = cot_delta1(np.array([k]))[0] - 1j
        dE = 1e-7 * (abs(E) + 1.0)
        kp = _k_second_sheet(E + dE)
        fp = cot_delta1(np.array([kp]))[0] - 1j
        df = (fp - f) / dE
        if abs(df) < 1e-30:
            break
        step = f / df
        # damp very large steps
        if abs(step) > 0.5 * (abs(E) + 1.0):
            step *= 0.3 * (abs(E) + 1.0) / abs(step)
        E -= step
        if abs(step) < 1e-13:
            break
    return E


# ---------- Breit-Wigner cross section + fit ---------------------------------
def sigma1(k):
    """Partial-wave cross section sigma_1(k) = (4 pi / k^2) (2 l + 1) sin^2 delta."""
    d = delta1_real(k)
    return (4.0 * np.pi / k ** 2) * 3.0 * np.sin(d) ** 2


def bw_sigma(E, ER, Gamma, kfac):
    """Breit-Wigner with overall on-shell prefactor (12 pi / k^2)
    where k = sqrt(E); we absorb that as 12 pi / E (since hbar = 2m = 1 means
    E = k^2). The factor kfac is a tiny non-resonant background scaler."""
    k2 = E
    return (12.0 * np.pi / k2) * (Gamma / 2) ** 2 / ((E - ER) ** 2 + (Gamma / 2) ** 2) * kfac


# ---------- Wigner time delay -----------------------------------------------
def time_delay(E_arr):
    """tau(E) = 2 hbar d delta_1 / d E.  hbar = 1 in our units."""
    k = np.sqrt(E_arr)
    d = delta1_real(k)
    tau = 2.0 * np.gradient(d, E_arr)
    return tau


# ---------- radial Hamiltonian + survival probability -----------------------
def build_H_radial(R_box=40.0, N=4000, V0=V0, ell=1):
    """Discretise H_l = -d^2/dr^2 + l(l+1)/r^2 + V(r) on [dr, R_box - dr]
    with Dirichlet BC. Returns (r, H) in units hbar = 2m = 1."""
    r  = np.linspace(0.0, R_box, N + 2)[1:-1]
    dr = r[1] - r[0]
    # finite difference Laplacian
    main = np.full(N, 2.0 / dr ** 2)
    off  = np.full(N - 1, -1.0 / dr ** 2)
    # potential + centrifugal
    Vc = ell * (ell + 1) / r ** 2
    Vw = np.where(r < A, -V0, 0.0)
    diag = main + Vc + Vw
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return r, dr, H


def quasi_stationary_state(r, dr, ER, sigma_r=2.0, r0=0.5):
    """Build a 'quasi-stationary' state: a Gaussian inside the well x sin(k_R r)
    with k_R = sqrt(E_R). This mimics the Gamow-state shape inside r < a."""
    kR = np.sqrt(ER)
    psi = np.exp(-(r - r0) ** 2 / (2 * sigma_r ** 2)) * np.sin(kR * r)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dr)
    return psi


def survival_curve(t_arr, evals, evecs, psi0_coeffs):
    """Return P(t) = |<psi0|psi(t)>|^2 from spectral expansion."""
    P = np.empty_like(t_arr)
    for i, t in enumerate(t_arr):
        amp = np.sum(np.abs(psi0_coeffs) ** 2 * np.exp(-1j * evals * t))
        P[i] = np.abs(amp) ** 2
    return P


# ---------- main -------------------------------------------------------------
def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.normpath(os.path.join(here, "..", "figures"))
    os.makedirs(fig_dir, exist_ok=True)

    # --------- (a) phase shift over a window covering the resonance
    k_arr = np.linspace(0.05, 2.5, 2000)
    E_arr = k_arr ** 2
    delta = delta1_real(k_arr)

    # --------- (b) cross section
    sig   = sigma1(k_arr)

    # --------- locate resonance roughly via maximum of sigma
    # mask the trivial rise near k -> 0 where 4 pi / k^2 blows up
    mask_search = E_arr > 0.05
    i_max = int(np.argmax(sig * mask_search))
    ER_guess = E_arr[i_max]

    # --------- (c) Breit-Wigner fit on a tight window around the peak
    win = (E_arr > 0.5 * ER_guess) & (E_arr < 2.0 * ER_guess)
    try:
        popt, _ = curve_fit(bw_sigma, E_arr[win], sig[win],
                            p0=[ER_guess, 0.3, 1.0], maxfev=40000)
        ER_fit, Gamma_fit, kfac = popt
    except Exception as exc:
        print(f"[ch14] BW fit failed: {exc}; using rough estimate")
        ER_fit, Gamma_fit, kfac = ER_guess, 0.3, 1.0
    # plot BW only inside the fit window
    sig_bw = bw_sigma(E_arr, ER_fit, Gamma_fit, kfac)

    # --------- (e) complex pole
    E_pole = find_pole_E(ER_fit - 0.5j * Gamma_fit)
    print(f"[ch14] BW fit: E_R = {ER_fit:.4f}, Gamma = {Gamma_fit:.4f}")
    print(f"[ch14] complex pole: E* = {E_pole.real:+.4f} {E_pole.imag:+.4f}j")
    print(f"[ch14] from pole: E_R = {E_pole.real:.4f}, "
          f"Gamma = {-2 * E_pole.imag:.4f}")

    # --------- (d) time delay
    tau = time_delay(E_arr)

    # --------- (f) survival probability via radial Hamiltonian
    # The Khalfin (long-time, algebraic) tail kicks in only at amplitudes
    # ~ 10^(-Q/pi) below the exponential, after the exponential decay has
    # eaten its dominant share.  In a finite box we always eventually
    # encounter recurrences set by t ~ R_box / v_R; we therefore use a
    # large box and stop before the recurrences contaminate the signal.
    print("[ch14] building radial Hamiltonian (this is the slow step) ...")
    R_box = 200.0
    N_box = 3000
    r, dr, H = build_H_radial(R_box=R_box, N=N_box)
    evals, evecs = np.linalg.eigh(H)
    psi0   = quasi_stationary_state(r, dr, ER_fit, sigma_r=0.30, r0=0.55)
    coeffs = evecs.T @ psi0
    # cap tmax before box recurrences:  t_recur ~ R_box / (group vel)
    v_R   = max(2.0 * np.sqrt(ER_fit), 0.5)   # group vel ~ 2 k = 2 sqrt(E)
    t_recur = 0.5 * R_box / v_R                # safety factor 1/2
    tmax  = min(40.0 / Gamma_fit, t_recur)
    t_arr = np.geomspace(0.05, tmax, 500)
    P_t   = survival_curve(t_arr, evals, evecs, coeffs)
    P_t  /= P_t[0]

    # --------- plotting ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))

    ax = axes[0, 0]
    # zoom on the rising-through-pi/2 region
    Emax_panel = min(4.0 * ER_fit, E_arr.max())
    sel = E_arr < Emax_panel
    ax.plot(E_arr[sel], delta[sel], lw=1.8, color="#1f77b4",
            label=r"$\delta_1(E)$")
    ax.axhline(np.pi / 2, color="grey", ls=":", lw=0.9, label=r"$\pi/2$")
    ax.axvline(ER_fit, color="#d62728", ls="--", lw=0.9,
               label=fr"$E_R = {ER_fit:.3f}$")
    ax.set_xlabel(r"$E = k^2$")
    ax.set_ylabel(r"$\delta_1$")
    ax.set_title(r"(a) p-wave phase shift $\delta_1(E)$")
    ax.set_xlim(0, Emax_panel)
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[0, 1]
    Emin_b = max(ER_fit - 4.0 * Gamma_fit, 0.05)
    Emax_b = ER_fit + 4.0 * Gamma_fit
    sel_b = (E_arr >= Emin_b) & (E_arr <= Emax_b)
    ax.plot(E_arr[sel_b], sig[sel_b], lw=1.8, color="#1f77b4",
            label=r"exact $\sigma_1$")
    ax.plot(E_arr[sel_b], sig_bw[sel_b], lw=1.0, ls="--", color="#d62728",
            label=fr"BW fit ($\Gamma={Gamma_fit:.3f}$)")
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\sigma_1$")
    ax.set_title("(b) Cross section + Breit-Wigner fit")
    ax.set_xlim(Emin_b, Emax_b)
    ax.set_ylim(0, 1.2 * sig[sel_b].max())
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1, 0]
    Emin_c = max(ER_fit - 4.0 * Gamma_fit, 0.05)
    Emax_c = ER_fit + 4.0 * Gamma_fit
    sel_c = (E_arr >= Emin_c) & (E_arr <= Emax_c)
    ax.plot(E_arr[sel_c], tau[sel_c], lw=1.8, color="#2ca02c",
            label=r"$\tau(E)$")
    ax.axhline(4.0 / Gamma_fit, color="grey", ls=":", lw=0.9,
               label=fr"$4/\Gamma = {4/Gamma_fit:.2f}$")
    ax.axvline(ER_fit, color="#d62728", ls="--", lw=0.9)
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\tau = 2\,d\delta_1/dE$")
    ax.set_title("(c) Wigner time delay")
    ax.set_xlim(Emin_c, Emax_c)
    ax.set_ylim(0, max(1.3 * tau[sel_c].max(), 1.2 * 4.0 / Gamma_fit))
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1, 1]
    # only plot the well-resolved (above floor) portion
    floor = 1e-10
    keep = P_t > floor
    ax.loglog(t_arr[keep], P_t[keep], lw=1.6, color="#1f77b4",
              label=r"$P(t)$ numerical")
    ax.loglog(t_arr, np.exp(-Gamma_fit * t_arr),
              ls="--", lw=1.0, color="#d62728",
              label=fr"$e^{{-\Gamma t}}$, $\Gamma = {Gamma_fit:.3f}$")
    # guide line for algebraic tail ~ t^-(2l+3) = t^-5 (l = 1 Khalfin)
    t_late = np.geomspace(2.0 / Gamma_fit, t_arr[-1], 50)
    P_late_anchor = 0.3 * np.exp(-Gamma_fit * (2.0 / Gamma_fit))
    ax.loglog(t_late,
              P_late_anchor * (t_late / t_late[0]) ** -5,
              ls=":", lw=1.2, color="#2ca02c",
              label=r"$\sim t^{-(2\ell+3)}=t^{-5}$ guide")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$P(t)$")
    ax.set_title("(d) Survival probability: exponential regime")
    ax.set_ylim(floor, 2.0)
    ax.set_xlim(t_arr[0], t_arr[-1])
    ax.legend(loc="lower left")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(fig_dir, "ch14_resonance_breitwigner.pdf")
    fig.savefig(out)
    print(f"[ch14] wrote {out}")


if __name__ == "__main__":
    main()
