"""ch10_numerov_phase.py
=================================================================
Numerov integration of the radial Schroedinger equation and
extraction of partial-wave phase shifts delta_l(k) for an
attractive square well

    V(r) = -V0   for r < a,
    V(r) =  0    for r >= a,

in units hbar = 2m = 1 (so E = k^2 and the radial equation reads
u_l''(r) + [k^2 - V(r) - l(l+1)/r^2] u_l(r) = 0,
with the convention V_eff = V here because 2m/hbar^2 = 1).

We integrate u_l(r) outwards from r = 0 with u_l(0) = 0 using the
Numerov 3-term recurrence (for u'' = -g u):

    u_{n+1}(1 + h^2 g_{n+1}/12) = 2 u_n (1 - 5 h^2 g_n/12)
                                  - u_{n-1}(1 + h^2 g_{n-1}/12),

where  g(r) = k^2 - V(r) - l(l+1)/r^2.

In the asymptotic free region (r > a) we match to
    u_l(r) = A * k r j_l(k r) - B * k r y_l(k r),
i.e. tan(delta_l) = B/A, by sampling u_l at two outer points.

Output: ../figures/ch10_numerov_phase.pdf  (4-panel figure).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# ---------- physics / numerics parameters --------------------------------
A_WELL = 1.0          # well radius (units of length)
R_MAX  = 25.0         # outer integration radius
H      = 1.0e-3       # Numerov step size

# ---------- Numerov integrator ------------------------------------------
def numerov_phase_shift(k, V0, ell, h=H, r_max=R_MAX, a=A_WELL):
    """Integrate u_l'' + g(r) u_l = 0 from r = 0 outward and extract delta_l.

    Returns the principal-value phase shift in (-pi/2, pi/2] via
    arctan2(B, A).  For the Levinson-counted absolute phase shift one
    additionally sweeps k and unwraps the resulting curve.
    """
    N = int(round(r_max / h))
    r = np.linspace(0.0, N * h, N + 1)

    # g(r) = k^2 - V(r) - l(l+1)/r^2  (the coefficient in u'' = -g u)
    V_of_r  = np.where(r < a, -V0, 0.0)
    centrif = np.zeros_like(r)
    centrif[r > 0] = ell * (ell + 1) / r[r > 0] ** 2
    g = k * k - V_of_r - centrif
    g[0] = 0.0   # r=0 row unused (u(0)=0)

    u = np.zeros_like(r)
    u[1] = h ** (ell + 1)        # small-r:  u_l(r) ~ r^{l+1}; norm is irrelevant

    f = 1.0 + h * h * g / 12.0   # (1 + h^2 g/12) factor

    # Numerov:  u_{n+1} f_{n+1} = 2 u_n (1 - 5 h^2 g_n/12) - u_{n-1} f_{n-1}
    for n in range(1, N):
        u[n + 1] = (2.0 * u[n] * (1.0 - 5.0 * h * h * g[n] / 12.0)
                    - u[n - 1] * f[n - 1]) / f[n + 1]

    # ---- match in the free region r > a to A k r j_l(kr) - B k r y_l(kr) ----
    # Pick two outer points well separated (different phases).
    r1 = 0.70 * r_max
    r2 = 0.95 * r_max
    i1 = int(round(r1 / h))
    i2 = int(round(r2 / h))
    R1, R2 = r[i1], r[i2]
    u1, u2 = u[i1], u[i2]

    j1 = R1 * spherical_jn(ell, k * R1)
    y1 = R1 * spherical_yn(ell, k * R1)
    j2 = R2 * spherical_jn(ell, k * R2)
    y2 = R2 * spherical_yn(ell, k * R2)
    # Solve [[j1, -y1], [j2, -y2]] @ [A, B]^T = [u1, u2]^T  (k absorbed into A,B)
    M = np.array([[j1, -y1], [j2, -y2]])
    rhs = np.array([u1, u2])
    A_coef, B_coef = np.linalg.solve(M, rhs)
    delta = np.arctan2(B_coef, A_coef)
    return delta


# ---------- analytic square-well phase shift (l = 0) ---------------------
def delta0_analytic(k, V0, a=A_WELL):
    """l=0 attractive square-well: tan(k a + delta_0) = (k/K) tan(K a),
    K = sqrt(k^2 + V0)."""
    K = np.sqrt(k * k + V0)
    val = (k / K) * np.tan(K * a)
    return np.arctan(val) - k * a   # principal value, mod pi


# ---------- run experiments ---------------------------------------------
def main():
    V0_list = [1.0, 4.0, 9.0, 16.0]
    k_grid  = np.linspace(0.05, 6.0, 120)

    # (1) delta_0(k) for the four well depths
    delta0_curves = {}
    for V0 in V0_list:
        d0 = np.array([numerov_phase_shift(k, V0, 0) for k in k_grid])
        delta0_curves[V0] = np.unwrap(d0)

    # (2) scattering length a_0(V0) = -lim_{k->0} delta_0(k)/k  (numerical)
    V0_scan = np.linspace(0.05, 25.0, 320)
    k_small = 0.02
    a0_scan = []
    for V0 in V0_scan:
        d = numerov_phase_shift(k_small, V0, 0)
        # principal value lies in (-pi/2, pi/2]; this *is* delta_0 mod pi,
        # which is precisely what enters tan(delta_0) used to define a_0.
        a0_scan.append(-np.tan(d) / k_small)
    a0_scan = np.array(a0_scan)

    # (3) sigma_l(k) for V0 = 4 and l = 0,1
    V0_partial = 4.0
    k_p = np.linspace(0.05, 4.0, 100)
    sig0 = np.zeros_like(k_p)
    sig1 = np.zeros_like(k_p)
    for i, k in enumerate(k_p):
        d0 = numerov_phase_shift(k, V0_partial, 0)
        d1 = numerov_phase_shift(k, V0_partial, 1)
        sig0[i] = (4.0 * np.pi / k**2) * (2 * 0 + 1) * np.sin(d0) ** 2
        sig1[i] = (4.0 * np.pi / k**2) * (2 * 1 + 1) * np.sin(d1) ** 2

    # (4) numerical vs. analytic delta_0 for V0 = 4
    k_v = np.linspace(0.1, 5.0, 60)
    d_num = np.array([numerov_phase_shift(k, 4.0, 0) for k in k_v])
    d_an  = delta0_analytic(k_v, 4.0)
    # bring both to same branch via arctan(tan(.)) so they coincide on (-pi/2, pi/2]
    d_num_pv = np.arctan(np.tan(d_num))
    d_an_pv  = np.arctan(np.tan(d_an))

    # ---------- plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))

    # (1) delta_0(k) for several V0
    ax = axes[0, 0]
    for V0 in V0_list:
        ax.plot(k_grid, delta0_curves[V0], lw=1.8, label=rf"$V_0={V0:g}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$k$  (units $1/a$)")
    ax.set_ylabel(r"$\delta_0(k)$  [rad, unwrapped]")
    ax.set_title(r"(A) S-wave phase shift, attractive square well")
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3)

    # (2) scattering length vs V0
    ax = axes[0, 1]
    ax.plot(V0_scan, a0_scan, lw=1.8, color="C3")
    for n in (0, 1, 2):
        Vc = ((2 * n + 1) * np.pi / 2.0) ** 2
        if Vc <= V0_scan[-1]:
            ax.axvline(Vc, color="gray", ls=":", lw=1)
            ax.text(Vc + 0.1, 13.5,
                    rf"$V_c=({2*n+1}\pi/2)^2$",
                    rotation=90, va="top", ha="left",
                    fontsize=8, color="gray", clip_on=True)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$V_0$ (units $1/a^2$)")
    ax.set_ylabel(r"$a_0 = -\tan\delta_0(k_{\rm small})/k_{\rm small}$")
    ax.set_title(r"(B) Scattering length: divergence at zero-energy bound states")
    ax.set_xlim(V0_scan[0], V0_scan[-1]); ax.set_ylim(-15, 15); ax.grid(alpha=0.3)

    # (3) partial-wave cross sections, V0 = 4
    ax = axes[1, 0]
    ax.plot(k_p, sig0, lw=1.8, label=r"$\sigma_0=(4\pi/k^2)\sin^2\delta_0$")
    ax.plot(k_p, sig1, lw=1.8, label=r"$\sigma_1=(12\pi/k^2)\sin^2\delta_1$")
    ax.set_xlabel(r"$k$  (units $1/a$)")
    ax.set_ylabel(r"$\sigma_\ell(k)$")
    ax.set_title(rf"(C) Partial-wave cross sections, $V_0={V0_partial:g}$")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1e3)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (4) numeric vs analytic delta_0 (V0 = 4)
    ax = axes[1, 1]
    ax.plot(k_v, d_num_pv, "o", ms=4, label="Numerov (numerical)", color="C0")
    ax.plot(k_v, d_an_pv, "-", lw=1.5, label="analytic square-well", color="C1")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$k$  (units $1/a$)")
    ax.set_ylabel(r"$\delta_0(k)$ mod $\pi$  (principal value)")
    ax.set_title(rf"(D) Verification at $V_0={4.0:g}$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(r"Numerov phase shifts: attractive square well "
                 r"$V(r)=-V_0\,\theta(a-r)$, units $\hbar=2m=a=1$",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ch10_numerov_phase.pdf")
    fig.savefig(out_path)
    print(f"Saved figure -> {out_path}")

    # quick numerical sanity check vs analytic at one point
    k_test = 1.0
    d_n = numerov_phase_shift(k_test, 4.0, 0)
    d_a = delta0_analytic(k_test, 4.0)
    print(f"k=1, V0=4 :  delta0 num = {d_n:+.5f},  analytic = {d_a:+.5f}")


if __name__ == "__main__":
    main()
