"""ch04_polarization.py
Polarization observables for spin-1/2 + spin-0 elastic scattering.

The amplitude is the 2x2 matrix
    F(theta, phi) = a(theta) * I + i*b(theta) * (n_hat . sigma),
with the scattering-plane normal  n_hat = (-sin phi, cos phi, 0).

Toy amplitudes (phi = 0, so n_hat = y_hat):
    a(theta) = cos(theta/2) * exp(-theta^2/4),
    b(theta) = 0.4 * sin(theta) * exp(-theta^2/4) * exp(i*0.6).
The constant phase 0.6 makes Im(a b*) nonzero so that
nu(theta) = 2 Im(a b*)/(|a|^2+|b|^2) is non-trivial.

Computes (a) unpolarized cross section, (b) outgoing pi for unpolarized
incident, (c) A_LR(theta) for pi_in along x_hat, y_hat, z_hat, and
(d) the analyzing power nu(theta) plus nu(60)*nu(60) for double scattering.

Output figure -> ../figures/ch04_polarization.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- toy amplitudes ----------
def a_amp(theta):
    return np.cos(theta / 2.0) * np.exp(-theta**2 / 4.0)

def b_amp(theta):
    # extra constant phase so a, b are not collinear in C  =>  Im(a b*) != 0
    return 0.4 * np.sin(theta) * np.exp(-theta**2 / 4.0) * np.exp(1j * 0.6)

# ---------- observables ----------
def unpolarized_xsec(theta):
    return np.abs(a_amp(theta))**2 + np.abs(b_amp(theta))**2

def analyzing_power(theta):
    """nu(theta) = 2 Im(a b*) / (|a|^2 + |b|^2)  in [-1, 1]."""
    a, b = a_amp(theta), b_amp(theta)
    return 2.0 * np.imag(a * np.conjugate(b)) / (np.abs(a)**2 + np.abs(b)**2)

def A_LR(theta, pi_in):
    """Single-scatter asymmetry contribution nu(theta) * (n_hat . pi_in)."""
    n_hat = np.array([0.0, 1.0, 0.0])    # phi = 0  =>  n_hat = y_hat
    return analyzing_power(theta) * np.dot(n_hat, pi_in)

# ---------- evaluation grid ----------
theta = np.linspace(1e-3, np.pi, 400)
sigma_unpol = unpolarized_xsec(theta)
nu          = analyzing_power(theta)
pi_out_mag  = np.abs(nu)                              # |pi_out| for unpolarized in
ALR_x = A_LR(theta, np.array([1.0, 0.0, 0.0]))        # transverse, in plane
ALR_y = A_LR(theta, np.array([0.0, 1.0, 0.0]))        # along normal n_hat
ALR_z = A_LR(theta, np.array([0.0, 0.0, 1.0]))        # longitudinal

# Double-scattering analyzing-power product: theta_1 = theta_2 = 60 deg
theta_star = np.deg2rad(60.0)
nu_star    = analyzing_power(theta_star)
double_AP  = nu_star * nu_star
print(f"nu(60 deg) = {nu_star:+.4f}")
print(f"A_LR (double scat, theta1=theta2=60 deg) = nu^2 = {double_AP:+.4f}")

# ---------- 2x2 figure ----------
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
deg = np.rad2deg(theta)

ax = axes[0, 0]                 # (A) unpolarized cross section
ax.plot(deg, sigma_unpol, lw=2, color="C0")
ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$|a|^2+|b|^2$")
ax.set_title(r"(A) Unpolarized $d\sigma/d\Omega$")
ax.grid(alpha=0.3)

ax = axes[0, 1]                 # (B) outgoing polarization magnitude
ax.plot(deg, pi_out_mag, lw=2, color="C3")
ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$|\pi_{\rm out}|$")
ax.set_title(r"(B) Mott-Schwinger $|\pi_{\rm out}|$ (unpolarized in)")
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

ax = axes[1, 0]                 # (C) A_LR for three input polarizations
ax.plot(deg, ALR_x, lw=2, label=r"$\pi_{\rm in}=\hat x$ (in plane)")
ax.plot(deg, ALR_y, lw=2, label=r"$\pi_{\rm in}=\hat y$ (along $\hat n$)")
ax.plot(deg, ALR_z, lw=2, ls="--", label=r"$\pi_{\rm in}=\hat z$ (longitudinal)")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$A_{LR}(\theta)$")
ax.set_title(r"(C) Single-scatter $\nu(\theta)\,\hat n\!\cdot\!\pi_{\rm in}$")
ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)

ax = axes[1, 1]                 # (D) analyzing power, with theta=60 deg marker
ax.plot(deg, nu, lw=2, color="C2")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(60.0, color="gray", ls=":", lw=1)
ax.plot(60.0, nu_star, "o", color="C2")
ax.text(62, nu_star,
        rf"$\nu(60^\circ)\!=\!{nu_star:+.3f}$" "\n"
        rf"$\nu(60^\circ)^2\!=\!{double_AP:+.3f}$",
        fontsize=9, va="center")
ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$\nu(\theta)$")
ax.set_title(r"(D) Analyzing power $\nu(\theta)$")
ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)

fig.suptitle(r"Polarization observables: spin-$1/2$ + spin-$0$ "
             r"with $F=a\,I+ib\,\hat n\cdot\sigma$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# ---------- save ----------
out_dir  = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ch04_polarization.pdf")
fig.savefig(out_path)
print(f"Saved figure -> {out_path}")
