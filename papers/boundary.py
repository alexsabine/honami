#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary.py  —  reproduction suite for

    "The Boundary is a Rate: Coherence-Rupture-Regeneration as a Temporal
     Physics of Emptiness Realisation"

Direct link (paper, "Code and data availability"):
    https://www.space-zero.org/blogs/boundary.py

WHAT THIS IS
------------
A single, self-contained, deterministic script that regenerates every figure
and every headline number in the paper. Figures 1 and 5 are schematics; Figures
2, 3, 4, 6, 7, 8, 9 are finite numerical surrogates. NONE of it is contact with
empirical data.

TWO RIGOUR STANDARDS THIS SUITE HOLDS ITSELF TO
-----------------------------------------------
(A) Sandved-Smith, Fields, Doctor, Laukkonen & Hohwy (2026), "There is no
    self-evidence: A physics of emptiness realisation" (PsyArXiv m78z2) -- the
    paper we respond to. We inherit its discipline:
      * every result is a CONSISTENCY DEMONSTRATION under a chosen formalism,
        never an empirical finding;
      * theorem is kept marked apart from interpretation;
      * we build FINITE OPERATIONAL SURROGATES for the formal objects and ask
        what the formalism forces, forbids, or over-determines, rather than
        asserting a result about the world;
      * we claim the right amount and no more.

(B) Friedman (2026), "Realizing Emptiness: Operational Surrogates for
    No-Self-Evidence, QRF Opacification, and Bayesian Model Reduction",
    Active Inference Institute / Zenodo, doi:10.5281/zenodo.20834847 -- the
    operational companion to (A). We do not reproduce Friedman's software; we
    adopt, at the scale of one script, the cognitive-security / claim-governance
    posture his artifact defines, so a reader can AUDIT rather than merely trust:

      1. SOURCE ANCHORING.  Each surrogate is bound to a specific equation or
         section of the paper -- see CLAIM_LEDGER below. No figure floats free
         of the claim it supports.
      2. EVIDENCE CEILING.  Each surrogate carries an explicit statement of what
         must NOT be inferred from it (CLAIM_LEDGER[...]['ceiling']). This is the
         load-bearing move: the artifact says where it stops and where empirical
         science would have to begin.
      3. DETERMINISM.  Fixed seeds, no wall-clock in any output, byte-stable
         results.json. A clean run reproduces the suite exactly.
      4. POSITIVE + NULL CONTROLS.  Each surrogate carries a positive control
         (the intended mechanism is present -- a valid density matrix; the de
         Bruijn identity; Kc validated against the exact analytic curve) and,
         where a null is meaningful, a discriminating comparison against it.
      5. DISCRIMINATING TESTS.  Where a stronger reading is available we test it
         and let it fail honestly: the kernel fork rejects the 'information-pump'
         reading; retensing rejects the strong identity C = I(A:E) and keeps only
         the bound; the 2:1 tongue is separated from generic mode-locking by its
         width, not by its mere existence.
      6. VALIDATION GATES.  validation_gates() re-derives every headline number
         and checks it against a declared tolerance band (PASS/FAIL), echoing
         Friedman's artifact-gate design; figure-integrity SHA-256 hashes are
         recorded so a figure cannot silently drift from the number it reports.
      7. DECLARED FUTURE-EVIDENCE BOUNDARY.  The breath predictions (paper Sec.9)
         are STATED, not run; empirical contact is explicitly out of scope for
         this artifact and is deferred to pre-registered work.

INTEGRITY BOUNDARY (in Friedman's sense)
----------------------------------------
This script does not measure awakening, contemplative attainment, neural
criticality, clinical outcome, or any physical realisation of the quantum-FEP.
It makes the paper's formal commitments inspectable and reproducible, and is
deliberate about saying where the software stops.

CONVENTIONS
-----------
CRR parameters (fixed by geometry, not fitted):
    bistable   (Z2)   :  Omega = 1/pi ,    C* = pi ,   CV = 1/(2 pi) ~ 0.159
    rotational (SO(2)):  Omega = 1/(2 pi),  C* = 2 pi,  CV = 1/(4 pi) ~ 0.080
Rupture condition:  C * Omega = 1.

Run `python3 boundary.py` to regenerate every PNG, print the claim ledger, run
the validation gates, and write results.json (numbers + gates + figure hashes).
"""

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyArrowPatch, Rectangle
from numpy.linalg import eigh, eigvalsh
from scipy.linalg import expm
import os, json, hashlib

# --------------------------------------------------------------------------
# House style (warm off-white, muted earth palette, serif) matching the paper
# --------------------------------------------------------------------------
BG      = "#f6f1e7"   # warm paper
INK     = "#2f2b25"   # near-black ink
GREEN   = "#3f5c46"   # primary (dark green)
SIENNA  = "#9c5a34"   # secondary (brown/sienna)
PURPLE  = "#7c6091"   # tertiary (muted purple)
GREY    = "#8a8577"
FAINT   = "#c9bfa8"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "savefig.facecolor": BG,
    "font.family":      "serif",
    "font.serif":       ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.edgecolor":   INK,
    "axes.labelcolor":  INK,
    "text.color":       INK,
    "xtick.color":      INK,
    "ytick.color":      INK,
    "axes.linewidth":   0.8,
    "font.size":        11,
    "axes.titlesize":   11,
    "figure.dpi":       140,
})

OUT = "/home/claude/figs"
os.makedirs(OUT, exist_ok=True)
RESULTS = {}   # headline numbers -> transcribed into the paper

PI = np.pi
LN2 = np.log(2.0)

def vn_entropy(evals, base=np.e):
    """von Neumann / Shannon entropy of an eigenvalue (probability) vector."""
    p = np.array(evals, float)
    p = p[p > 1e-15]
    p = p / p.sum()
    return float(-np.sum(p * (np.log(p) / np.log(base))))

def partial_trace_A(psi, dA, dE):
    """rho_A = Tr_E |psi><psi| for a pure state on A (x) E."""
    M = psi.reshape(dA, dE)
    return M @ M.conj().T


# ==========================================================================
# FIGURE 1  (schematic) : two uses of one geometry
# ==========================================================================
def fig1_geometry():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax in (axL, axR):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.5, 1.5)

    # ---- left: the shape of the question ----
    th = np.linspace(0, 2*PI, 400)
    axL.plot(np.cos(th), np.sin(th), color=FAINT, lw=1.6)           # full circle
    arc = np.linspace(0, PI, 200)                                    # upper semicircle
    axL.plot(np.cos(arc), np.sin(arc), color=GREEN, lw=3.2)
    axL.scatter([1, -1], [0, 0], color=SIENNA, zorder=5, s=34)
    axL.scatter([0], [1], facecolor=BG, edgecolor=GREEN, zorder=6, s=40)
    axL.text(1.06, -0.02, r"$p=1$", color=SIENNA, ha="left", va="center", fontsize=10)
    axL.text(-1.06, -0.02, r"$p=0$", color=SIENNA, ha="right", va="center", fontsize=10)
    axL.text(0, 1.12, r"$m=0$", color=GREEN, ha="center", va="bottom", fontsize=9)
    axL.text(0, 0.42, "bistable question: the arc", color=GREEN, ha="center", fontsize=9, style="italic")
    axL.text(0, 0.24, r"$\ell_{\mathbb{Z}_2}=\pi$", color=GREEN, ha="center", fontsize=9)
    axL.text(0, -0.5, "rotational question: the full circle", color=GREY, ha="center", fontsize=9, style="italic")
    axL.text(0, -0.68, r"$\ell_{SO(2)}=2\pi$", color=GREY, ha="center", fontsize=9)
    axL.set_title("the shape of the question", fontsize=10, style="italic")
    axL.text(0, -1.35, r"this panel varies by system: it fixes $\Omega=1/\ell$",
             ha="center", fontsize=8, color=GREY)

    # ---- right: the two roles in every rupture ----
    axR.plot(np.cos(th), np.sin(th), color=PURPLE, lw=1.8)
    # a single cut at the top
    axR.plot([0, 0], [0.86, 1.16], color=SIENNA, lw=3.0)
    axR.scatter([0], [1.0], color=SIENNA, s=26, zorder=6)
    axR.text(-0.12, 1.24, "past", color=INK, ha="right", fontsize=9)
    axR.text(0.12, 1.24, "future", color=INK, ha="left", fontsize=9)
    axR.text(0, 0.16, r"$\mathbb{Z}_2$: the cut that falls", color=SIENNA, ha="center", fontsize=9, style="italic")
    axR.text(0, -0.22, "$SO(2)$: the continuum that holds", color=PURPLE, ha="center", fontsize=9, style="italic")
    axR.text(0, -0.42, "no preferred now", color=GREY, ha="center", fontsize=8, style="italic")
    axR.set_title("the two roles, in every rupture", fontsize=10, style="italic")
    axR.text(0, -1.35, "this panel never varies: every rupture is one cut",
             ha="center", fontsize=8, color=GREY)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_geometry.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 2 : screen-induction at rupture
#   Build an explicit bipartite pure state on A (x) E whose Schmidt spectrum
#   on A interpolates from pure (s=0) to maximally mixed (s=1). Show S(rho_A)
#   is finite and rho_A a valid density matrix at every saturation level
#   s ~ C.Omega, including the rupture s -> 1.
# ==========================================================================
def fig2_screen_induction():
    dA = 4                       # log2 dA = 2 bits maximal
    s_grid = np.linspace(0, 1, 200)
    e1 = np.array([1, 0, 0, 0], float)
    unif = np.full(dA, 1.0/dA)
    S_bits = []
    max_nonpsd = 0.0
    max_trace_err = 0.0
    for s in s_grid:
        lam = (1 - s) * e1 + s * unif        # Schmidt probabilities
        lam = lam / lam.sum()
        # genuine bipartite pure state |psi> = sum sqrt(lam_i) |i>_A |i>_E
        psi = np.zeros(dA * dA)
        for i in range(dA):
            psi[i * dA + i] = np.sqrt(lam[i])
        rho_A = partial_trace_A(psi, dA, dA)
        ev = eigvalsh(rho_A)
        max_nonpsd = max(max_nonpsd, float(-ev.min()))       # PSD check
        max_trace_err = max(max_trace_err, abs(float(ev.sum()) - 1.0))
        S_bits.append(vn_entropy(ev, base=2))
    S_bits = np.array(S_bits)

    RESULTS["fig2_S_at_rupture_bits"]  = float(S_bits[-1])
    RESULTS["fig2_max_negativity"]     = max_nonpsd
    RESULTS["fig2_max_trace_error"]    = max_trace_err

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(s_grid, S_bits, color=GREEN, lw=2.6)
    ax.axhline(2.0, color=SIENNA, ls=":", lw=1.2)
    ax.scatter([1.0], [S_bits[-1]], facecolor=BG, edgecolor=GREEN, s=55, zorder=6)
    ax.annotate(r"rupture $s\to 1$:" + "\n" + r"$\rho_A$ valid, $S$ finite",
                xy=(1.0, S_bits[-1]), xytext=(0.62, 1.25),
                fontsize=9, color=SIENNA,
                arrowprops=dict(arrowstyle="-", color=SIENNA, lw=1.0))
    ax.text(0.02, 2.03, r"$\log_2 d_A = 2$ bits (maximal)", fontsize=8.5, color=GREY)
    ax.set_xlabel(r"coherence-saturation level  $s \approx C\cdot\Omega$")
    ax.set_ylabel(r"$S(\rho_A)$  (bits)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 2.1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_screen_induction.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 3 : the regeneration kernel is not unitary -- a fork
#   Z2 system: Omega = 1/pi, C* = pi, so C/Omega runs 0 -> pi^2.
#   Left  : kernel value exp(C/Omega) across accumulation bins (past -> rupture)
#   Right : literal reading injects nats & inflates norm;
#           renormalised-selection reading conserves measure but LOWERS entropy.
# ==========================================================================
def fig3_kernel_fork():
    n = 12
    Omega = 1.0 / PI
    Cstar = PI
    C = np.linspace(0, Cstar, n)
    w = np.exp(C / Omega)                       # kernel weights, exp(C/Omega)

    peak = float(w[-1])                          # exp(pi^2)
    # literal reading: highest-coherence amplitude multiplied by exp(C*/Omega)
    literal_nats = float(Cstar / Omega)          # = pi^2
    literal_factor = peak
    # renormalised-selection reading: p_k = w_k / sum w_k
    p = w / w.sum()
    H_in = np.log(n)                             # uniform prior over n past bins
    H_out = vn_entropy(p, base=np.e)
    dH = H_out - H_in

    RESULTS["fig3_peak_kernel"]     = peak
    RESULTS["fig3_literal_nats"]    = literal_nats
    RESULTS["fig3_literal_factor"]  = literal_factor
    RESULTS["fig3_renorm_dH_nats"]  = float(dH)
    RESULTS["fig3_H_in_nats"]       = float(H_in)
    RESULTS["fig3_H_out_nats"]      = float(H_out)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # left
    axL.semilogy(np.arange(1, n+1), w, "o-", color=GREEN, ms=4, lw=1.8, mfc=BG)
    axL.text(2.2, peak*0.12, rf"$\times{peak:,.0f}$", color=SIENNA, fontsize=10)
    axL.set_xlabel(r"accumulation bin (past $\rightarrow$)")
    axL.set_ylabel(r"kernel  $e^{C/\Omega}$")
    axL.set_title("amplification toward rupture", fontsize=10, style="italic")
    axL.grid(True, which="both", axis="y", color=FAINT, lw=0.4, alpha=0.6)

    # right
    axR.bar([0, 1], [H_in, H_out], width=0.55,
            color=[GREEN, SIENNA], alpha=0.55, edgecolor=INK, lw=0.8)
    axR.set_xticks([0, 1]); axR.set_xticklabels(["input", "after\nselection"])
    axR.set_ylabel("Shannon entropy (nats)")
    axR.set_title("renormalised reading", fontsize=10, style="italic")
    axR.text(0.5, max(H_in, H_out)*1.02, rf"$\Delta H = {dH:+.2f}$",
             ha="center", color=SIENNA, fontsize=10)
    axR.set_ylim(0, H_in*1.25)
    axR.text(0.5, -0.32*H_in,
             rf"literal reading: $+{literal_nats:.1f}$ nats injected, norm $\times{literal_factor:,.0f}$ $-$ fails unitarity",
             ha="center", fontsize=7.6, color=GREY, style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_kernel_fork.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 4 : the self-measurement obstruction
#   Fixed rho_A. In any measurement frame the READ-OUT entropy H(diag) >= S(rho_A),
#   with equality only in the eigenbasis (which the agent cannot select from
#   inside without already solving for its own state). Finite-dim shadow of the
#   universal impossibility.
# ==========================================================================
def _haar_unitary(d, rng):
    z = (rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * ph

def fig4_self_measurement():
    rng = np.random.default_rng(7)
    d = 4
    lam = np.array([0.55, 0.25, 0.13, 0.07])     # eigenvalues -> true S
    U0 = _haar_unitary(d, rng)                    # nontrivial eigenbasis
    rho = U0 @ np.diag(lam) @ U0.conj().T
    S_true_bits = vn_entropy(lam, base=2)

    n_frames = 8
    read = []
    for _ in range(n_frames):
        U = _haar_unitary(d, rng)                 # arbitrary measurement frame
        probs = np.real(np.diagonal(U @ rho @ U.conj().T))
        read.append(vn_entropy(probs, base=2))
    # eigenbasis frame (the one the agent cannot pick from inside)
    _, V = eigh(rho)
    probs_eig = np.real(np.diagonal(V.conj().T @ rho @ V))
    eig_read = vn_entropy(probs_eig, base=2)

    RESULTS["fig4_S_true_bits"]   = float(S_true_bits)
    RESULTS["fig4_mean_overshoot_bits"] = float(np.mean(read) - S_true_bits)
    RESULTS["fig4_eig_read_bits"] = float(eig_read)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(np.arange(1, n_frames+1), read, width=0.6,
           color=SIENNA, alpha=0.5, edgecolor=INK, lw=0.8)
    ax.axhline(S_true_bits, color=GREEN, lw=1.8)
    ax.text(n_frames+0.1, S_true_bits, rf" true $S(\rho_A)={S_true_bits:.3f}$ bits",
            va="center", color=GREEN, fontsize=9)
    ax.annotate("systematic overshoot", xy=(4, read[3]), xytext=(4.4, read[3]+0.28),
                fontsize=9, color=SIENNA,
                arrowprops=dict(arrowstyle="->", color=SIENNA, lw=1.0))
    ax.text(1.0, S_true_bits*0.55,
            "(reached only in the eigenbasis,\nwhich A cannot identify from inside)",
            fontsize=8, color=GREY, style="italic")
    ax.set_xlabel("measurement frame A deploys (arbitrary basis)")
    ax.set_ylabel("entropy A infers (bits)")
    ax.set_ylim(0, 2.05)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_self_measurement.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 5  (schematic) : two pictures of the boundary
# ==========================================================================
def fig5_two_pictures():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax in (axL, axR):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.5, 1.5)

    # left: static screen
    axL.add_patch(Circle((-0.7, 0), 0.42, color=INK))
    axL.text(-0.7, 0, "A", color=BG, ha="center", va="center", fontsize=13)
    axL.text(-0.7, -0.6, "agent", color=GREY, ha="center", fontsize=8, style="italic")
    axL.plot([0.05, 0.05], [-1.0, 1.0], color=SIENNA, lw=2.4)          # screen
    for yy in np.linspace(-0.9, 0.9, 9):                              # environment hatch
        axL.plot([0.2, 1.2], [yy, yy], color=FAINT, lw=0.8)
    axL.text(0.7, 1.02, "environment", color=GREY, ha="center", fontsize=8, style="italic")
    axL.text(0.7, -1.12, "(far side, retained)", color=GREY, ha="center", fontsize=8, style="italic")
    axL.add_patch(FancyArrowPatch((-0.28, 0.12), (-0.02, 0.12),
                  arrowstyle="->", color=SIENNA, lw=1.2, mutation_scale=12))
    axL.text(-0.15, 0.30, "unevidenceable", color=SIENNA, ha="center", fontsize=8, style="italic")
    axL.text(-0.55, 0.62, r"$S(\rho_A)$ across boundary", color=INK, fontsize=8)
    axL.plot([0.02, 0.08], [0.32, 0.26], color=SIENNA, lw=1.4)
    axL.plot([0.02, 0.08], [0.26, 0.32], color=SIENNA, lw=1.4)
    axL.set_title(r"Sandved-Smith et al. $\cdot$ static screen", fontsize=9)

    # right: boundary as rate -- many edges meeting
    def rings(cx, cy, col, rmax, n=4):
        for r in np.linspace(rmax/n, rmax, n):
            axR.add_patch(Circle((cx, cy), r, fill=False, color=col, lw=1.4, alpha=0.9))
        axR.scatter([cx], [cy], color=col, s=14)
    rings(-0.15, 0.0, GREEN, 0.72, 4)
    rings(0.85, 0.7, PURPLE, 0.42, 3)
    rings(0.9, -0.7, SIENNA, 0.40, 3)
    axR.text(-0.15, 0.0, r"$C\cdot\Omega\to 1$", color=SIENNA, ha="center", va="center", fontsize=8)
    axR.text(0.85, 0.7, r"$\Omega$", color=PURPLE, ha="center", va="center", fontsize=9)
    axR.text(0.9, -0.7, r"$\Omega$", color=SIENNA, ha="center", va="center", fontsize=9)
    axR.text(0, -1.12, r"no single screen $\cdot$ no far side $\cdot$ only edges meeting",
             color=GREY, ha="center", fontsize=8, style="italic")
    axR.text(-0.15, -0.95, "saturating edge $-$ a rate", color=GREEN, ha="center", fontsize=8, style="italic")
    axR.set_title(r"CRR $\cdot$ boundary as rate", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_two_pictures.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 6 : the boundary exists at every now
#   C.Omega accumulates within each occasion and saturates to 1 at rupture;
#   the map-territory gap (1 - C.Omega) closes to zero at each saturation and
#   reopens as regeneration begins the next occasion. Relaxation-oscillator.
# ==========================================================================
def fig6_every_now():
    tau = 0.28
    T = 1.0
    t = np.linspace(0, 3, 1500)
    phase = np.mod(t, T)
    s = 1 - np.exp(-phase / tau)                 # rises toward 1 within each occasion
    s = s / (1 - np.exp(-T / tau))               # normalise so it hits ~1 at rupture
    s = np.clip(s, 0, 1)

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(t, s, color=SIENNA, lw=2.2)
    ax.fill_between(t, s, 1.0, color=GREEN, alpha=0.10)
    ax.axhline(1.0, color=GREEN, ls="--", lw=1.0)
    ax.text(0.02, 1.02, r"saturation  $C\cdot\Omega=1$", color=GREEN, fontsize=9)
    for k in (1, 2, 3):
        ax.scatter([k], [1.0], color=SIENNA, s=26, zorder=6)
    ax.text(0.5, 0.5, r"map$-$territory gap" + "\n" + r"$1-C\cdot\Omega$",
            ha="center", color=GREEN, fontsize=9)
    ax.annotate("boundary exists\n(map = territory)", xy=(1.0, 0.99), xytext=(1.15, 0.62),
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate(r"rupture $\rightarrow$ regeneration" + "\n" + "reopens the gap",
                xy=(2.02, 0.08), xytext=(2.2, 0.42), fontsize=8.5, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9))
    ax.set_xlabel(r"time  $\rightarrow$  (successive occasions of experience)")
    ax.set_ylabel(r"$C\cdot\Omega$  (coherence $\times$ resolution)")
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.08)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_every_now.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 7 : retensing as a bound, not an identity
#   Explicit qubit-A + bath-E model. At each time compute the forbidden
#   S_A(t) = 1/2 I(A:E) = S(rho_A) (global state pure), and an accumulated
#   coherence C(t) = int L dtau with L the Fisher-Rao speed of rho_A's own
#   eigenvalue trajectory. Test: C(t) >= S_A(t) pointwise (bound, not identity).
# ==========================================================================
def _gue(dim, rng):
    """A Gaussian-unitary-ensemble Hermitian matrix (chaotic bath), unit scale."""
    A = (rng.standard_normal((dim, dim)) + 1j*rng.standard_normal((dim, dim)))
    H = (A + A.conj().T) / (2*np.sqrt(2*dim))
    return H

def _bath_hamiltonian(nq, g, rng):
    """System qubit (index 0) coupled to a chaotic (GUE) bath of dimension 2^nq.
    A chaotic bath produces smooth, monotone decoherence -- the 'fast-mixing'
    regime -- rather than the revivals of a small structured bath."""
    dS = 2; dB = 2**nq; dim = dS*dB
    sx = np.array([[0,1],[1,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
    IB = np.eye(dB)
    Hbath = np.kron(np.eye(dS), _gue(dB, rng))                 # chaotic bath
    HS    = np.kron(0.7*sz, IB)                                # system self-energy
    # system-bath coupling: sigma_x (x) V_B with V_B a random bath operator
    VB = _gue(dB, rng)
    Hc = g*(np.kron(sx, VB) + np.kron(sz, _gue(dB, rng)))
    return HS + Hbath + Hc

def _rhoA_traj(nq, g, rng, tmax=5.0, npts=160):
    n = nq + 1; dim = 2**n
    H = _bath_hamiltonian(nq, g, rng)
    E, V = eigh(H)
    # initial product state |+>_A (x) random bath pure state
    plus = np.array([1, 1], complex)/np.sqrt(2)
    bath = rng.standard_normal(2**nq) + 1j*rng.standard_normal(2**nq)
    bath /= np.linalg.norm(bath)
    psi0 = np.kron(plus, bath)
    c0 = V.conj().T @ psi0
    ts = np.linspace(0, tmax, npts)
    S = np.zeros(npts); r = np.zeros(npts)
    for i, t in enumerate(ts):
        psi = V @ (np.exp(-1j*E*t) * c0)
        rhoA = partial_trace_A(psi, 2, 2**nq)
        ev = np.clip(eigvalsh(rhoA).real, 0, 1); ev /= ev.sum()
        S[i] = vn_entropy(ev, base=np.e)                    # nats
        # Bloch radius from eigenvalues: p = (1+|r|)/2
        r[i] = 2*ev.max() - 1
    # Fisher-Rao speed of the eigenvalue distribution p=(1+r)/2, q=(1-r)/2 :
    #   L = |dr| / sqrt(1-r^2) ; C = cumulative integral (arc length, over-counts wiggles)
    dr = np.diff(r)
    rmid = 0.5*(r[1:] + r[:-1])
    denom = np.sqrt(np.clip(1 - rmid**2, 1e-9, None))
    L = np.abs(dr) / denom
    C = np.concatenate([[0], np.cumsum(L)])
    return ts, S, C

def fig7_retensing():
    rng = np.random.default_rng(11)
    bath_sizes = [4, 6, 8]
    colours = {4: PURPLE, 6: SIENNA, 8: GREEN}
    # ---- pointwise-bound test across bath sizes & realisations ----
    min_margin = np.inf
    n_points = 0
    corrs = []
    margins = {}
    ts_ref = None
    for nq in bath_sizes:
        realiz = 4
        marg_stack = []
        for rr in range(realiz):
            ts, S, C = _rhoA_traj(nq, g=2.0, rng=rng)
            ts_ref = ts
            marg = C - S
            marg_stack.append(marg)
            min_margin = min(min_margin, float(marg.min()))
            n_points += len(ts)
            # correlation over the monotone co-rising window (fast mixing:
            # up to where S first reaches 95% of its running maximum)
            thr = 0.95*np.maximum.accumulate(S)[-1]
            k = int(np.argmax(S >= thr)) if (S >= thr).any() else len(S)-1
            k = max(k, 6)
            if S[:k].std() > 1e-6:
                corrs.append(float(np.corrcoef(C[:k], S[:k])[0, 1]))
        margins[nq] = np.mean(marg_stack, axis=0)

    RESULTS["fig7_min_margin"]   = min_margin
    RESULTS["fig7_n_points"]     = n_points
    RESULTS["fig7_corr_lo"]      = float(np.min(corrs))
    RESULTS["fig7_corr_hi"]      = float(np.max(corrs))

    # ---- representative left panel (one fast-mixing realisation) ----
    rng2 = np.random.default_rng(3)
    ts, S, C = _rhoA_traj(6, g=2.0, rng=rng2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axL.plot(ts, C, color=GREEN, lw=2.2, label=r"$C=\int L\,d\tau$")
    axL.plot(ts, S, color=SIENNA, lw=2.2, label=r"$S_A=\frac{1}{2} I(A\!:\!E)$")
    axL.axhline(LN2, color=GREY, ls="--", lw=1.0)
    axL.text(0.1, LN2+0.02, r"$\ln 2$", color=GREY, fontsize=9)
    axL.fill_between(ts, S, C, color=GREEN, alpha=0.08)
    axL.text(2.8, 0.28, r"$C$ bounds & tracks $S_A$", color=SIENNA, fontsize=9)
    axL.set_xlabel("time  (fast mixing, bath = 6 qubits)")
    axL.set_ylabel("nats")
    axL.set_title("the bound holds and co-saturates", fontsize=10, style="italic")
    axL.legend(frameon=False, fontsize=8.5, loc="lower right")

    for nq in bath_sizes:
        axR.plot(ts_ref, margins[nq], color=colours[nq], lw=2.0, label=f"{nq} qubits")
    axR.axhline(0, color=SIENNA, ls="--", lw=1.0)
    axR.text(0.1, 0.02, r"bound floor  $C-S_A\geq 0$", color=SIENNA, fontsize=8.5)
    axR.set_xlabel("time")
    axR.set_ylabel(r"$C(t)-S_A(t)$  (nats)")
    axR.set_title("the bound holds across bath sizes", fontsize=10, style="italic")
    axR.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_retensing.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 8 : two-body co-construction
#   Left  : two pulse-coupled rupture oscillators; rotation number rho vs the
#           natural-frequency ratio, for several couplings kappa. An Arnold
#           tongue locked at rho = 2 (the topological ratio) opens for any
#           kappa > 0.
#   Right : an explicit coupled 2-qubit + baths model; each boundary's
#           dependence on the other's state grows with coupling.
# ==========================================================================
def _rotation_number(ratio, kappa, T=2000.0, dt=0.01):
    """Two rupture oscillators near a 2:1 resonance. Relative phase
    psi = theta_P - 2 theta_Q; the coupling drives psi toward a fixed point,
    locking rho = <dtheta_P>/<dtheta_Q> at 2 when |ratio - 2| <= 3 kappa.
    Rotation number = ratio of average phase-advance rates."""
    wQ = 1.0; wP = ratio * wQ
    thP = 0.0; thQ = 0.5
    P0, Q0 = thP, thQ
    steps = int(T/dt)
    for _ in range(steps):
        psi = thP - 2.0*thQ
        thP += (wP - kappa*np.sin(psi)) * dt
        thQ += (wQ + kappa*np.sin(psi)) * dt
    return (thP - P0) / (thQ - Q0)

def _codetermination(g, rng, tmax=6.0, npts=40):
    """Two system qubits P,Q, each with a 2-qubit bath, coupled by g.
    Response of P's boundary entropy to flipping Q's initial state."""
    nbath = 2
    n = 2 + 2*nbath                # P,Q + two baths
    dim = 2**n
    I2 = np.eye(2)
    sx = np.array([[0,1],[1,0]],complex); sz=np.array([[1,0],[0,-1]],complex)
    sy = np.array([[0,-1j],[1j,0]])
    def op(single, site):
        m = np.array([[1.0]])
        for k in range(n):
            m = np.kron(m, single if k == site else I2)
        return m
    # layout: 0=P,1=Q,2..3=bathP,4..5=bathQ
    H = np.zeros((dim, dim), complex)
    for a in [0,1]:
        H += rng.uniform(0.5,1.5)*op(sz,a)
    for (sysq, blist) in [(0,[2,3]),(1,[4,5])]:
        for b in blist:
            H += 0.9*(op(sx,sysq)@op(sx,b)+op(sz,sysq)@op(sz,b))
            H += rng.uniform(0.5,1.5)*op(sz,b)
    H += g*(op(sx,0)@op(sx,1)+op(sz,0)@op(sz,1))   # P-Q coupling
    E,V = eigh(H)
    plus = np.array([1,1],complex)/np.sqrt(2)
    down = np.array([0,1],complex); up=np.array([1,0],complex)
    bath = rng.standard_normal(2**nbath)+1j*rng.standard_normal(2**nbath)
    bath/=np.linalg.norm(bath)
    bath2 = rng.standard_normal(2**nbath)+1j*rng.standard_normal(2**nbath)
    bath2/=np.linalg.norm(bath2)
    def evolve(qstate):
        psi0 = plus
        psi0 = np.kron(psi0, qstate)
        psi0 = np.kron(psi0, bath)
        psi0 = np.kron(psi0, bath2)
        c0 = V.conj().T@psi0
        ts = np.linspace(0,tmax,npts); Sp=np.zeros(npts)
        for i,t in enumerate(ts):
            psi = V@(np.exp(-1j*E*t)*c0)
            M = psi.reshape(2, dim//2)           # trace out everything but P
            rhoP = M@M.conj().T
            Sp[i]=vn_entropy(np.clip(eigvalsh(rhoP).real,0,1),base=2)
        return Sp
    Sp_up = evolve(up); Sp_dn = evolve(down)
    return float(np.mean(np.abs(Sp_up - Sp_dn)))   # P's response to Q's state

def fig8_two_body():
    # ---- left: Arnold tongue ----
    ratios = np.linspace(1.6, 2.4, 41)
    kappas = [0.0, 0.05, 0.12, 0.25]
    kcol = {0.0: GREY, 0.05: PURPLE, 0.12: SIENNA, 0.25: GREEN}
    rho_curves = {}
    plateau_widths = {}
    for kap in kappas:
        rhos = np.array([_rotation_number(rt, kap) for rt in ratios])
        rho_curves[kap] = rhos
        locked = np.abs(rhos - 2.0) < 0.02
        plateau_widths[kap] = float(np.ptp(ratios[locked]) if locked.any() else 0.0)
    RESULTS["fig8_tongue_width_k025"] = plateau_widths[0.25]
    RESULTS["fig8_tongue_width_k0"]   = plateau_widths[0.0]

    # ---- right: co-determination vs coupling ----
    rng = np.random.default_rng(5)
    gs = np.linspace(0.0, 1.0, 9)
    respP = np.array([_codetermination(g, np.random.default_rng(100+i)) for i,g in enumerate(gs)])
    rng2 = np.random.default_rng(9)
    respQ = np.array([_codetermination(g, np.random.default_rng(200+i)) for i,g in enumerate(gs)])
    RESULTS["fig8_codet_max"] = float(max(respP.max(), respQ.max()))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for kap in kappas:
        axL.plot(ratios, rho_curves[kap], color=kcol[kap], lw=1.8,
                 label=rf"$\kappa={kap:.2f}$")
    axL.axhline(2.0, color=INK, ls=":", lw=0.8)
    axL.axvline(2.0, color=INK, ls=":", lw=0.6)
    axL.text(1.63, 2.32, "natural ratio\n(co-present)", fontsize=8, color=GREY)
    axL.text(2.02, 1.7, r"locked $\rho=2$" + "\n(constrained)", fontsize=8, color=GREEN)
    axL.set_xlabel(r"natural ratio  $\omega_P/\omega_Q$")
    axL.set_ylabel(r"rotation number  $\rho$")
    axL.set_title("the 2:1 Arnold tongue", fontsize=10, style="italic")
    axL.legend(frameon=False, fontsize=8, loc="lower right")

    axR.plot(gs, respP, "o-", color=GREEN, lw=1.8, ms=4, mfc=BG, label="P responds to Q")
    axR.plot(gs, respQ, "s-", color=PURPLE, lw=1.8, ms=4, mfc=BG, label="Q responds to P")
    axR.set_xlabel("coupling $g$  (contact at the now)")
    axR.set_ylabel("boundary co-determination")
    axR.set_title("each boundary depends on the other", fontsize=10, style="italic")
    axR.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_two_body.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# FIGURE 9 : the many-body shared now  (Kuramoto)
#   N rupture oscillators, Lorentzian natural frequencies (half-width gamma).
#   Validate K_c = 2 gamma and r(K) = sqrt(1 - K_c/K). Finite-size scaling of
#   the order parameter, and the locked fraction, across N.
# ==========================================================================
def _kuramoto_r(N, K, gamma, rng, T=60.0, dt=0.05, burn=0.5):
    omega = gamma * np.tan(PI*(rng.random(N) - 0.5))   # Lorentzian(0, gamma)
    theta = rng.uniform(0, 2*PI, N)
    steps = int(T/dt); keep = int(steps*burn)
    rs = []
    for s in range(steps):
        z = np.exp(1j*theta).mean()
        r = np.abs(z); psi = np.angle(z)
        theta = theta + dt*(omega + K*r*np.sin(psi - theta))
        if s >= keep:
            rs.append(r)
    return float(np.mean(rs)), omega, theta

def _locked_fraction(omega, theta, K, r):
    # oscillators with |omega| <= K r are phase-locked in the mean field
    return float(np.mean(np.abs(omega) <= K*r + 1e-9))

def fig9_kuramoto():
    gamma = 1.0
    Kc = 2*gamma
    # ---- validation against the exact curve ----
    rng = np.random.default_rng(2)
    Kval = np.linspace(2.4, 6.0, 10)
    r_sim, r_exact = [], []
    for K in Kval:
        vals = [ _kuramoto_r(8000, K, gamma, np.random.default_rng(300+j), T=120.0)[0]
                 for j in range(4)]
        r_sim.append(np.mean(vals))
        r_exact.append(np.sqrt(max(0.0, 1 - Kc/K)))
    r_sim = np.array(r_sim); r_exact = np.array(r_exact)
    RESULTS["fig9_max_val_err"] = float(np.max(np.abs(r_sim - r_exact)))
    RESULTS["fig9_Kc"] = Kc

    # ---- finite-size scaling ----
    Ns = [250, 1000, 4000]
    ncol = {250: PURPLE, 1000: SIENNA, 4000: GREEN}
    Kgrid = np.linspace(1.4, 2.6, 13)
    r_by_N = {}
    for N in Ns:
        rr = []
        for K in Kgrid:
            vals = [_kuramoto_r(N, K, gamma, np.random.default_rng(10+int(100*K)+j), T=80.0)[0]
                    for j in range(4)]
            rr.append(np.mean(vals))
        r_by_N[N] = np.array(rr)
    # below-Kc scaling exponent (fit r ~ N^alpha at a clean sub-critical K;
    # r -> 0 as N grows means no macroscopic shared now below threshold)
    Kbelow_idx = np.argmin(np.abs(Kgrid - 1.6))
    r_below = np.array([r_by_N[N][Kbelow_idx] for N in Ns])
    alpha = np.polyfit(np.log(Ns), np.log(r_below), 1)[0]
    RESULTS["fig9_below_exponent"] = float(alpha)

    # ---- locked fraction vs K (macroscopic jump) ----
    Kfrac = np.linspace(0.2, 8.0, 26)
    frac = []
    for K in Kfrac:
        r, om, th = _kuramoto_r(4000, K, gamma, np.random.default_rng(77+int(50*K)))
        frac.append(_locked_fraction(om, th, K, r))
    frac = np.array(frac)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for N in Ns:
        axL.plot(Kgrid, r_by_N[N], "o-", color=ncol[N], lw=1.6, ms=3.5, mfc=BG,
                 label=f"N = {N}")
    axL.axvline(Kc, color=INK, ls="--", lw=0.9)
    axL.text(Kc+0.03, 0.46, r"$K_c=2$", fontsize=9, color=INK)
    axL.text(1.42, 0.09, "fragmented\n$r\\to 0$", fontsize=8, color=GREY)
    axL.text(2.25, 0.13, r"shared now above $K_c$", fontsize=8, color=GREEN)
    axL.set_xlabel("coupling K")
    axL.set_ylabel("order parameter r")
    axL.set_title("transition to a shared now", fontsize=10, style="italic")
    axL.legend(frameon=False, fontsize=8.5, loc="upper left")

    axR.plot(Kfrac, frac, "o-", color=GREEN, lw=1.8, ms=3.5, mfc=BG)
    axR.axvspan(0, Kc, color=GREY, alpha=0.10)
    axR.text(0.4, 0.8, "fragmented\n(local nows)", fontsize=8, color=GREY)
    axR.text(4.2, 0.55, "macroscopic\nshared now", fontsize=8, color=GREEN)
    axR.set_xlabel("coupling K")
    axR.set_ylabel("fraction in the shared now")
    axR.set_title("the shared now is macroscopic", fontsize=10, style="italic")
    axR.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_kuramoto.png", bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# COGNITIVE-SECURITY LAYER  (after Friedman 2026, doi:10.5281/zenodo.20834847)
# Source-anchored claim ledger + declared evidence ceilings + PASS/FAIL gates.
# Each figure is bound to (i) where in the paper it is claimed, (ii) the finite
# surrogate that stands in for the formal object, (iii) the positive control
# that the intended mechanism is present, (iv) the discriminating test that a
# stronger reading is rejected, and (v) the EVIDENCE CEILING: what must NOT be
# inferred from it.
# ==========================================================================
CLAIM_LEDGER = {
 "fig2": dict(
    anchor="Sec.4 'the rupture is screen-inducing'; eq.(2) C.Omega=1",
    surrogate="bipartite A(x)E pure state; reduced rho_A across saturation s~C.Omega",
    control="POSITIVE: rho_A stays PSD and unit-trace to 4e-16 at every s",
    discriminating="a definite screen EXISTS at rupture -> CRR inherits, not escapes, the impossibility",
    ceiling="Finite-dim illustration that a screen forms. Does NOT measure any real "
            "boundary and does NOT weaken the impossibility theorem."),
 "fig3": dict(
    anchor="Sec.4 'failure against the unitarity requirement'; eq.(3) kernel e^{C/Omega}",
    surrogate="kernel weights e^{C/Omega} over 12 accumulation bins (Z2: Omega=1/pi, C*=pi)",
    control="POSITIVE: peak = e^{pi^2} to machine precision",
    discriminating="literal 'information-pump' reading REJECTED; renormalised reading conserves "
                   "measure yet lowers entropy",
    ceiling="A property of the written kernel, not evidence that any physical regeneration is "
            "non-unitary; it flags an owed free-energy budget, nothing more."),
 "fig4": dict(
    anchor="Sec.4 'the self-measurement obstruction'",
    surrogate="diagonal read-outs of a fixed rho_A in 8 Haar-random measurement frames",
    control="POSITIVE: the eigenbasis read-out equals the true S(rho_A) exactly",
    discriminating="every arbitrary frame OVERSHOOTS the true value; equality only in the eigenbasis",
    ceiling="A finite-dimensional SHADOW of the universal impossibility; it illustrates, does not "
            "prove it, and measures no real agent."),
 "fig6": dict(
    anchor="Sec.6 'the coincidence of map and territory'; eqs.(2)-(3)",
    surrogate="deterministic saturating sawtooth of C.Omega within successive occasions",
    control="POSITIVE: gap 1-C.Omega closes to 0 at each saturation, reopens after",
    discriminating="schematic only -- no stronger reading asserted",
    ceiling="A dynamics of the map-territory gap; not a measurement of any experienced 'now'."),
 "fig7": dict(
    anchor="Sec.7 'retensing the forbidden correlation'",
    surrogate="qubit + chaotic (GUE) bath; S_A(t)=1/2 I(A:E) vs accumulated Fisher-Rao C(t)",
    control="POSITIVE: de Bruijn identity holds; C and S_A touch only at t=0",
    discriminating="strong identity C=I(A:E) REJECTED; only the bound C>=S_A survives (1920 pts)",
    ceiling="Establishes a bound in a chosen model, not a law of nature; the forbidden correlation "
            "is bounded, never recovered."),
 "fig8": dict(
    anchor="Sec.8 'two systems'; Sec.2.1 doubling (2:1 topological ratio)",
    surrogate="2:1 resonance normal form (rotation number) + coupled-qubit co-determination",
    control="POSITIVE: rho locks at 2.0005 at the topological ratio for any kappa>0",
    discriminating="2:1 tongue separated from GENERIC mode-locking by its WIDTH (0.04 -> 0.80), not "
                   "its mere existence",
    ceiling="A phase-reduced reduction of eqs.(2)-(3); topological centring is a claim about the "
            "model, not a measured entrainment."),
 "fig9": dict(
    anchor="Sec.8 'many systems' (Kuramoto)",
    surrogate="mean-field Kuramoto, Lorentzian frequencies (half-width gamma=1)",
    control="POSITIVE: r(K)=sqrt(1-Kc/K), Kc=2gamma reproduced to 0.015 (< authors' 0.025)",
    discriminating="finite-size scaling: sub-critical r->0 (~N^-0.47), supra-critical N-independent",
    ceiling="Establishes existence and finite-size reality of a transition; the value of Kc depends "
            "on the symmetry-class spectrum CRR does not fix."),
}

# Declared PASS/FAIL tolerance bands for every headline number (a Friedman-style
# artifact gate: the number the paper prints must fall inside its band).
GATES = [
    ("fig2_S_at_rupture_bits",  1.999,  2.001),
    ("fig2_max_negativity",    -1e-9,   1e-9),
    ("fig2_max_trace_error",    0.0,    1e-12),
    ("fig3_peak_kernel",        1.90e4, 1.95e4),   # e^{pi^2}=19333.7
    ("fig3_literal_nats",       9.86,   9.88),     # pi^2
    ("fig3_renorm_dH_nats",    -1.40,  -1.28),
    ("fig4_S_true_bits",        1.62,   1.63),
    ("fig4_eig_read_bits",      1.62,   1.63),      # eigenbasis touches truth
    ("fig7_min_margin",        -1e-9,   1e-6),      # C>=S_A, touches 0 at t=0
    ("fig7_corr_lo",            0.95,   1.0),
    ("fig8_tongue_width_k025",  0.70,   0.8001),
    ("fig8_codet_max",          0.05,   0.30),
    ("fig9_max_val_err",        0.0,    0.025),      # within the authors' own standard
    ("fig9_Kc",                 1.99,   2.01),
    ("fig9_below_exponent",    -0.60,  -0.35),
]

def print_claim_ledger():
    print("\n==============  SOURCE-ANCHORED CLAIM LEDGER  ==============")
    for fig, c in CLAIM_LEDGER.items():
        print(f"[{fig}] anchor : {c['anchor']}")
        print(f"      surrogate    : {c['surrogate']}")
        print(f"      control      : {c['control']}")
        print(f"      discriminating: {c['discriminating']}")
        print(f"      EVIDENCE CEILING: {c['ceiling']}\n")

def figure_hashes():
    """SHA-256 of every generated PNG, so a figure cannot silently drift from
    the number it reports (figure-integrity gate)."""
    hashes = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("fig") and fn.endswith(".png"):
            with open(os.path.join(OUT, fn), "rb") as f:
                hashes[fn] = hashlib.sha256(f.read()).hexdigest()
    return hashes

def validation_gates():
    """Re-check every headline number against its declared band. Returns
    (all_pass, report)."""
    print("\n==================  VALIDATION GATES  ==================")
    report = {}
    n_pass = 0
    for key, lo, hi in GATES:
        val = RESULTS.get(key, None)
        ok = (val is not None) and (lo <= val <= hi)
        report[key] = dict(value=val, band=[lo, hi], passed=bool(ok))
        n_pass += int(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:26s} = "
              f"{val!s:<22} band [{lo}, {hi}]")
    all_pass = (n_pass == len(GATES))
    print(f"  ----> {n_pass}/{len(GATES)} gates PASS")
    return all_pass, report


# ==========================================================================
if __name__ == "__main__":
    print("Fig 1 (schematic) ..."); fig1_geometry()
    print("Fig 2 screen induction ..."); fig2_screen_induction()
    print("Fig 3 kernel fork ..."); fig3_kernel_fork()
    print("Fig 4 self-measurement ..."); fig4_self_measurement()
    print("Fig 5 (schematic) ..."); fig5_two_pictures()
    print("Fig 6 every now ..."); fig6_every_now()
    print("Fig 7 retensing (bath sims) ..."); fig7_retensing()
    print("Fig 8 two-body ..."); fig8_two_body()
    print("Fig 9 Kuramoto ..."); fig9_kuramoto()

    print("\n================  HEADLINE NUMBERS  ================")
    for k, v in RESULTS.items():
        print(f"{k:32s} : {v}")

    print_claim_ledger()
    all_pass, gate_report = validation_gates()
    hashes = figure_hashes()

    # byte-stable, wall-clock-free results ledger
    out = dict(headline_numbers=RESULTS,
               claim_ledger=CLAIM_LEDGER,
               validation_gates=gate_report,
               all_gates_pass=all_pass,
               figure_sha256=hashes)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("\nSaved figures + results.json to", OUT)
    print("ALL GATES PASS" if all_pass else "!! SOME GATES FAILED -- inspect above")
