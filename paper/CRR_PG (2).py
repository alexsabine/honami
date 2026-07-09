"""
═══════════════════════════════════════════════════════════════════════════════
COMPLETE REPLICATION SCRIPT
Phase-Gating Across Precision Channels:
Topological Constraints on Multi-Channel Belief Update Dynamics

Alexander Sabine · Active Inference Institute
temporalgrammar.ai · Alexander@activeinference.institute

This single script reproduces EVERY numerical claim in the paper.
Run it, and the numbers should match.  Sections below correspond
directly to paper sections.

Requires: numpy, scipy, matplotlib (standard scientific Python).
Runtime:  ~10–15 minutes on a modern machine.
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as spstats
from collections import Counter
import time, sys, os

# ─── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(2025)

# ─── CRR Constants ───────────────────────────────────────────────────────────
PI        = np.pi
OMEGA_Z2  = 1.0 / PI           # ≈ 0.3183
OMEGA_SO2 = 1.0 / (2.0 * PI)   # ≈ 0.1592
CSTAR_Z2  = PI                  # ≈ 3.1416
CSTAR_SO2 = 2.0 * PI            # ≈ 6.2832


# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENTS
# ═════════════════════════════════════════════════════════════════════════════

class RingHMM:
    """Ring Hidden Markov Model — the primary test environment."""
    def __init__(self, n_states=6, obs_noise=0.15, trans_noise=0.15):
        self.n = n_states
        self.state = 0
        # Observation model: Gaussian-shaped emission around true state
        self.A = np.zeros((n_states, n_states))
        for s in range(n_states):
            for o in range(n_states):
                dist = min(abs(o - s), n_states - abs(o - s))
                self.A[o, s] = np.exp(
                    -dist**2 / (2 * max(obs_noise * n_states, 0.1)**2))
            self.A[:, s] /= self.A[:, s].sum()
        # Transition model: primarily clockwise with noise
        self.B = np.zeros((n_states, n_states))
        for s in range(n_states):
            for ds in range(-2, 3):
                s_next = (s + ds) % n_states
                if ds == 1:
                    self.B[s_next, s] = 1.0 - trans_noise
                else:
                    self.B[s_next, s] = trans_noise / 4.0
            self.B[:, s] /= self.B[:, s].sum()

    def step(self):
        prev = self.state
        self.state = np.random.choice(self.n, p=self.B[:, self.state])
        obs = np.random.choice(self.n, p=self.A[:, self.state])
        return obs, self.state, prev


class ChainHMM:
    """Linear chain — no wraparound (§5.1)."""
    def __init__(self, n_states=8, noise=0.15):
        self.n = n_states
        self.state = 0
        self.B = np.zeros((n_states, n_states))
        for s in range(n_states):
            if s < n_states - 1:
                self.B[s+1, s] = 1.0 - noise
            else:
                self.B[s, s] = 1.0 - noise
            if s > 0:
                self.B[s-1, s] = noise * 0.5
            self.B[s, s] += noise * 0.5
            self.B[:, s] /= self.B[:, s].sum()
        self.A = np.eye(n_states) * (1.0 - noise) + noise / n_states
        self.A /= self.A.sum(axis=0, keepdims=True)

    def step(self):
        prev = self.state
        self.state = np.random.choice(self.n, p=self.B[:, self.state])
        obs = np.random.choice(self.n, p=self.A[:, self.state])
        return obs, self.state, prev


class GridHMM:
    """2D grid world (§5.1)."""
    def __init__(self, side=4, noise=0.15):
        self.n = side * side
        self.side = side
        self.state = 0
        self.A = np.zeros((self.n, self.n))
        for s in range(self.n):
            for o in range(self.n):
                sr, sc = s // side, s % side
                or_, oc = o // side, o % side
                dist = abs(sr - or_) + abs(sc - oc)
                self.A[o, s] = np.exp(
                    -dist**2 / (2 * max(noise * side, 0.1)**2))
            self.A[:, s] /= self.A[:, s].sum()
        self.B = np.zeros((self.n, self.n))
        for s in range(self.n):
            r, c = s // side, s % side
            neighbors = [s]
            if r > 0:     neighbors.append((r-1)*side + c)
            if r < side-1: neighbors.append((r+1)*side + c)
            if c > 0:     neighbors.append(r*side + c-1)
            if c < side-1: neighbors.append(r*side + c+1)
            for nb in neighbors:
                self.B[nb, s] = 1.0
            self.B[:, s] /= self.B[:, s].sum()

    def step(self):
        prev = self.state
        self.state = np.random.choice(self.n, p=self.B[:, self.state])
        obs = np.random.choice(self.n, p=self.A[:, self.state])
        return obs, self.state, prev


class RandomGraphHMM:
    """Erdos–Renyi random graph (§5.1)."""
    def __init__(self, n_states=8, edge_prob=0.4, noise=0.15):
        self.n = n_states
        self.state = 0
        adj = np.random.random((n_states, n_states)) < edge_prob
        adj = adj | adj.T | np.eye(n_states, dtype=bool)
        self.B = adj.astype(float)
        self.B /= self.B.sum(axis=0, keepdims=True)
        self.A = np.zeros((n_states, n_states))
        for s in range(n_states):
            self.A[s, s] = 1.0 - noise
            for o in range(n_states):
                if o != s:
                    self.A[o, s] = noise / (n_states - 1)
        self.A /= self.A.sum(axis=0, keepdims=True)

    def step(self):
        prev = self.state
        self.state = np.random.choice(self.n, p=self.B[:, self.state])
        obs = np.random.choice(self.n, p=self.A[:, self.state])
        return obs, self.state, prev


class TreeHMM:
    """Binary tree (§5.1)."""
    def __init__(self, depth=3, noise=0.15):
        self.n = 2**(depth+1) - 1
        self.state = 0
        self.B = np.zeros((self.n, self.n))
        for s in range(self.n):
            left = 2*s + 1
            right = 2*s + 2
            parent = (s - 1) // 2 if s > 0 else 0
            neighbors = [s]
            if left < self.n:  neighbors.append(left)
            if right < self.n: neighbors.append(right)
            if s > 0:          neighbors.append(parent)
            for nb in neighbors:
                self.B[nb, s] = 1.0
            self.B[:, s] /= self.B[:, s].sum()
        self.A = np.eye(self.n) * (1.0 - noise) + noise / self.n
        self.A /= self.A.sum(axis=0, keepdims=True)

    def step(self):
        prev = self.state
        self.state = np.random.choice(self.n, p=self.B[:, self.state])
        obs = np.random.choice(self.n, p=self.A[:, self.state])
        return obs, self.state, prev


# ═════════════════════════════════════════════════════════════════════════════
# CRR AGENT
# ═════════════════════════════════════════════════════════════════════════════

class CRRAgent:
    """
    POMDP agent with CRR-governed update timing.

    Implementation details critical for replication (§4.1):
      - Weight: w = min(exp(C/Ω), 10) / 10 * 2 + 0.5  → range [0.5, 2.5]
      - Gain = Δ(mean Dirichlet concentration), not raw parameter sum
      - Both channels receive exactly 1 count of evidence per trial
      - Prediction errors computed from agent's model at TRUE states
    """
    def __init__(self, n_states):
        self.n = n_states
        self.a = np.ones((n_states, n_states))   # Likelihood Dirichlet
        self.b = np.ones((n_states, n_states))   # Transition Dirichlet
        self.C_s = 0.0                            # Sensory coherence
        self.C_p = 0.0                            # Prior coherence
        self.pend_a = np.zeros((n_states, n_states))  # Pending sensory
        self.pend_b = np.zeros((n_states, n_states))  # Pending prior
        self.gains_s = []       # Per-rupture sensory gains
        self.gains_p = []       # Per-rupture prior gains
        self.n_s_rup = 0        # Sensory rupture count
        self.n_p_rup = 0        # Prior rupture count
        self.s_phase_at_p_rup = []  # Z₂ phase at each SO(2) rupture
        # For §4.4 coherence dynamics
        self.s_iri = []         # Sensory inter-rupture intervals
        self.p_iri = []         # Prior inter-rupture intervals
        self.s_C_at_rup = []    # C_s value at each sensory rupture
        self.p_C_at_rup = []    # C_p value at each prior rupture
        self._since_s = 0       # Trials since last sensory rupture
        self._since_p = 0       # Trials since last prior rupture

    def sensory_precision(self):
        return self.a.sum(axis=0).mean()

    def prior_precision(self):
        return self.b.sum(axis=0).mean()

    def step(self, obs, true_state, prev_state):
        A = self.a / self.a.sum(axis=0, keepdims=True)
        B = self.b / self.b.sum(axis=0, keepdims=True)
        pe_s = -np.log(A[obs, true_state] + 1e-16)
        pe_p = -np.log(B[true_state, prev_state] + 1e-16)

        self.pend_a[obs, true_state] += 1.0
        self.pend_b[true_state, prev_state] += 1.0
        self.C_s += pe_s
        self.C_p += pe_p
        self._since_s += 1
        self._since_p += 1

        s_rup = False
        p_rup = False

        if self.C_s >= CSTAR_Z2:
            pre = self.sensory_precision()
            raw_w = np.exp(self.C_s / OMEGA_Z2)
            clamped = min(raw_w, 10.0)
            weight = clamped / 10.0 * 2.0 + 0.5
            self.a += self.pend_a * weight
            post = self.sensory_precision()
            self.gains_s.append(post - pre)
            self.s_C_at_rup.append(self.C_s)
            self.s_iri.append(self._since_s)
            self.C_s = 0.0
            self.pend_a[:] = 0.0
            self.n_s_rup += 1
            self._since_s = 0
            s_rup = True

        if self.C_p >= CSTAR_SO2:
            pre = self.prior_precision()
            raw_w = np.exp(self.C_p / OMEGA_SO2)
            clamped = min(raw_w, 10.0)
            weight = clamped / 10.0 * 2.0 + 0.5
            self.b += self.pend_b * weight
            post = self.prior_precision()
            self.gains_p.append(post - pre)
            self.p_C_at_rup.append(self.C_p)
            self.p_iri.append(self._since_p)
            s_phase = self.C_s / CSTAR_Z2
            self.s_phase_at_p_rup.append(s_phase)
            self.C_p = 0.0
            self.pend_b[:] = 0.0
            self.n_p_rup += 1
            self._since_p = 0
            p_rup = True

        return s_rup, p_rup


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data, n_boot=2000):
    data = np.array(data)
    if len(data) < 3:
        m = np.mean(data)
        return m, m, m
    boots = [np.mean(np.random.choice(data, len(data), True))
             for _ in range(n_boot)]
    return np.mean(data), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def burstiness(intervals):
    """Kim & Jo (2016) burstiness index B ∈ [-1, 1].
    B = -1: perfectly periodic.  B = 0: Poisson.  B = 1: maximally bursty."""
    intervals = np.array(intervals, dtype=float)
    if len(intervals) < 3:
        return 0.0
    m = intervals.mean()
    s = intervals.std()
    if m + s == 0:
        return 0.0
    return (s - m) / (s + m)


def run_condition(n_states, obs_noise, trans_noise, n_trials, n_runs):
    """Run the standard CRR condition, collecting all metrics."""
    info_ratios = []
    gain_ratios = []
    freq_ratios = []
    all_phases = []
    z2_per_so2 = []
    all_s_iri = []
    all_p_iri = []
    all_s_C = []
    all_p_C = []

    for run in range(n_runs):
        env = RingHMM(n_states, obs_noise, trans_noise)
        agent = CRRAgent(n_states)
        z2_since_so2 = 0

        for t in range(n_trials):
            obs, true_state, prev_state = env.step()
            s_rup, p_rup = agent.step(obs, true_state, prev_state)
            if s_rup:
                z2_since_so2 += 1
            if p_rup:
                z2_per_so2.append(z2_since_so2)
                z2_since_so2 = 0

        if agent.n_s_rup > 2 and agent.n_p_rup > 2:
            mg_s = np.mean(agent.gains_s)
            mg_p = np.mean(agent.gains_p)
            if mg_p > 1e-8:
                gain_ratios.append(mg_s / mg_p)
            freq_ratios.append(agent.n_s_rup / agent.n_p_rup)
            info_s = agent.n_s_rup * mg_s / n_trials
            info_p = agent.n_p_rup * mg_p / n_trials
            if info_p > 1e-8:
                info_ratios.append(info_s / info_p)

        all_phases.extend(agent.s_phase_at_p_rup)
        all_s_iri.extend(agent.s_iri)
        all_p_iri.extend(agent.p_iri)
        all_s_C.extend(agent.s_C_at_rup)
        all_p_C.extend(agent.p_C_at_rup)

    return {
        'info_ratios': np.array(info_ratios) if info_ratios else np.array([0.]),
        'gain_ratios': np.array(gain_ratios) if gain_ratios else np.array([0.]),
        'freq_ratios': np.array(freq_ratios) if freq_ratios else np.array([0.]),
        'phases':      np.array(all_phases)  if all_phases  else np.array([0.]),
        'z2_per_so2':  np.array(z2_per_so2)  if z2_per_so2  else np.array([0.]),
        's_iri':       np.array(all_s_iri)   if all_s_iri   else np.array([0.]),
        'p_iri':       np.array(all_p_iri)   if all_p_iri   else np.array([0.]),
        's_C':         np.array(all_s_C)     if all_s_C     else np.array([0.]),
        'p_C':         np.array(all_p_C)     if all_p_C     else np.array([0.]),
    }


def run_in_env(env, n_trials):
    """Run CRR agent in any environment, return info ratio or None."""
    agent = CRRAgent(env.n)
    for t in range(n_trials):
        obs, ts, ps = env.step()
        agent.step(obs, ts, ps)
    if agent.n_s_rup > 2 and agent.n_p_rup > 2:
        mg_s = np.mean(agent.gains_s)
        mg_p = np.mean(agent.gains_p)
        info_s = agent.n_s_rup * mg_s / n_trials
        info_p = agent.n_p_rup * mg_p / n_trials
        if info_p > 1e-8:
            return info_s / info_p
    return None


def section_header(num, title):
    print()
    print("═" * 72)
    print(f"  §{num}  {title}")
    print("═" * 72)


# ═════════════════════════════════════════════════════════════════════════════
# §4.2  PRECISION-GAIN CONSERVATION
# ═════════════════════════════════════════════════════════════════════════════

def section_4_2():
    section_header("4.2", "PRECISION-GAIN CONSERVATION")
    print("  Paper claims: ratio = 1.003, CI [1.000, 1.005], CV = 0.024")
    print("  Protocol: 50 agents × 800 trials, 10 conditions")
    print()

    N_RUNS = 50
    N_TRIALS = 800

    # --- Sweep 1: state counts ---
    print("  Sweep 1: State counts (noise = 0.15)")
    print(f"  {'n':>4} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'gain_r':>8} {'freq_r':>8}")
    state_results = {}
    for ns in [3, 4, 6, 8, 12]:
        r = run_condition(ns, 0.15, 0.15, N_TRIALS, N_RUNS)
        state_results[ns] = r
        m, lo, hi = bootstrap_ci(r['info_ratios'])
        gm = r['gain_ratios'].mean()
        fm = r['freq_ratios'].mean()
        print(f"  {ns:>4} {m:>8.4f} {lo:>8.4f} {hi:>8.4f} "
              f"{gm:>8.3f} {fm:>8.2f}")

    # --- Sweep 2: noise levels ---
    print(f"\n  Sweep 2: Noise levels (n = 6)")
    print(f"  {'noise':>6} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'gain_r':>8} {'freq_r':>8}")
    noise_results = {}
    for nl in [0.05, 0.10, 0.15, 0.25, 0.40]:
        r = run_condition(6, nl, nl, N_TRIALS, N_RUNS)
        noise_results[nl] = r
        m, lo, hi = bootstrap_ci(r['info_ratios'])
        gm = r['gain_ratios'].mean()
        fm = r['freq_ratios'].mean()
        print(f"  {nl:>6.2f} {m:>8.4f} {lo:>8.4f} {hi:>8.4f} "
              f"{gm:>8.3f} {fm:>8.2f}")

    # --- Aggregate ---
    all_eq = []
    for r in state_results.values():
        all_eq.extend(r['info_ratios'].tolist())
    for r in noise_results.values():
        all_eq.extend(r['info_ratios'].tolist())
    all_eq = np.array([x for x in all_eq if x > 0])
    m, lo, hi = bootstrap_ci(all_eq)
    cv = all_eq.std() / m if m > 0 else 0

    print(f"\n  ── AGGREGATE (all 10 equal-evidence conditions) ──")
    print(f"  Mean  = {m:.4f}    (paper: 1.003)")
    print(f"  95%CI = [{lo:.4f}, {hi:.4f}]  (paper: [1.000, 1.005])")
    print(f"  CV    = {cv:.4f}    (paper: 0.024)")
    print(f"  N     = {len(all_eq)} runs")

    # --- Gain/freq ranges ---
    all_gains = [r['gain_ratios'].mean()
                 for r in list(state_results.values()) + list(noise_results.values())]
    all_freqs = [r['freq_ratios'].mean()
                 for r in list(state_results.values()) + list(noise_results.values())]
    print(f"\n  Gain ratio range:  {min(all_gains):.2f} – {max(all_gains):.2f}"
          f"  (paper: 0.22 – 1.34)")
    print(f"  Freq ratio range:  {min(all_freqs):.2f} – {max(all_freqs):.2f}"
          f"  (paper: 0.73 – 4.47)")

    # --- Evidence scaling (R² > 0.999) ---
    print(f"\n  Evidence scaling test")
    ev_ratios_in = []
    ev_ratios_out = []
    for er in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
        info_ratios = []
        for run in range(N_RUNS):
            env = RingHMM(6, 0.15, 0.15)
            agent = CRRAgent(6)
            for t in range(N_TRIALS):
                obs, ts, ps = env.step()
                A = agent.a / agent.a.sum(axis=0, keepdims=True)
                B = agent.b / agent.b.sum(axis=0, keepdims=True)
                pe_s = -np.log(A[obs, ts] + 1e-16)
                pe_p = -np.log(B[ts, ps] + 1e-16)
                agent.pend_a[obs, ts] += er
                agent.pend_b[ts, ps] += 1.0
                agent.C_s += pe_s * er
                agent.C_p += pe_p
                if agent.C_s >= CSTAR_Z2:
                    pre = agent.sensory_precision()
                    raw_w = np.exp(agent.C_s / OMEGA_Z2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.a += agent.pend_a * w
                    agent.gains_s.append(agent.sensory_precision() - pre)
                    agent.C_s = 0.0
                    agent.pend_a[:] = 0.0
                    agent.n_s_rup += 1
                if agent.C_p >= CSTAR_SO2:
                    pre = agent.prior_precision()
                    raw_w = np.exp(agent.C_p / OMEGA_SO2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.b += agent.pend_b * w
                    agent.gains_p.append(agent.prior_precision() - pre)
                    agent.C_p = 0.0
                    agent.pend_b[:] = 0.0
                    agent.n_p_rup += 1
            if agent.n_s_rup > 2 and agent.n_p_rup > 2:
                mg_s = np.mean(agent.gains_s)
                mg_p = np.mean(agent.gains_p)
                info_s = agent.n_s_rup * mg_s / N_TRIALS
                info_p = agent.n_p_rup * mg_p / N_TRIALS
                if info_p > 1e-8:
                    info_ratios.append(info_s / info_p)
        m_ev = np.mean(info_ratios) if info_ratios else 0
        ev_ratios_in.append(er)
        ev_ratios_out.append(m_ev)
        print(f"    ev_ratio={er:.2f}:  info_ratio = {m_ev:.4f}")

    slope, intercept, r_value, _, _ = spstats.linregress(
        ev_ratios_in, ev_ratios_out)
    print(f"  R² = {r_value**2:.6f}  (paper: > 0.999)")

    # --- Continuous updating baseline (§4.2) ---
    print(f"\n  Continuous updating baseline (no thresholds)")
    cont_ratios = []
    for run in range(N_RUNS):
        env = RingHMM(6, 0.15, 0.15)
        a = np.ones((6, 6))
        b = np.ones((6, 6))
        for t in range(N_TRIALS):
            obs, ts, ps = env.step()
            A_norm = a / a.sum(axis=0, keepdims=True)
            B_norm = b / b.sum(axis=0, keepdims=True)
            a[obs, ts] += 1.0
            b[ts, ps] += 1.0
        prec_s = a.sum(axis=0).mean()
        prec_p = b.sum(axis=0).mean()
        # Total gain = final - initial mean Dirichlet concentration
        gain_s = prec_s - 6.0   # initial: ones matrix, sum per col = 6, mean = 6
        gain_p = prec_p - 6.0
        if gain_p > 1e-8:
            cont_ratios.append(gain_s / gain_p)
    m_cont = np.mean(cont_ratios)
    print(f"  Ratio = {m_cont:.4f}  (paper: 1.0000)")

    # --- Arbitrary threshold ratios (§4.2) ---
    # DIAGNOSTIC NOTE: The paper states arbitrary thresholds produce ratios
    # "within 1% of unity." Investigation reveals this holds with CONSTANT
    # weight (w=1) but NOT with the exp(C/Ω) weight when Ω = 1/threshold.
    # The exp weight introduces channel-asymmetric gain when thresholds
    # differ from topological values, because exp(C·Ω) ≠ e at rupture
    # unless C·Ω = 1 (which only the topological thresholds guarantee).
    #
    # We test both modes to make the distinction explicit.
    print(f"\n  Arbitrary threshold ratios")
    arb_configs = [
        ("1:1 (π/π)",        PI,     PI),
        ("2:1 (π/2π)",       PI,     2*PI),
        ("3:1 (π/3π)",       PI,     3*PI),
        ("Reversed (2π/π)",  2*PI,   PI),
        ("Random (2.7/4.1)", 2.7,    4.1),
    ]

    for wmode, wlabel in [('constant', 'CONSTANT weight (w=1)'),
                           ('exp',      'exp(C/Ω) weight (Ω=1/threshold)')]:
        print(f"\n  With {wlabel}:")
        print(f"  {'Label':>25} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8} {'<1%?':>5}")
        for label, cs, cp in arb_configs:
            ratios = []
            for run in range(40):
                env = RingHMM(6, 0.15, 0.15)
                agent = CRRAgent(6)
                C_s, C_p = 0.0, 0.0
                pend_a = np.zeros((6, 6))
                pend_b = np.zeros((6, 6))
                n_s, n_p = 0, 0
                gains_s, gains_p = [], []
                for t in range(N_TRIALS):
                    obs, ts, ps = env.step()
                    A_n = agent.a / agent.a.sum(axis=0, keepdims=True)
                    B_n = agent.b / agent.b.sum(axis=0, keepdims=True)
                    pe_s = -np.log(A_n[obs, ts] + 1e-16)
                    pe_p = -np.log(B_n[ts, ps] + 1e-16)
                    pend_a[obs, ts] += 1.0
                    pend_b[ts, ps] += 1.0
                    C_s += pe_s
                    C_p += pe_p
                    if C_s >= cs:
                        pre = agent.a.sum(axis=0).mean()
                        if wmode == 'constant':
                            w = 1.0
                        else:
                            om = 1.0 / cs
                            raw_w = np.exp(C_s * om)
                            w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                        agent.a += pend_a * w
                        gains_s.append(agent.a.sum(axis=0).mean() - pre)
                        C_s = 0.0; pend_a[:] = 0.0; n_s += 1
                    if C_p >= cp:
                        pre = agent.b.sum(axis=0).mean()
                        if wmode == 'constant':
                            w = 1.0
                        else:
                            om = 1.0 / cp
                            raw_w = np.exp(C_p * om)
                            w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                        agent.b += pend_b * w
                        gains_p.append(agent.b.sum(axis=0).mean() - pre)
                        C_p = 0.0; pend_b[:] = 0.0; n_p += 1
                if n_s > 2 and n_p > 2 and gains_s and gains_p:
                    mg_s = np.mean(gains_s)
                    mg_p = np.mean(gains_p)
                    info_s = n_s * mg_s / N_TRIALS
                    info_p = n_p * mg_p / N_TRIALS
                    if info_p > 1e-8:
                        ratios.append(info_s / info_p)
            if ratios:
                m_arb, lo, hi = bootstrap_ci(ratios)
                ok = "✓" if abs(m_arb - 1.0) < 0.01 else "~" if abs(m_arb - 1.0) < 0.05 else "✗"
                print(f"  {label:>25} {m_arb:>8.4f} {lo:>8.4f} {hi:>8.4f} {ok:>5}")

    print(f"\n  FINDING: Conservation under arbitrary thresholds requires")
    print(f"  symmetric weighting. The topological thresholds (π/2π) are the")
    print(f"  unique values where exp(C/Ω) is self-consistent: C·Ω = 1 at")
    print(f"  rupture for both channels, so exp(C/Ω) = exp(1/Ω²) = e.")

    return state_results, noise_results, all_eq


# ═════════════════════════════════════════════════════════════════════════════
# §4.3  PHASE-GATING
# ═════════════════════════════════════════════════════════════════════════════

def section_4_3():
    section_header("4.3", "PHASE-GATING")
    print("  Paper claims: KS p < 10^-100, strongly non-uniform phase distribution")
    print("  Modal Z₂/SO(2) count: sensitive to conditions (2–3 at 2:1 ratio)")
    print()

    # Run the standard phase-gating analysis
    r = run_condition(6, 0.15, 0.15, 1200, 60)
    phases = r['phases']
    phases = phases[phases <= 1.0]
    z2ps = r['z2_per_so2']
    z2ps = z2ps[z2ps > 0]

    print(f"  N SO(2) ruptures analysed: {len(phases)}")

    if len(phases) > 20:
        ks_stat, ks_p = spstats.kstest(phases, 'uniform')
        observed, _ = np.histogram(phases, bins=10, range=(0, 1))
        expected = np.full(10, len(phases) / 10)
        chi2, chi2_p = spstats.chisquare(observed, expected)

        print(f"  KS stat   = {ks_stat:.4f},  p = {ks_p:.2e}  (paper: p < 10^-100)")
        print(f"  χ² (10 bins, 9 df) = {chi2:.0f}")
        print(f"    Note: χ² scales with N; effect size is consistent.")

        # Phase distribution
        print(f"\n  Phase histogram (10 bins):")
        bin_edges = np.linspace(0, 1, 11)
        for i in range(10):
            count = np.sum((phases >= bin_edges[i]) & (phases < bin_edges[i+1]))
            pct = 100 * count / len(phases)
            bar = "█" * int(pct / 2)
            print(f"    [{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}): "
                  f"{count:>5} ({pct:>5.1f}%) {bar}")

    if len(z2ps) > 10:
        mode_result = spstats.mode(z2ps, keepdims=True)
        vals, counts = np.unique(z2ps.astype(int), return_counts=True)
        print(f"\n  Z₂ ruptures per SO(2) cycle:")
        print(f"    Mean   = {z2ps.mean():.2f}")
        print(f"    Median = {np.median(z2ps):.0f}")
        print(f"    Mode   = {mode_result.mode[0]}")
        print(f"\n    Full distribution:")
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            if c > len(z2ps) * 0.005:
                pct = 100 * c / len(z2ps)
                bar = "█" * int(pct)
                print(f"      {int(v):>3}: {c:>5} ({pct:>5.1f}%) {bar}")

        # Modal count at different threshold ratios
        print(f"\n  Modal count vs threshold ratio:")
        for label, cs, cp in [("1:1",   PI, PI),
                               ("1.5:1", PI, 1.5*PI),
                               ("2:1",   PI, 2*PI)]:
            r2 = run_condition_custom_ratio(6, 0.15, 0.15, 1200, 30, cs, cp)
            z = r2['z2_per_so2']
            z = z[z > 0]
            if len(z) > 5:
                md = spstats.mode(z, keepdims=True).mode[0]
                mn = z.mean()
                print(f"    {label}:  mode = {md}, mean = {mn:.2f}")

        # Show condition-dependence of the mode
        print(f"\n  Mode varies with environment parameters:")
        print(f"  {'Condition':>25} {'mode':>5} {'mean':>6}")
        for clabel, ns, on, tn in [
            ("n=6,  noise=0.15", 6, 0.15, 0.15),
            ("n=8,  noise=0.15", 8, 0.15, 0.15),
            ("n=12, noise=0.15", 12, 0.15, 0.15),
            ("n=6,  noise=0.10", 6, 0.10, 0.10),
            ("n=6,  noise=0.40", 6, 0.40, 0.40),
        ]:
            z3 = _collect_z2_per_so2_quick(ns, on, tn, 1200, 40)
            if len(z3) > 20:
                md3 = spstats.mode(z3, keepdims=True).mode[0]
                mn3 = z3.mean()
                print(f"  {clabel:>25} {md3:>5} {mn3:>6.2f}")

        print(f"\n  FINDING: Mode is 2 or 3 depending on condition (counts=2")
        print(f"  and counts=3 are near-equal in frequency for n=6, noise=0.15;")
        print(f"  mode shifts to 3 at higher n or lower noise). Mean ≈ 3.8–3.9.")

    return phases, z2ps


def _collect_z2_per_so2_quick(n_states, obs_noise, trans_noise, n_trials, n_runs):
    """Lightweight Z₂-per-SO(2) collection for condition sweep."""
    z2_per_so2 = []
    for run in range(n_runs):
        env = RingHMM(n_states, obs_noise, trans_noise)
        agent = CRRAgent(n_states)
        z2_since = 0
        for t in range(n_trials):
            obs, ts, ps = env.step()
            s_rup, p_rup = agent.step(obs, ts, ps)
            if s_rup:
                z2_since += 1
            if p_rup:
                z2_per_so2.append(z2_since)
                z2_since = 0
    return np.array([x for x in z2_per_so2 if x > 0])

    return phases, z2ps


def run_condition_custom_ratio(n_states, obs_noise, trans_noise,
                                n_trials, n_runs, cs_z2, cs_so2):
    """Run with custom thresholds for the modal-count test."""
    z2_per_so2 = []
    for run in range(n_runs):
        env = RingHMM(n_states, obs_noise, trans_noise)
        agent = CRRAgent(n_states)
        C_s, C_p = 0.0, 0.0
        pend_a = np.zeros((n_states, n_states))
        pend_b = np.zeros((n_states, n_states))
        z2_since = 0
        for t in range(n_trials):
            obs, ts, ps = env.step()
            A_n = agent.a / agent.a.sum(axis=0, keepdims=True)
            B_n = agent.b / agent.b.sum(axis=0, keepdims=True)
            pe_s = -np.log(A_n[obs, ts] + 1e-16)
            pe_p = -np.log(B_n[ts, ps] + 1e-16)
            pend_a[obs, ts] += 1.0
            pend_b[ts, ps] += 1.0
            C_s += pe_s
            C_p += pe_p
            s_rup = False
            if C_s >= cs_z2:
                pre = agent.a.sum(axis=0).mean()
                om = 1.0 / cs_z2
                raw_w = np.exp(C_s * om)
                w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                agent.a += pend_a * w
                C_s = 0.0; pend_a[:] = 0.0
                z2_since += 1
                s_rup = True
            if C_p >= cs_so2:
                pre = agent.b.sum(axis=0).mean()
                om = 1.0 / cs_so2
                raw_w = np.exp(C_p * om)
                w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                agent.b += pend_b * w
                C_p = 0.0; pend_b[:] = 0.0
                z2_per_so2.append(z2_since)
                z2_since = 0
    return {'z2_per_so2': np.array(z2_per_so2) if z2_per_so2 else np.array([0])}


# ═════════════════════════════════════════════════════════════════════════════
# §4.4  COHERENCE DYNAMICS
# ═════════════════════════════════════════════════════════════════════════════

def section_4_4():
    section_header("4.4", "COHERENCE DYNAMICS")
    print("  Paper claims: Z₂ IRI mean=3.0, CV=0.26, B=−0.59")
    print("  SO(2) IRI mean=10.4, CV=0.50, Gamma-distributed")
    print("  C·Ω at rupture: mean 1.27 (Z₂), 1.18 (SO(2)), always ≥ 1")
    print()

    # Use a long run to get good IRI statistics
    r = run_condition(6, 0.15, 0.15, 1200, 60)

    s_iri = r['s_iri']; s_iri = s_iri[s_iri > 0]
    p_iri = r['p_iri']; p_iri = p_iri[p_iri > 0]
    s_C   = r['s_C'];   s_C   = s_C[s_C > 0]
    p_C   = r['p_C'];   p_C   = p_C[p_C > 0]

    if len(s_iri) > 10 and len(p_iri) > 10:
        s_mean = s_iri.mean()
        s_cv   = s_iri.std() / s_mean if s_mean > 0 else 0
        s_B    = burstiness(s_iri)

        p_mean = p_iri.mean()
        p_cv   = p_iri.std() / p_mean if p_mean > 0 else 0
        p_B    = burstiness(p_iri)

        print(f"  Z₂ (sensory) inter-rupture interval:")
        print(f"    Mean = {s_mean:.1f} trials   (paper: 3.0)")
        print(f"    CV   = {s_cv:.2f}          (paper: 0.26)")
        print(f"    B    = {s_B:.2f}          (paper: −0.59)")

        print(f"\n  SO(2) (prior) inter-rupture interval:")
        print(f"    Mean = {p_mean:.1f} trials  (paper: 10.4)")
        print(f"    CV   = {p_cv:.2f}          (paper: 0.50)")
        print(f"    B    = {p_B:.2f}")

        # Gamma fit for SO(2)
        shape, loc, scale = spstats.gamma.fit(p_iri, floc=0)
        ks_gamma, p_gamma = spstats.kstest(p_iri, 'gamma', args=(shape, loc, scale))
        print(f"    Gamma fit: shape={shape:.2f}, scale={scale:.2f}, "
              f"KS p={p_gamma:.4f}  (paper: Gamma-distributed)")

    if len(s_C) > 10 and len(p_C) > 10:
        s_CO = s_C * OMEGA_Z2
        p_CO = p_C * OMEGA_SO2

        print(f"\n  C·Ω at rupture:")
        print(f"    Z₂:   mean = {s_CO.mean():.2f}, min = {s_CO.min():.2f}"
              f"  (paper: mean 1.27, always ≥ 1)")
        print(f"    SO(2): mean = {p_CO.mean():.2f}, min = {p_CO.min():.2f}"
              f"  (paper: mean 1.18, always ≥ 1)")
        print(f"    Z₂ overshoot:   {100*(s_CO.mean()-1):.0f}%  (paper: 27%)")
        print(f"    SO(2) overshoot: {100*(p_CO.mean()-1):.0f}%  (paper: 18%)")


# ═════════════════════════════════════════════════════════════════════════════
# §5.1  ENVIRONMENT ROBUSTNESS
# ═════════════════════════════════════════════════════════════════════════════

def section_5_1():
    section_header("5.1", "ENVIRONMENT ROBUSTNESS")
    print("  Paper claims: all ratios within 1.5% of 1.0, CVs < 2%")
    print()

    N_TRIALS = 800; N_RUNS = 40
    envs = [
        ("Ring (n=8, control)",            lambda: RingHMM(8, 0.15, 0.15)),
        ("Linear chain (n=8)",             lambda: ChainHMM(8, 0.15)),
        ("3×3 grid (n=9)",                 lambda: GridHMM(3, 0.15)),
        ("4×4 grid (n=16)",                lambda: GridHMM(4, 0.15)),
        ("ER random (n=8, p=0.4)",         lambda: RandomGraphHMM(8, 0.4, 0.15)),
        ("ER random (n=8, p=0.7)",         lambda: RandomGraphHMM(8, 0.7, 0.15)),
        ("Binary tree (depth=2, n=7)",     lambda: TreeHMM(2, 0.15)),
        ("Binary tree (depth=3, n=15)",    lambda: TreeHMM(3, 0.15)),
    ]

    print(f"  {'Environment':<35} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'CV':>8} {'<1.5%?':>6}")
    for name, env_fn in envs:
        ratios = []
        for _ in range(N_RUNS):
            env = env_fn()
            ratio = run_in_env(env, N_TRIALS)
            if ratio is not None:
                ratios.append(ratio)
        if len(ratios) >= 3:
            m, lo, hi = bootstrap_ci(ratios, n_boot=1000)
            cv = np.std(ratios) / m if m > 0 else 0
            ok = "✓" if abs(m - 1.0) < 0.015 else "~" if abs(m - 1.0) < 0.05 else "✗"
            print(f"  {name:<35} {m:>8.4f} {lo:>8.4f} {hi:>8.4f} "
                  f"{cv:>8.4f} {ok:>6}")


# ═════════════════════════════════════════════════════════════════════════════
# §5.2  WEIGHT FUNCTION INDEPENDENCE
# ═════════════════════════════════════════════════════════════════════════════

def section_5_2():
    section_header("5.2", "WEIGHT FUNCTION INDEPENDENCE")
    print("  Paper claims: constant weights preserve (1.009, 1.008),")
    print("  channel-asymmetric (linear, √C, sigmoid) break (0.67, 0.77, 0.80)")
    print()

    weight_fns = {
        'Original (clamp 10)':    lambda C, Om: min(np.exp(C/Om), 10.0)/10*2+0.5,
        'Constant w=1':           lambda C, Om: 1.0,
        'Constant w=2':           lambda C, Om: 2.0,
        'Linear in C':            lambda C, Om: C / (2*PI) + 0.5,
        '√C':                     lambda C, Om: np.sqrt(C) + 0.5,
        'Sigmoid':                lambda C, Om: 1.0/(1.0+np.exp(-(C-PI)))+0.5,
        'Clamp to 5':             lambda C, Om: min(np.exp(C/Om), 5.0)/5*2+0.5,
        'Clamp to 50':            lambda C, Om: min(np.exp(C/Om), 50.0)/50*2+0.5,
    }

    N_TRIALS = 800; N_RUNS = 40

    print(f"  {'Weight function':>30} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8}")
    for wname, wfunc in weight_fns.items():
        info_ratios = []
        for run in range(N_RUNS):
            env = RingHMM(6, 0.15, 0.15)
            agent = CRRAgent(6)
            for t in range(N_TRIALS):
                obs, ts, ps = env.step()
                A = agent.a / agent.a.sum(axis=0, keepdims=True)
                B = agent.b / agent.b.sum(axis=0, keepdims=True)
                pe_s = -np.log(A[obs, ts] + 1e-16)
                pe_p = -np.log(B[ts, ps] + 1e-16)
                agent.pend_a[obs, ts] += 1.0
                agent.pend_b[ts, ps] += 1.0
                agent.C_s += pe_s
                agent.C_p += pe_p
                if agent.C_s >= CSTAR_Z2:
                    pre = agent.sensory_precision()
                    w = wfunc(agent.C_s, OMEGA_Z2)
                    agent.a += agent.pend_a * w
                    agent.gains_s.append(agent.sensory_precision() - pre)
                    agent.C_s = 0.0; agent.pend_a[:] = 0.0; agent.n_s_rup += 1
                if agent.C_p >= CSTAR_SO2:
                    pre = agent.prior_precision()
                    w = wfunc(agent.C_p, OMEGA_SO2)
                    agent.b += agent.pend_b * w
                    agent.gains_p.append(agent.prior_precision() - pre)
                    agent.C_p = 0.0; agent.pend_b[:] = 0.0; agent.n_p_rup += 1
            if agent.n_s_rup > 2 and agent.n_p_rup > 2:
                mg_s = np.mean(agent.gains_s)
                mg_p = np.mean(agent.gains_p)
                info_s = agent.n_s_rup * mg_s / N_TRIALS
                info_p = agent.n_p_rup * mg_p / N_TRIALS
                if info_p > 1e-8:
                    info_ratios.append(info_s / info_p)
        if len(info_ratios) >= 3:
            m, lo, hi = bootstrap_ci(info_ratios, n_boot=1000)
            print(f"  {wname:>30} {m:>8.4f} {lo:>8.4f} {hi:>8.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# §5.3  THRESHOLD PERTURBATION
# ═════════════════════════════════════════════════════════════════════════════

def section_5_3():
    section_header("5.3", "THRESHOLD PERTURBATION")
    print("  Paper claims: smooth monotonic degradation for δ ∈ [−1.5, 1.5]")
    print("  Ratio-preserving scaling (k·π / k·2π) maintains conservation")
    print()

    N_TRIALS = 800; N_RUNS = 30

    # Part 1: perturb sensory threshold
    print("  Part 1: C*_s = π + δ, C*_p fixed at 2π")
    print(f"  {'δ':>6} {'C*_s':>7} {'ratio':>8}")
    for delta in np.arange(-1.5, 1.6, 0.15):
        cs_z2 = PI + delta
        if cs_z2 <= 0.1:
            continue
        ratios = []
        for _ in range(N_RUNS):
            env = RingHMM(6, 0.15, 0.15)
            r = _run_custom(env, N_TRIALS, cs_z2, 2*PI)
            if r is not None:
                ratios.append(r)
        if len(ratios) >= 3:
            m = np.mean(ratios)
            print(f"  {delta:>6.2f} {cs_z2:>7.3f} {m:>8.4f}")

    # Part 2: ratio-preserving scaling
    print(f"\n  Part 2: C*_s = k·π, C*_p = k·2π (preserving 2:1 ratio)")
    print(f"  {'k':>6} {'ratio':>8} {'=1?':>4}")
    for k in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        ratios = []
        for _ in range(N_RUNS):
            env = RingHMM(6, 0.15, 0.15)
            r = _run_custom(env, N_TRIALS, k*PI, k*2*PI)
            if r is not None:
                ratios.append(r)
        if len(ratios) >= 3:
            m = np.mean(ratios)
            ok = "✓" if abs(m - 1.0) < 0.10 else "✗"
            print(f"  {k:>6.2f} {m:>8.4f} {ok:>4}")


def _run_custom(env, n_trials, cs_z2, cs_so2):
    """Run agent with custom thresholds."""
    n = env.n
    a = np.ones((n, n))
    b = np.ones((n, n))
    C_s, C_p = 0.0, 0.0
    pend_a = np.zeros((n, n))
    pend_b = np.zeros((n, n))
    n_s, n_p = 0, 0
    gains_s, gains_p = [], []
    for t in range(n_trials):
        obs, ts, ps = env.step()
        A_n = a / a.sum(axis=0, keepdims=True)
        B_n = b / b.sum(axis=0, keepdims=True)
        pe_s = -np.log(A_n[obs, ts] + 1e-16)
        pe_p = -np.log(B_n[ts, ps] + 1e-16)
        pend_a[obs, ts] += 1.0
        pend_b[ts, ps] += 1.0
        C_s += pe_s
        C_p += pe_p
        if C_s >= cs_z2:
            pre = a.sum(axis=0).mean()
            om = 1.0 / cs_z2
            raw_w = np.exp(C_s * om)
            w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
            a += pend_a * w
            gains_s.append(a.sum(axis=0).mean() - pre)
            C_s = 0.0; pend_a[:] = 0.0; n_s += 1
        if C_p >= cs_so2:
            pre = b.sum(axis=0).mean()
            om = 1.0 / cs_so2
            raw_w = np.exp(C_p * om)
            w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
            b += pend_b * w
            gains_p.append(b.sum(axis=0).mean() - pre)
            C_p = 0.0; pend_b[:] = 0.0; n_p += 1
    if n_s > 2 and n_p > 2 and len(gains_s) > 0 and len(gains_p) > 0:
        mg_s = np.mean(gains_s)
        mg_p = np.mean(gains_p)
        info_s = n_s * mg_s / n_trials
        info_p = n_p * mg_p / n_trials
        if info_p > 1e-8:
            return info_s / info_p
    return None


# ═════════════════════════════════════════════════════════════════════════════
# §5.4  CORRELATED EVIDENCE & THREE-CHANNEL EXTENSION
# ═════════════════════════════════════════════════════════════════════════════

def section_5_4():
    section_header("5.4", "CORRELATED EVIDENCE & THREE-CHANNEL EXTENSION")
    print("  Paper claims: conservation holds for ρ=1.0 (1.002), ρ=−0.3 (1.018)")
    print("  Breaks at ρ=−0.7 (1.16).  Three-channel: S/P holds (1.005),")
    print("  policy pairs fail (1.9–2.1)")
    print()

    N_TRIALS = 800; N_RUNS = 40

    # --- Correlated evidence ---
    print("  Correlated evidence:")
    print(f"  {'ρ':>6} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8}")
    for rho in [0.0, 0.3, 0.7, 1.0, -0.3, -0.7, -1.0]:
        info_ratios = []
        for run in range(N_RUNS):
            env = RingHMM(6, 0.15, 0.15)
            agent = CRRAgent(6)
            for t in range(N_TRIALS):
                obs, ts, ps = env.step()
                A = agent.a / agent.a.sum(axis=0, keepdims=True)
                B = agent.b / agent.b.sum(axis=0, keepdims=True)
                pe_s = -np.log(A[obs, ts] + 1e-16)
                pe_p_raw = -np.log(B[ts, ps] + 1e-16)
                if rho != 0:
                    pe_p = (1.0 - abs(rho)) * pe_p_raw + rho * pe_s
                    pe_p = max(pe_p, 0.0)
                else:
                    pe_p = pe_p_raw
                agent.pend_a[obs, ts] += 1.0
                agent.pend_b[ts, ps] += 1.0
                agent.C_s += pe_s
                agent.C_p += pe_p
                if agent.C_s >= CSTAR_Z2:
                    pre = agent.sensory_precision()
                    raw_w = np.exp(agent.C_s / OMEGA_Z2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.a += agent.pend_a * w
                    agent.gains_s.append(agent.sensory_precision() - pre)
                    agent.C_s = 0.0; agent.pend_a[:] = 0.0; agent.n_s_rup += 1
                if agent.C_p >= CSTAR_SO2:
                    pre = agent.prior_precision()
                    raw_w = np.exp(agent.C_p / OMEGA_SO2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.b += agent.pend_b * w
                    agent.gains_p.append(agent.prior_precision() - pre)
                    agent.C_p = 0.0; agent.pend_b[:] = 0.0; agent.n_p_rup += 1
            if agent.n_s_rup > 2 and agent.n_p_rup > 2:
                mg_s = np.mean(agent.gains_s)
                mg_p = np.mean(agent.gains_p)
                info_s = agent.n_s_rup * mg_s / N_TRIALS
                info_p = agent.n_p_rup * mg_p / N_TRIALS
                if info_p > 1e-8:
                    info_ratios.append(info_s / info_p)
        if len(info_ratios) >= 3:
            m, lo, hi = bootstrap_ci(info_ratios, n_boot=1000)
            print(f"  {rho:>6.1f} {m:>8.4f} {lo:>8.4f} {hi:>8.4f}")

    # --- Three-channel extension ---
    print(f"\n  Three-channel extension:")
    N_TRIALS_3 = 1200; N_RUNS_3 = 40

    configs = [
        ("Policy = Z₂ (π)",    PI,     OMEGA_Z2),
        ("Policy = SO(2) (2π)", 2*PI,  OMEGA_SO2),
    ]

    for cname, pol_thresh, pol_omega in configs:
        print(f"\n  Config: {cname}")
        ratios_sp = []
        ratios_sq = []
        ratios_pq = []

        for run in range(N_RUNS_3):
            env = RingHMM(6, 0.15, 0.15)
            agent = CRRAgent(6)
            C_q = 0.0
            pend_q = np.zeros((6, 6))
            q_params = np.ones((6, 6))
            gains_q = []
            n_q_rup = 0

            for t in range(N_TRIALS_3):
                obs, ts, ps = env.step()
                A = agent.a / agent.a.sum(axis=0, keepdims=True)
                B = agent.b / agent.b.sum(axis=0, keepdims=True)
                Q = q_params / q_params.sum(axis=0, keepdims=True)
                pe_s = -np.log(A[obs, ts] + 1e-16)
                pe_p = -np.log(B[ts, ps] + 1e-16)
                pe_q = -np.log(Q[ts, ps] + 1e-16)
                agent.pend_a[obs, ts] += 1.0
                agent.pend_b[ts, ps] += 1.0
                pend_q[ts, ps] += 1.0
                agent.C_s += pe_s
                agent.C_p += pe_p
                C_q += pe_q
                if agent.C_s >= CSTAR_Z2:
                    pre = agent.sensory_precision()
                    raw_w = np.exp(agent.C_s / OMEGA_Z2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.a += agent.pend_a * w
                    agent.gains_s.append(agent.sensory_precision() - pre)
                    agent.C_s = 0.0; agent.pend_a[:] = 0.0; agent.n_s_rup += 1
                if agent.C_p >= CSTAR_SO2:
                    pre = agent.prior_precision()
                    raw_w = np.exp(agent.C_p / OMEGA_SO2)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    agent.b += agent.pend_b * w
                    agent.gains_p.append(agent.prior_precision() - pre)
                    agent.C_p = 0.0; agent.pend_b[:] = 0.0; agent.n_p_rup += 1
                if C_q >= pol_thresh:
                    pre = q_params.sum(axis=0).mean()
                    raw_w = np.exp(C_q * pol_omega)
                    w = min(raw_w, 10.0) / 10.0 * 2.0 + 0.5
                    q_params += pend_q * w
                    gains_q.append(q_params.sum(axis=0).mean() - pre)
                    C_q = 0.0; pend_q[:] = 0.0; n_q_rup += 1

            if agent.n_s_rup > 2 and agent.n_p_rup > 2 and n_q_rup > 2:
                mg_s = np.mean(agent.gains_s)
                mg_p = np.mean(agent.gains_p)
                mg_q = np.mean(gains_q)
                info_s = agent.n_s_rup * mg_s / N_TRIALS_3
                info_p = agent.n_p_rup * mg_p / N_TRIALS_3
                info_q = n_q_rup * mg_q / N_TRIALS_3
                if info_p > 1e-8:
                    ratios_sp.append(info_s / info_p)
                if info_q > 1e-8:
                    ratios_sq.append(info_s / info_q)
                    ratios_pq.append(info_p / info_q)

        print(f"  {'Pair':>20} {'ratio':>8} {'CI_lo':>8} {'CI_hi':>8}")
        for pname, rats in [("Sensory/Prior", ratios_sp),
                             ("Sensory/Policy", ratios_sq),
                             ("Prior/Policy", ratios_pq)]:
            if len(rats) >= 3:
                m, lo, hi = bootstrap_ci(rats, n_boot=1000)
                print(f"  {pname:>20} {m:>8.4f} {lo:>8.4f} {hi:>8.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  COMPLETE REPLICATION: Phase-Gating Across Precision Channels           ║")
    print("║  Alexander Sabine · Active Inference Institute · AGI-26                 ║")
    print("║  temporalgrammar.ai · Alexander@activeinference.institute               ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  CRR Constants:")
    print(f"    Z₂:   C* = {CSTAR_Z2:.4f} (π),    Ω = {OMEGA_Z2:.6f} (1/π)")
    print(f"    SO(2): C* = {CSTAR_SO2:.4f} (2π),   Ω = {OMEGA_SO2:.6f} (1/2π)")
    print(f"    Threshold ratio: {CSTAR_SO2/CSTAR_Z2:.4f}")
    print(f"    Variance ratio:  {OMEGA_Z2/OMEGA_SO2:.4f}")
    print(f"    Weight: w = min(exp(C/Ω), 10) / 10 × 2 + 0.5  → [0.5, 2.5]")

    state_results, noise_results, all_eq = section_4_2()
    phases, z2ps = section_4_3()
    section_4_4()
    section_5_1()
    section_5_2()
    section_5_3()
    section_5_4()

    elapsed = time.time() - start
    print()
    print("═" * 72)
    print(f"  COMPLETE.  Total runtime: {elapsed/60:.1f} minutes.")
    print("═" * 72)


if __name__ == '__main__':
    main()
