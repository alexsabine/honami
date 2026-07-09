# Phase-Gating Across Precision Channels — AGI 2026 Companion Materials

Welcome! If you've arrived here from the QR code on the conference poster, this
folder contains the published paper, the poster itself, full replication code,
an interactive simulation, a companion pre-print, and two audio discussions.

**Paper citation:**

> Sabine, A.: *Phase-Gating Across Precision Channels: Topological Constraints
> on Multi-Channel Belief Update Dynamics.* In: Artificial General Intelligence
> (AGI 2026), Chapter 17. LNCS, vol. 16855. Springer Nature (2026).

**Author:** Alexander Sabine — Independent Researcher, Active Inference Institute
· [temporalgrammar.ai](https://www.temporalgrammar.ai) ·
Alexander@activeinference.institute

---

## What's in this folder

### 📄 `AGI_2026.pdf` — the published paper

The full conference paper (18 pages). It introduces the
**Coherence–Rupture–Regeneration (CRR)** framework: the proposal that every
finite, persisting system cycles through three temporal operations — it
accumulates evidence (*coherence*), reaches the limits of its current regime
(*rupture*), and reconstructs from its own history (*regeneration*). The
dynamics are governed by a single geometric parameter **Ω**, fixed by the
topology of the system's statistical manifold. Two symmetry classes yield two
thresholds with **no free parameters**: **π** for bistable (Z₂) systems and
**2π** for rotational (SO(2)) systems — a ratio of exactly 2.

The paper applies CRR to a concrete problem in multi-channel AGI architecture:
how multiple precision channels coordinate their belief updates in Active
Inference. Assigning Z₂ symmetry to sensory precision and SO(2) to prior
precision, and testing in a POMDP with Dirichlet learning, the primary finding
is **phase-gating**: the topological thresholds produce a strongly non-uniform
phase relationship between channels (χ² = 8,041) that determines whether each
update drives learning or action. The result is consistent across environment
topologies, independent of the weight function, and structurally compatible
with recent empirical findings on neuromodulatory timing (Jang et al.).
Sections: Introduction · Related Work · The CRR Framework · Experiments ·
Stress Tests · Discussion · Conclusion.

### 🖼️ `CRR_ Temporal Grammar - Phase Gating.pdf` — the conference poster

The one-page AGI 2026 poster. A visual tour of the temporal grammar: dynamic
vs. geometric CRR, low-Ω rigidity vs. high-Ω flexibility, the rupture condition
C · Ω = 1, the "holding structure" loop of exploration and safe return, and
some open ontological questions ("only the past has content — what is *now*?").
It closes with the question the paper leaves for the field: *when will AGI
learn to break safely?*

### 🐍 `CRR_PG (2).py` — replication script for the paper

A single, self-contained script that reproduces **every numerical claim in the
paper**, with sections matching the paper's sections. It implements the test
environments (ring, chain, and other HMM topologies), the two-channel CRR
dynamics with the fixed Z₂/SO(2) constants, and the phase-gating analysis.

- Requires: `numpy`, `scipy`, `matplotlib` (standard scientific Python)
- Run: `python3 "CRR_PG (2).py"`
- Runtime: ~10–15 minutes on a modern machine; fixed seed for reproducibility

### 🌐 `human_networks (2).html` — interactive simulation

The interactive Active Inference Network simulation referenced in the paper
(live version: [temporalgrammar.ai/human_networks.html](https://www.temporalgrammar.ai/human_networks.html)).
Open it in any browser — no installation needed. It animates a complete graph
of agents: each **node** is an agent whose beliefs cycle on an SO(2) manifold
(rupturing at 2π), each **edge** is a Markov blanket with bistable Z₂ dynamics
(flipping at π). You can watch the gain–frequency trade-off (many small sensory
updates vs. few large prior updates), turn up an entrainment slider to see the
2:1 frequency ratio collapse to 1:1 in a synchronisation phase transition, and
explore what changes at scale — including the quadratic wall and Dunbar's
number.

### 📄 `boundary_paper (pre-print).pdf` — companion pre-print

*"The Boundary is a Rate: Coherence–Rupture–Regeneration as a Temporal Physics
of Emptiness Realisation"* (pre-print, 18 pages). A response to Sandved-Smith,
Fields, Doctor, Laukkonen & Hohwy's result that no finite system can evidence
its own boundary. The paper accepts that result and asks what survives it when
the self/environment boundary is read not as a static holographic screen but as
a **rate** — the moving locus at which accumulated Fisher information saturates
the Cramér–Rao bound (C · Ω = 1). It embeds CRR's equations in the
quantum-reference-frame formalism and relocates "emptiness" from an epistemic
limit on measurement to an ontological claim about a present with no content of
its own. All results are consistency demonstrations under a chosen formalism,
not empirical findings.

### 🐍 `boundary.py` — reproduction suite for the pre-print

A deterministic, self-auditing script that regenerates every figure and
headline number in the boundary paper. It is built for **audit rather than
trust**: every surrogate is anchored to a specific equation in a claim ledger,
carries an explicit "evidence ceiling" stating what must *not* be inferred from
it, includes positive and null controls, and passes through validation gates
with declared tolerance bands. Fixed seeds and figure hashes make a clean run
byte-stable. Run `python3 boundary.py` to regenerate all PNGs, print the claim
ledger, and write `results.json`.

### 🎧 Audio discussions (MP3)

Two accessible, conversational audio overviews of the ideas — good for
listening on the journey home from the conference:

- **`Why AGI Needs Rhythmic Partners.mp3`** — a discussion of the AGI 2026
  paper's themes: temporal grammar, phase-gating, and why the *timing* of
  belief updates (not just their magnitude) matters for AGI architectures.
- **`Why perfect technology stunts the self.mp3`** — a discussion of the
  boundary paper's themes: rupture as a feature rather than a failure, and why
  systems that never break may never genuinely learn or grow.

---

## About `index.html` (repository root)

The root of this repository is the website for **[Honami](https://www.honami.co.uk/)**
— a UK consultancy for human + AI alignment and change, working where
contemplative practice, complex-systems science, and the craft of building
technology meet. The site is a single self-contained `index.html` (no build
step, no framework) whose hero is an animated HTML5 canvas rendering the word
*Honami* as wind moving across a field of grain. It covers the consultancy's
approach (Integration and Action · Aesthetic Compression · Playful by Design ·
Gracefully Adaptive), selected work and collaborations (Active Inference
Institute, Space Zero, Temporal Grammar, and others), selected publications
including this AGI 2026 paper, and contact details.

---

## Quick start for conference attendees

1. **Read the paper:** open `AGI_2026.pdf`.
2. **Play with the ideas:** open `human_networks (2).html` in your browser, or
   visit the live version at
   [temporalgrammar.ai/human_networks.html](https://www.temporalgrammar.ai/human_networks.html).
3. **Check the numbers:** run `CRR_PG (2).py` to reproduce every result.
4. **Go deeper:** read `boundary_paper (pre-print).pdf` and run `boundary.py`.
5. **Listen:** put on one of the MP3s.

Questions and conversations welcome: Alexander@activeinference.institute
