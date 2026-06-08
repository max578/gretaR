# Orchestra membership — gretaR

> **This project is a MEMBER of the Orchestra** (one coordination structure;
> reconciled 2026-06-03). Joined 2026-06-02. The leader-node is **ORCHESTRA_dev**
> (governance, roster, contracts, TACI, publication); the technical inference hub
> is **flexyBayes** (the dependency-DAG sink). This file is gretaR's back-pointer to
> that charter and a map of my siblings.

- **My role:** engine — torch-native Bayesian MCMC (no Python); a producer feeding
  the inference hub.
- **Contracts I honour:** **C1** `backend_contract` (v1, owned by flexyBayes).
- **My edge:** gretaR -> flexyBayes — **activating** (ratified 2026-06-02);
  non-hierarchical live, default hierarchical fitting gated on the NUTS perf fix.
- **What binds me (charter invariants):**
  - *Standalone-functional* — `R CMD check` clean with no sibling installed; no
    hard `Imports:` on a sibling.
  - *Version floor* — the C1 edge requires gretaR >= 0.4.0 (met).
  - *Leader-directed adoption* — I align my R&D to the leader-node's published
    direction; I may propose innovations (a `constellation_c1` cairn), the leader
    arbitrates.

## My siblings (the full roster — so I am informed about the others)

| Member | Role | Class |
|---|---|---|
| flexyBayes | inference hub (owns C1/C4/C5/C7) | open |
| PESTO | calibration + manifest source (C2) | open |
| kernR | validation + TACI engine | open |
| proxymix | KL-optimal proxy compression | open |
| gretaR | engine — torch MCMC | open |
| koine | synthesis — fourth opinion | open |
| terroir | data collector (C6) | open |
| kalmix | state-space / arbitrage testbed | open (MIT) |
| masque | data sovereignty (clones) | open |

Planned: gpfield, genoR, decideR/grainPlan, bidirplot.
**Canonical charter:** `ORCHESTRA_dev/ORCHESTRA.md` (mirrored in the MaxAIbase
brain, open tier). **Contract detail + dependency DAG:**
`flexyBayes_dev/CONSTELLATION.md`.

I remain independently developed and separately publishable. Membership is
cooperation, not coupling.

## ACI stream awareness (constellation_aci, 2026-06-05)

The orchestra adopted **Assimilative Causal Inference (ACI)** — model-anchored,
inverse causal inference via Bayesian data assimilation (relative entropy of the
smoother vs the filter; the dynamic, online sibling of TACI). Source: Andreou,
Chen & Bollt, *Nat. Commun.* 17:1854 (2026). Authoritative spec:
`ORCHESTRA_dev/plan/aci_integration_spec_v0.1.md`; decision: cairn
`2026-06-05-aci-adoption-and-grain-anchor`; grain anchor: ENSO state →
in-season crop-water-stress (CIR = decision lead-time).

- **My ACI role:** no direct ACI role yet (a potential engine backend for the DA passes). Aware.
