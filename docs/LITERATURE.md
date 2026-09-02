# Literature and novelty audit

**Status:** pre-result. Audit completed 2026-09-02, before any confirmatory run.
**Purpose:** establish what is already known, so that this track does not claim it.

Every arXiv identifier below was verified against the arXiv API (title, first
author, submission date) rather than taken from a search snippet. Where a claim
about a paper's *content* matters to our positioning, the abstract was fetched
directly.

---

## 1. What we must not claim

Three claims are unavailable and are not made anywhere in this repository:

1. **"Routing to experts is new."** It is not. Sparsely-gated mixture-of-experts
   routing is a decade-old line running through Shazeer et al. (2017,
   arXiv:1701.06538), Switch Transformer (Fedus et al., 2021, arXiv:2101.03961),
   and expert-choice routing (Zhou et al., 2022, arXiv:2202.09368), and it is
   actively worked in continual learning: two-level routing grouped MoE for
   multi-domain CL (Zhou et al., 2025, arXiv:2508.07738), stable routing for MoE
   in class-incremental learning (Guo et al., 2026, arXiv:2605.17571), and even
   the removal of the router entirely (Liu et al., 2026, arXiv:2604.00801).

2. **"Using multiple memory systems is new."** It is not. Complementary learning
   systems, fast/slow weights, and multi-timescale memory are long-standing.
   Recent instances include Titans (Behrouz et al., 2024, arXiv:2501.00663),
   multi-timescale memory dynamics for continual knowledge updating (Pattichis &
   Dovrolis, 2026, arXiv:2605.05097), and "Learning, Fast and Slow" (Tiwari et
   al., 2026, arXiv:2605.12484). Lampinen et al. (2025, arXiv:2509.16189) give
   the cleanest statement of *why* the systems are complementary: episodic
   memory enables flexible reuse that parametric learning alone does not.

3. **"Deciding whether to write to memory is new."** It is not. Memory-R1 (Yan
   et al., 2025, arXiv:2508.19828) trains ADD/UPDATE/DELETE/NOOP over external
   memory with reinforcement learning. "When to Forget" (Simsek, 2026,
   arXiv:2604.12007) treats forgetting as a governance primitive. MemRouter (Hu
   et al., 2026, arXiv:2605.00356) routes among memory operations. Memoir (Jaber
   et al., 2026, arXiv:2607.20792) asks whether a model should write to memory
   while it thinks.

## 2. Adjacent clusters, and the exact distance from each

### A. Expert/module routing and modular continual learning

Progressive Neural Networks (Rusu et al., 2016, arXiv:1606.04671), PathNet
(Fernando et al., 2017, arXiv:1701.08734), and Dynamically Expandable Networks
(Yoon et al., 2017, arXiv:1708.01547) route computation to modules or grow
capacity. CN-DPM (Lee et al., 2020, arXiv:2001.00689) makes the *spawn* decision
principled and task-free via a Bayesian nonparametric criterion. Dorovatas et
al. (2026, arXiv:2603.01761) argue positionally that modular memory is the key
to continual learning agents.

**Distance.** These route *which module computes* or *when to add capacity*.
They do not decide *how durably a given experience should be written*. Their
action space is module identity; ours is persistence depth. This is why EXP-001
deliberately excludes `SPAWN_NEW_MODULE` and `UPDATE_EXISTING_MODULE_i` (see
§5) — including them would put us inside this cluster's contribution and would
confound routing benefit with added capacity.

### B. Learned plasticity, learned optimizers, selective parameter updates

Differentiable plasticity (Miconi et al., 2018, arXiv:1804.02464) and
Backpropamine (Miconi et al., 2020, arXiv:2002.10585) learn *how much* each
synapse should change, with a neuromodulatory signal gating Hebbian updates.
NeuroPlastic (Jiang et al., 2026, arXiv:2604.26297) is a plasticity-modulated
optimizer. GateRA (Ou et al., 2025, arXiv:2511.17582) gates parameter-efficient
updates at token granularity.

**Distance.** This cluster learns a *continuous, differentiable magnitude* of
change within one parametric substrate, driven by immediate loss. It does not
make a discrete allocation across substrates with different persistence and
different resource prices, and it is not trained against delayed future
utility net of storage cost. The closest conceptual overlap is that both learn
"where change should go" — but "which synapse, how much" and "which substrate,
at what durability and cost" are different questions with different action
spaces.

### C. Knowledge editing: weights vs retrieval vs context

SERAC (Mitchell et al., 2022, arXiv:2206.06520) routes a query to a
counterfactual model when a stored edit is relevant. "When to Write and When to
Suppress" (Zhang et al., 2026, arXiv:2606.14668) uses a relevance router to send
a prompt to an edit adapter or a locality adapter.

**Distance.** These route at *read* time between a memory and a model, or
between two adapters at the same depth. The write destination is fixed by the
method designer.

### D. Agent memory: learned write policies over external stores

This is the nearest cluster and deserves the most precision.

*   **Memory-R1** (Yan et al., 2025, arXiv:2508.19828): RL-trained ADD / UPDATE /
    DELETE / NOOP. One substrate (external memory); the operation varies, the
    depth does not.
*   **MemRL** (Zhang et al., 2026, arXiv:2601.03192) and **RoMeRL** (Yang et al.,
    2026, arXiv:2608.02508): learn episodic-memory *utilities* at runtime, with
    utility Q-values over experiences. Learned, utility-driven, and continual —
    but the substrate is fixed to episodic memory; what is learned is retention
    and retrieval value, not write depth.
*   **BudgetMem** (Zhang et al., 2026, arXiv:2602.06025, ICML 2026): a compact
    neural policy trained with RL performs *budget-tier* routing across memory
    modules, with explicit performance-cost control. This is genuinely
    resource-aware learned routing. **However**, its tiers are levels of
    *memory-construction compute* (method complexity, reasoning depth, module
    model size). Every write lands in the same external store; no parameters are
    updated. It answers "how much compute should I spend building this memory",
    not "which substrate should absorb this experience".
*   **Dual-Layer Agentic Memory** (Li et al., 2026, arXiv:2608.22215): the
    closest work. Cost-aware epistemic routing at the *write* phase classifies
    incoming information as non-write / write-new / write-update via a
    small-to-large model cascade, and separately consolidates external memory
    into parameters by periodic supervised fine-tuning.

    **Distance from the closest work.** Three specific gaps. (i) Their routed
    decision is whether to externalise, taken *within* the external substrate;
    the external→parametric depth transition is a **periodic schedule**, not a
    per-item routed decision. (ii) Their router is a prompted/cascaded LLM
    judging epistemic redundancy, not a policy trained from **delayed future
    task utility** under a unified budget. (iii) Cost in their setting is
    routing overhead and memory bloat; there is no single objective in which
    write cost, storage occupancy, compute, and interference-driven forgetting
    are commensurable and jointly optimised.

*   Surveys confirm the shape of the field rather than the specific gap: Huang
    et al. (2026, arXiv:2602.06052), Jiang et al. (2026, arXiv:2602.19320,
    which specifically catalogues evaluation and system limitations), Zhou et
    al. (2026, arXiv:2604.08224).

### E. Resource-aware continual learning

"Forget to Improve" (Wu et al., 2026, arXiv:2606.25115) curates on-device agent
memory by net value per byte. Dennis (2026, arXiv:2605.24657) compares
weight-based consolidation against cascading compaction. These establish that
byte-level and compute-level accounting is expected practice, which is why
resource accounting here is mandatory rather than optional.

### F. The result that most threatens this hypothesis

**Neural Subspace Reallocation** (Yoon, 2026, arXiv:2606.30067) runs a
controlled comparison between a learned RL allocation controller and a simple
similarity-based retrieval rule, and reports that the heuristic matches or beats
the learned controller. Its stated conclusion is that "the memory mechanism --
compression and similarity retrieval -- rather than a learned allocation policy,
drives continual-learning performance under fixed capacity."

This is a **negative result for the class of hypothesis this track is testing**,
from a closely related setting. Three consequences, all binding:

1. The prior on "learned allocation beats heuristic allocation" is *low*, not
   high. This repository states that explicitly and does not treat a positive
   result as the expected outcome.
2. The fixed heuristic router is promoted from a courtesy baseline to the
   **primary comparator**. Kill criterion K1 in
   [`KILL-CRITERIA.md`](KILL-CRITERIA.md) fires on failing to beat it.
3. A negative replication here would be a genuine contribution, and is
   preregistered as a publishable outcome.

### G. Sibling track

`in-c0/state-promotion` asks whether *evidence-gated promotion* between
timescales beats fixed-schedule consolidation, for a frozen small LM. Related
formulations include state commitment learning (Ding et al., 2026,
arXiv:2606.05201) and Memoir (Jaber et al., 2026, arXiv:2607.20792), whose
preregistered fast-write coupling test produced a negative result that exposed
write-volume confounds. This track is *not* that experiment: State Promotion
fixes the pathway (fast → slow) and learns *when* to move along it; this track
asks *which pathway* an experience should take at all, including not writing.

---

## 3. The surviving claim

After the audit, the defensible question is narrower than "where should learning
go?" and narrower than "can an agent route to experts":

> Can a policy learn, from **delayed future task utility under a unified
> resource budget**, to allocate each experience to a substrate of differing
> **persistence and interference character** — and is any resulting gain
> attributable to *routing*, rather than to added capacity, added compute, or
> access to privileged task identity?

Four properties jointly distinguish this from every cluster above:

| Property | MoE / modular CL | Memory-R1 / MemRL | BudgetMem | Dual-Layer | Here |
|---|---|---|---|---|---|
| Action space is *persistence depth* | no (module id) | no (operation) | no (compute tier) | partial (write/no-write) | **yes** |
| Depth transition is *routed per item*, not scheduled | n/a | n/a | n/a | no (periodic SFT) | **yes** |
| Trained from *delayed future utility* | no | yes | yes | no | **yes** |
| Unified budget over write + storage + compute + interference | no | no | partial | partial | **yes** |
| Factorial separation of routing vs capacity vs compute vs task-ID | no | no | no | no | **yes** |

No single row is novel on its own. The claim is the conjunction, plus the
controls that make the conjunction measurable — not the conjunction alone.

## 4. Novelty risk register

| Risk | Severity | Mitigation |
|---|---|---|
| A reviewer reads this as "MoE with extra steps" | high | §1 and §2A state the difference in action space; module actions are excluded from EXP-001 |
| Dual-Layer (2608.22215) is judged to already cover it | high | §2D(i–iii); our depth transition is per-item and routed, theirs is scheduled |
| BudgetMem (2602.06025) is judged to already cover it | medium | its tiers are construction-compute, not persistence; no parametric writes |
| Gains turn out to be capacity or compute, not routing | high | control arms C1/C2 in [`BASELINES.md`](BASELINES.md); kill criterion K7 |
| Learned routing does not beat the heuristic (Yoon 2026 replicates) | **expected-plausible** | preregistered as a publishable negative result; K1 |
| Toy world does not transfer to LMs | high | no LM claim is made; EXP-002 is gated (see README) |

## 5. Why EXP-001 uses four actions

The candidate action set for the track is `IGNORE`, `EPISODIC_ONLY`,
`UPDATE_LATENT`, `UPDATE_FAST`, `UPDATE_EXISTING_MODULE_i`, `PROMOTE_TO_SLOW`,
`SPAWN_NEW_MODULE`. EXP-001 uses the smallest subset that keeps the question
meaningful:

`IGNORE`, `EPISODIC`, `FAST`, `SLOW`.

*   **`IGNORE` is required.** Without it, storage cost is trivially minimised by
    never routing anywhere, the noise class has no correct answer, and the
    resource term in the objective becomes decorative.
*   **Three depths are the minimum for "depth" to be an axis.** Two substrates
    give a binary write/no-write decision, which is Memory-R1's and Dual-Layer's
    setting. Three ordered persistence levels (exact-but-evictable,
    decaying-parametric, durable-parametric) are the smallest set in which
    "deeper" is meaningful and mis-routing has *direction*.
*   **`UPDATE_LATENT` is excluded** because persistent latent state is the
    sibling track's variable; including it would confound the two.
*   **Module actions are excluded** because they add capacity, which is the
    single most dangerous confound for this claim (§4), and because they place
    the work inside cluster A.

Whether the four-action set is *justified* is not assumed. It is tested: the
benchmark is admissible only if the empirically optimal class→action mapping is
a **bijection**, i.e. every one of the four actions is uniquely best for exactly
one hidden class. See [`BENCHMARK.md`](BENCHMARK.md) §5.

---

## Verified references

All entries verified against the arXiv API on 2026-09-02.

| ID | First author | Date | Title |
|---|---|---|---|
| 1606.04671 | Rusu | 2016-06-15 | Progressive Neural Networks |
| 1701.08734 | Fernando | 2017-01-30 | PathNet: Evolution Channels Gradient Descent in Super Neural Networks |
| 1708.01547 | Yoon | 2017-08-04 | Lifelong Learning with Dynamically Expandable Networks |
| 1804.02464 | Miconi | 2018-04-06 | Differentiable plasticity: training plastic neural networks with backpropagation |
| 1902.10486 | Chaudhry | 2019-02-27 | On Tiny Episodic Memories in Continual Learning |
| 2001.00689 | Lee | 2020-01-03 | A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning |
| 2002.10585 | Miconi | 2020-02-24 | Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity |
| 2004.07211 | Buzzega | 2020-04-15 | Dark Experience for General Continual Learning: a Strong, Simple Baseline |
| 2501.00663 | Behrouz | 2024-12-31 | Titans: Learning to Memorize at Test Time |
| 2508.07738 | Zhou | 2025-08-11 | Separation and Collaboration: Two-Level Routing Grouped Mixture-of-Experts for Multi-Domain Continual Learning |
| 2508.19828 | Yan | 2025-08-27 | Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning |
| 2509.16189 | Lampinen | 2025-09-19 | Latent learning: episodic memory complements parametric learning by enabling flexible reuse of experiences |
| 2511.17582 | Ou | 2025-11-15 | GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning |
| 2601.03192 | Zhang | 2026-01-06 | MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory |
| 2602.06025 | Zhang | 2026-02-05 | Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory |
| 2602.06052 | Huang | 2026-01-14 | A Survey of Agent Memory in the Second Half: Towards Self-Evolving and Long-Horizon Agents |
| 2602.19320 | Jiang | 2026-02-22 | Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations |
| 2603.01761 | Dorovatas | 2026-03-02 | Position: Modular Memory is the Key to Continual Learning Agents |
| 2604.00801 | Liu | 2026-04-01 | Routing-Free Mixture-of-Experts |
| 2604.08224 | Zhou | 2026-04-09 | Externalization in LLM Agents: A Unified Review of Memory, Skills, Protocols and Harness Engineering |
| 2604.12007 | Simsek | 2026-04-13 | When to Forget: A Memory Governance Primitive |
| 2604.26297 | Jiang | 2026-04-29 | NeuroPlastic: A Plasticity-Modulated Optimizer for Biologically Inspired Learning Dynamics |
| 2605.00356 | Hu | 2026-05-01 | MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents |
| 2605.05097 | Pattichis | 2026-05-06 | Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics |
| 2605.12484 | Tiwari | 2026-05-12 | Learning, Fast and Slow: Towards LLMs That Adapt Continually |
| 2605.17571 | Guo | 2026-05-17 | Stable Routing for Mixture-of-Experts in Class-Incremental Learning |
| 2605.24657 | Dennis | 2026-05-23 | Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction |
| 2606.05201 | Ding | 2026-05-22 | State commitment learning: training language models to distinguish computation from memory |
| 2606.09430 | Yuan | 2026-06-08 | LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models |
| 2606.14668 | Zhang | 2026-06-12 | When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing |
| 2606.25115 | Wu | 2026-06-23 | Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory |
| 2606.30067 | Yoon | 2026-06-29 | Neural Subspace Reallocation: Continual Learning as Retrieval-Based Subspace Memory Management |
| 2607.20792 | Jaber | 2026-07-22 | Memoir: Should a Model Write to Its Memory While It Thinks? |
| 2608.02508 | Yang | 2026-08-03 | RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap in Self-Evolving Agent Memory via Reduced-Order Utility States |
| 2608.04746 | Bhandari | 2026-08-05 | Caching for the Future: Scrub Jay Episodic Memory Principles for Agent Memory Systems |
| 2608.22215 | Li | 2026-08-23 | Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation |

Cited from established literature without arXiv re-verification (widely known,
stable references): Shazeer et al. 2017 (arXiv:1701.06538); Fedus et al. 2021
(arXiv:2101.03961); Zhou et al. 2022 expert-choice routing (arXiv:2202.09368);
Kirkpatrick et al. 2017 EWC (arXiv:1612.00796); Mitchell et al. 2022 SERAC
(arXiv:2206.06520); Aljundi et al. 2019 task-free continual learning
(arXiv:1812.03596). These are marked here so that a reader knows which citations
carry machine-verified metadata and which do not.
