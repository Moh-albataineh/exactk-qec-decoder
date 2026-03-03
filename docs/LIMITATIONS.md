# Known Limitations

## Factory Infrastructure (Days 1–14)

### Sandbox Constraints (Day 1)
- **Docker required**: The sandbox runner requires Docker. No fallback for environments without Docker (e.g., HPC clusters with Singularity/Apptainer).
- **No GPU passthrough**: the sandbox does not mount GPU devices. All simulation is CPU-only, limiting throughput for large-scale generation.
- **Fixed resource limits**: CPU, memory, and PID limits are per-container parameters, not auto-tuned. Sub-optimal settings waste resources or cause premature OOM kills.
- **Timeout is hard kill**: if a job exceeds `timeout_s`, the container is killed with no graceful shutdown. Partial results are lost.

### Sharding (Day 1.5)
- **Fixed record size**: Each shard assumes uniform `det_bytes_per_shot + obs_bytes_per_shot` per record. Cross-distance or cross-observable-count data cannot share a shard.
- **No compression**: `.bin` shards are raw bitpacked data. No zstd/gzip compression is applied, leading to larger storage for high-distance codes.
- **Shard-level metadata only**: The `.meta.jsonl` records per-block (per-run) metadata. Individual shot-level provenance is not tracked.

### Batch Orchestrator (Day 2)
- **Sequential execution**: The orchestrator runs one Docker job at a time. No parallel job scheduling or distributed execution across multiple machines.
- **Single SQLite DB**: All metadata is in one SQLite file. No support for concurrent writers or distributed databases. The DB can become a bottleneck for very large campaigns.
- **Pack definitions are YAML-only**: No programmatic pack definition API. Complex parameter sweeps require manual YAML editing.

### Sinter QC (Day 3)
- **MWPM-only QC**: Threshold estimation uses PyMatching (MWPM) as the reference decoder. QC does not evaluate ML decoders, so there's no built-in feedback loop between ML performance and data generation.
- **No multi-decoder QC**: only one decoder type is used for quality checks. A pack could be "QC-pass" for MWPM but still uninformative for an ML decoder.

### Pack Building (Days 5–9)
- **Text-level injection**: Noise injection operates on Stim circuit source code as text, not on the compiled circuit object. This is fragile to Stim output format changes and doesn't validate physical correctness of injected noise.
- **Idle qubit tracking heuristic**: Qubits are considered "idle" if not mentioned between TICK boundaries. This doesn't account for qubits that are intentionally left idle in some circuit designs.
- **No error budget accounting**: The noise injection does not verify that injected noise stays within physical constraints (e.g., total error probabilities ≤ 1) — this is delegated to `PauliChannel.clamp()` at construction time.

### Pack Catalog + Release (Days 11–12)
- **Preflight is static**: Preflight checks validate file existence and format, not statistical properties of the data. A pack could pass preflight but contain degenerate data.
- **No versioned releases**: Released packs are frozen but not version-controlled. If a release is found to have issues, there's no mechanism to publish a corrected version with the same pack name.
- **Reproduce scripts are best-effort**: Generated `reproduce.ps1`/`reproduce.sh` scripts assume the same Docker image and Stim version are available. No pinning of Docker image hashes.

---

## Noise Model Constraints

### Pauli-Only Noise
All noise models in this factory are restricted to independent Pauli channels, with one exception: the `correlated_crosstalk_like` model (Day 30) adds `CORRELATED_ERROR` XX errors on 2Q gate pairs. This excludes:
- **Non-Markovian noise** (e.g., 1/f noise, drift)
- **Amplitude damping** and other non-Pauli channels
- **Spatially extended correlations** beyond adjacent gate pairs

While Pauli noise is the standard assumption in most QEC threshold studies, real hardware exhibits significant correlations that can shift thresholds.

### Limited Noise Model Diversity
Five noise models implemented: `baseline_symmetric`, `biased_z`, `sd6_like`, `si1000_like`, `correlated_crosstalk_like`. Real hardware noise profiles are considerably more diverse and complex.

### Correlated Noise Approximation (Day 30)
The `correlated_crosstalk_like` model uses Stim's `CORRELATED_ERROR` instruction after DEPOLARIZE2 lines. This is a pragmatic approximation:
- Correlations are **pairwise** (XX on gate-pair qubits only), not multi-qubit
- Correlation pattern is **uniform** across all 2Q gates, not spatially varying
- `corr_strength` parameter is a simple budget split, not calibrated to hardware data
- The model produces k>2 hyperedges in the DEM via `decompose_errors=True`, but these may also appear in uncorrelated models (DEPOLARIZE2 decomposition)

---

## Threshold Estimation

### Heuristic Approach
Threshold values are estimated by identifying the flip_rate crossing point between distances. The current method uses:
- Pairwise comparison of accepted samples at different `p` values
- Linear interpolation between adjacent points

This is **not** a rigorous finite-size scaling analysis. Published thresholds should use:
- Multiple distances (d = 3, 5, 7, 9+)
- Bootstrap confidence intervals
- Proper finite-size scaling collapse

### Binning Thresholds
The `p_easy` and `p_hard` binning thresholds are configurable per pack but currently use fixed defaults. Optimal values depend on code family, noise model, and distance range.

---

## Benchmark Methodology (Day 14)

### Simulation-Only
The Day 14 benchmark measures **proposer efficiency** using a synthetic acceptance oracle. It does **not** measure actual circuit simulation time, Docker container overhead, or real-world scheduler contention.

### Synthetic Oracle
Log-normal noise around `p` to simulate flip_rate variance. Real variance depends on circuit structure, distance, and shot count.

### Sample Size
500 proposals × 3 seeds — sufficient for demonstrating trends but not for publication-grade statistical analysis.

---

## Data Quality

### Pre-Day 10 Samples
Samples generated before Day 10 lack `quality_status`, `quality_reason_code`, shard-level health checks, and physics provenance fields. Marked as `legacy:*` in `sample_key`.

### QC Inconclusive
Samples with zero logical errors and fewer than `shots_max` shots are marked `inconclusive`. The adaptive shot scheduler mitigates this, but some clean-regime samples remain inconclusive.

---

## ML Data Pipeline (Days 15–20)

### Fixed Detector Count Assumption
Each shard is generated with a fixed `num_detectors` tied to (distance, rounds). The ML pipeline pads or truncates to a uniform detector count within each experiment. Cross-distance training requires careful handling of varying input dimensions.

### Block-Level Splits Only
Splits are at the shard-block level, not sample level. This prevents leakage but means:
- Train/test ratios are approximate (depend on block sizes)
- Some configurations may be underrepresented in small splits
- Cross-model split with only 1 physics group degenerates to pack+p grouping

### Y-Rate Varies with Distance
Observable flip rate scales with distance (d=3 ~10.8%, d=5 ~19.8%, d=7 ~26.3%). This creates significant class imbalance that shifts between training and testing if splits cross distances.

### Single Observable
All current experiments decode a single observable (logical Z or X flip). Multi-observable decoding is not yet supported.

---

## GNN Decoder (Days 18–25)

### Small Training Budget
All experiments use **3 training epochs** with `hidden_dim=64`. This is deliberately small for fast iteration but leaves significant performance on the table. Published GNN decoders typically train for 50–200+ epochs.

### No Learning Rate Schedule
Training uses fixed LR with no warmup, cosine decay, or plateau reduction. Adding these would likely improve convergence.

### Limited Architecture Search
Only two architectures explored (GNN v0, v1). No hyperparameter optimization (grid/random/Bayesian search) performed. Better architectures (e.g., GAT, GIN, deeper networks) likely exist.

### Single Distance Training
All Day 18–26 experiments train on **d=5 only**. Cross-distance generalization (train on d=3,5 → test on d=7) has not been evaluated. The GNN input dimension is currently tied to the detector count.

### Feature Gating Trade-offs
- `v1_nop` (no p_value) prevents shortcut learning but removes potentially useful information for in-distribution tasks
- `v1_nop_nodist` (no p_value, no distance_norm) further limits features but may help in extreme OOD scenarios
- The optimal featureset may be task-dependent

### Calibration Sensitivity
- Threshold calibration on validation set can overfit to validation distribution
- Different metrics (bal_acc, f1, bal_acc_minus_fpr) produce significantly different thresholds
- Lambda parameter in `bal_acc_minus_fpr` is sensitive: λ=0.25 causes OOD collapse, λ=0.05 is too weak for cross-model

---

## MWPM vs GNN Comparison (Day 26)

### Unfair Advantage for MWPM
MWPM has access to the **exact detector error model** (DEM) matching the noise that generated the data. The GNN must learn error correlations from data alone. This makes the comparison informative (quantifies the gap) but not fair (MWPM has privileged information).

### MWPM Evaluated on Full Dataset
MWPM is evaluated on the entire merged dataset (all p-values, all noise models), **not** on the same train/test splits as the GNN. This means:
- MWPM numbers are not directly comparable to per-split GNN numbers
- A fairer comparison would evaluate MWPM only on each split's test set (future work)

### Latency Not Directly Comparable
- **GNN time** includes training (3 epochs) + inference — not pure inference latency
- **MWPM time** is decode-only (graph build amortized)
- A fair latency comparison should measure inference-only for both decoders on the same hardware

### Single Distance Only
Day 26 benchmark runs on **d=5 only**. MWPM typically improves relative to GNN at smaller distances and degrades at larger distances (where the matching graph becomes more complex). Multi-distance benchmarking is needed.

### No GPU for GNN
All experiments run on **CPU only**. GNN inference on GPU with batched forward passes would significantly reduce the latency gap.

---

## Calibration Sweep (Day 25)

### Limited Grid
32 configurations explored — a small fraction of the possible hyperparameter space. Key dimensions not swept:
- Learning rate, hidden_dim, number of layers
- Training epochs (fixed at 3)
- pos_weight_max values

### Pareto Selection on Point Estimates
Selection uses single-run point estimates without confidence intervals. Stochastic training means results may vary across seeds. A robust selection would require multiple seeds per configuration.

### Collapse Guard Thresholds Fixed
The 0.5% / 95% thresholds for collapse detection are heuristic. Optimal thresholds depend on base rate and may need adjustment for different datasets.

---

## Reproducibility

### Seed Control
Training uses fixed seeds but Stim circuit sampling during data generation uses independent RNG. Full reproducibility requires re-using the exact same shard files.

### Hardware Dependence
Floating-point operations may produce slightly different results across CPU architectures, PyTorch versions, and Python versions. Results are reproducible within the same environment but not bit-for-bit across platforms.

### No Containerization
Experiments run in a local virtual environment. Exact dependency versions are pinned in `requirements.txt` but no Docker image is provided for guaranteed reproducibility.

---

## DEM Graph Builder (Day 27)

### Hyperedge Approximation
k>2 detector error terms are approximated via clique expansion (all pairs). This overstates connectivity and can introduce spurious correlations. For d=5, 64% of DEM terms are hyperedges (k>2). A proper hypergraph GNN would be more accurate but is significantly more complex.

> [!WARNING]
> **Clique expansion is an approximation and may significantly inflate graph connectivity and node degree.** A k-detector hyperedge produces k(k−1)/2 pairwise edges, which can dominate the graph topology at higher distances. GNN training on clique-expanded graphs should be interpreted with this caveat.

### DEM-Data Mismatch Risk
DEM graphs are rebuilt from shard metadata. If circuit parameters are slightly different from the actual circuit used during data generation (e.g., due to builder updates), the DEM graph will not match the data. This is mitigated by using the same `rebuild_stim_circuit` function for both data generation and DEM extraction.

### p-Dependent Graph Topology
While the graph *structure* (edge_index) is the same for all p-values within a given (distance, basis, noise_model), the edge *weights* change with p. This means a separate DEM graph is needed per p-value, increasing storage and computation for large p-grids.

### Boundary Node Semantics
The virtual boundary node has no physical meaning — it aggregates all single-detector error mechanisms. Its features in GNN training must be handled carefully (e.g., zero-initialized or learned embedding).

---

## GNN Decoder V2 (Day 28)

### Batch Homogeneity Constraint
All samples in a batch must share the same DEM graph (same `distance`, `basis`, `noise_model`, `p`). Cross-distance or cross-p batching is not supported in V2. This limits V2 to single-config experiments; multi-config training would require per-sample graph indexing.

### Sigmoid Saturation at Low p
For very low error rates (p < 0.005), matching weights `W = ln((1-p)/p)` are large (> 5.3), and `sigmoid(W) > 0.995`. This means all edges receive near-identical scaling, reducing the transform's discriminative power. A learnable or alternative transform (e.g., `tanh(αW)` with learned `α`) could address this.

### Boundary Node Feature Leakage
The `is_boundary` feature marks the virtual boundary node. If the GNN learns to rely heavily on this indicator, it may overfit to the boundary node rather than learning meaningful detector correlations. This is mitigated by the small feature contribution (1 of F+1 dimensions) but should be monitored.

### Clique Expansion in V2
V2 inherits the clique expansion limitation from Day 27. The inflated connectivity may cause over-smoothing in deeper GNN architectures, as the effective receptive field grows faster than with the true hypergraph.

---

## Benchmark v3 (Day 30)

### Suite D F1=100% at Low p
At p=0.001 with d=5, the error rate is so low that the test set contains almost no logical errors, making classification trivial. MWPM achieves F1=100% which is technically correct but uninformative. Higher p-values (≥0.005) or larger test sets are needed for meaningful correlated noise comparisons.

### Inference-Only Latency
The latency phase currently measures MWPM only. GNN inference-only latency requires loading a trained model checkpoint, which is not yet automated in the measurement loop.

### Suite C Multi-P Caveat
Suite C iterates over `mismatch_p_values` but currently uses the same merged dataset for all p-values rather than filtering by p-bucket. The `data_p` label is set but decoding uses the full dataset.

---

## Benchmark v3 — Day 31 (Arena v2)

### p-Grid Selection is Heuristic
The pre-scan uses 512–1024 shots per candidate p, which may not be representative for all noise models. Band boundaries (low/mid/high) are hand-tuned. The relaxation fallback (50% threshold relaxation) may select marginally informative points.

### GNN Performance on Smoke Budget
With only 3 smoke epochs and 2048 samples, GNN_V2_DEM may significantly underperform its potential. Per-p subsets further reduce training data. Results should not be interpreted as GNN's upper bound.

### Per-p Data Splitting
~~Suite D v2 filters the merged correlated dataset using a ±10% p window. If shards were generated at discrete p values that don't align with the selected grid, subsets may be very small or empty.~~
**Resolved (Day 31.5)**: Replaced with nearest-p binning (`assign_nearest_p`) and generate-on-demand fallback.

### No Global MWPM vs GNN Claim
Day 31 results are limited to the tested regimes (d=5, X basis, correlated_crosstalk_like with corr_strength=0.5). No global superiority claim is made for either decoder.

---

## Benchmark v3 — Day 31.5 (Arena Fix Pack)

### Heuristic Gate Thresholds
The Day 31.5 gates use hand-tuned thresholds:
- `corr_mass_fail=0.02`, `corr_mass_warn=0.05` for k>2 mass ratio
- `bin_fail_samples=512`, `bin_warn_samples=2048` for per-p bin minimums
- GNN collapse thresholds: PPR < 0.5% or > 95%

These values are pragmatic starting points and may need adjustment for different noise models or code distances.

### Normal-Approximation CI
The seeds+CI computation assumes normal distribution of per-seed mean deltas, using `mean ± 1.96 * std / √K`. With K=5 seeds, this approximation is rough. In the benchmark run, the CI was [−0.0062, −0.0007] — narrow enough to be significant, but bootstrap CI would provide more accurate intervals.

### On-Demand Data Generation
`generate_mini_dataset()` creates synthetic samples by running Stim simulation directly. These differ from shard data in that they bypass the full QC pipeline and have no provenance metadata. In the benchmark run, **all 5 p-values required on-demand generation** (no matching shard data), triggering the `p_bin_min_samples` FAIL gate. Results on generated data should be interpreted cautiously.

### GNN Skipped for Generated Data
GNN evaluation requires block-based metadata for train/test splitting (`split_within_model`). Generated datasets have empty meta, so GNN is skipped. In the benchmark run, all 5 p-points used generated data, meaning **no GNN per-p results were produced** in Suite D v2. GNN results are available in Suite D (old) which uses the merged correlated dataset.

### Nearest-p Binning Edge Cases
Samples very far from any grid point (relative distance > `max_distance=0.5`) are discarded. In the benchmark run, the correlated shard data did not contain samples at the selected p-values (0.01–0.04), so all bins were empty and generation was required.

---

## Day 32 — Factor-Graph Decoder

### Undertraining Risk
The factor-graph decoder v0 uses 3 bipartite MP layers and is trained within the same per-p sample budget as other decoders. For small datasets (< 2048 samples) and short epoch counts (3 epochs in smoke mode), the model may be undertrained. Improvement over GNN v2 is expected primarily in the high-p correlated regime where k>2 structure matters most.

### Generated Data Constraint
Like GNN, the factor-graph decoder can run on generated data (empty meta) via `p_override`. However, it uses a simple 80/20 random split rather than block-based splitting, which weakens leakage guarantees for shard data with physics-hash structure.

### Observable Mask Sensitivity
The readout pools only error nodes with `observable_mask=True`. If no error mechanism affects the observable (degenerate DEM), the model falls back to pooling all error nodes, which may produce poor predictions.

### Bipartite Graph Size
For large codes (d≥7) or high rounds, the number of error mechanisms (V_E) can be large (hundreds to thousands). This increases memory and compute in the bipartite MP layers proportionally. No pruning or sparsification is currently applied.

### Hash Sensitivity
The `dem_topology_hash` is computed from the canonical graph structure (nodes, edges, masks). Any change in Stim's DEM decomposition order or floating-point rounding could produce a different hash, even for physically identical circuits. The hash is used for provenance checking, not physical equivalence.

---

## Day 33 — Factor-Graph v1

### Focal Loss Precision–Recall Tradeoff
Focal loss (γ=2.0) reduces gradients for well-classified examples. This helps precision but may slow recall improvement in early training — the model learns "don't predict positive" before learning "predict positive correctly."

### F0.5 Calibration Sacrifices Recall
F0.5 weights precision 4× over recall. For applications where missing an error is costlier than a false alarm (i.e., most real QEC), this tradeoff is wrong. The user should benchmark with F1 or F2 calibration for safety-critical deployments.

### No Real-Data Benchmark Yet
V1 has been unit-tested but not benchmarked on real correlated shard data. The focal loss + F0.5 improvements are validated by unit tests on synthetic data only. Real smoke tests require correlated shards to be generated.

### Same Bipartite Architecture
V1 shares v0's bipartite MP architecture. The precision improvements come from loss + calibration only. Architectural improvements (e.g., attention, edge features, multi-scale) are not yet explored.

### Generated Data Only (Per-p)
In Suite D v2, FG v1 runs on generated data (on-demand or filtered shards) per p-value. The 80/20 random split is weaker than block-based splitting from shard metadata.

---

## Day 34–37 — Density Residualization + Diagnostics

### Density Prior Dominance
The density-only baseline (Day 36) achieves near-identical accuracy to the full model on many seeds. This suggests the model's apparent "topology gain" may largely be a density lookup, not learned error structure. Residualization (Day 37) removes the density baseline, forcing topology-only evaluation.

### Regime Lock Fragility
`RegimeLock` pins a single (distance, p, basis, corr_strength, seed) configuration. Results are highly sensitive to this regime — a model that works at p=0.04 may fail at p=0.02, and vice versa. No cross-regime validation is performed.

### TopologyGain Metric Sensitivity
TopologyGain (SliceC_clean − SliceC_scrambled) is a single-number summary of a complex phenomenon. It can be positive (topology helps), negative (topology hurts), or near zero (ambiguous). The scrambler itself (K-matched shuffling) may introduce artifacts if K-bins are too sparse.

### Single Distance, Single Basis
All Day 34–58 experiments run on d=5, X basis, p=0.04, correlated noise only. No cross-distance, cross-basis, or cross-p validation performed. Results may not generalize.

---

## Day 38–41 — Curriculum Transfer + Architecture Experiments

### Curriculum Transfer Failure (Day 38)
Training at p=0.02 then fine-tuning at p=0.04 did not improve topology gain. The learned representations at low-p may not transfer to high-p due to fundamentally different error patterns.

### BP Check-Node Architecture Failure (Day 39)
The belief-propagation check-node architecture added +0.004 to topology gain — negligible improvement for significant complexity increase.

### Recombination Fragility (Days 40–41)
Prior + residual recombination with learned alpha is unstable. Alpha sweep (Day 41) confirmed the negative result is genuine — not a hyperparameter issue.

---

## Day 42–46 — K-Leakage Diagnosis + Mitigation

### Nonlinear K Leakage (Day 42)
MLP probe (R²_MLP) detects significant nonlinear K information in the residual logit. Linear probes underestimate leakage. This means simple linear corrections (orthogonalization) may be insufficient.

### KCS Topology Collapse (Day 44)
K-Conditioned Standardization + GRL reduces R²_MLP from 0.19 to 0.003 (97% reduction) but simultaneously collapses topology (SliceC=0.451, below random baseline). The mechanism is effective at removing K but destroys useful signal.

### Tradeoff: K-Removal vs Topology (Day 45)
Tempered GRL (lower lambda) finds an intermediate point (R²=0.035, TG=+0.026) but cannot achieve both low leakage AND preserved topology. This may be a fundamental limitation of gradient-based K-removal.

### |scr| Blowup (Day 46)
Leakage penalty causes scrambled-residual norm |scr| to blow up (from ~0.06 to ~2.1). The penalty successfully reduces R² but by scaling up scrambled representations rather than removing K information.

---

## Day 47–49 — Null-Space Alignment Experiments

### 5/6 Gates but |scr| Remains (Day 47)
First 5/6 gate pass achieved (R²=0.000, TG=+0.096) but |scr|=0.20 fails the G4 gate. The remaining gate is systematically hard — it requires matching scrambled and clean residual magnitudes.

### Normalizer Gaming Exploit (Day 48)
KCS on scrambled residuals introduces a systematic -1.0 offset due to the normalizer seeing scrambled statistics. This creates an artifact that looks like G4 improvement but is actually a measurement error. Reverted null-space and per-bin penalty to operate on Z_raw.

### Bias-Free Head Catastrophe (Day 49.3)
Removing bias from the head (bias=False) combined with EMA centering and shielded null-space caused: tanh saturation, KCS collapse, |scr| values of 36–297. Multiple mechanisms interacting pathologically.

---

## Day 50 — Baseline-Centered Null-Space (G4 Solved)

### Seed Sensitivity
6/6 gate pass on seed=47000 but NOT on seeds 49000/49200. The mechanism (per-step detached baseline centering) works but success is seed-dependent. R² is 0.00 on the best seed but 0.57 on others.

### Detached Baseline Limitation
The baseline centering uses stop-gradient (detached) to prevent gradient interference. This means the centering target does not adapt to training — it snapshots the scrambled baseline at each step. If the model's representation drifts significantly, the centering becomes stale.

---

## Day 51–54 — Sigma EMA + Controller Experiments

### Sigma EMA Collapse (Days 51–52)
Sigma EMA (exponential moving average of K-bin standard deviation) collapses to the floor (sigma_floor=0.1) after a few epochs. This makes KCS standardization operate with an artificial fixed denominator rather than tracking true variance.

### Alpha Frozen (Day 51)
With only 8 epochs, the learned alpha (interpolation weight between prior and residual) stays stuck at initialization (0.500). More epochs or warmer learning rate needed but Day 52 showed that more epochs cause A_ctrl regression (6/6 → 4/6).

### Controller Phase Sensitivity (Day 54)
The 3-phase controller (warmup → active → decay) has sensitive phase boundaries. R² oscillates wildly (0 ↔ 0.6) within a single run. A 5/6 near-miss on seed 49200 suggests the mechanism could work with better tuning.

---

## Day 55 — G1 Probe Stabilization

### MLP Probe Instability
The original MLP R² probe (Day 42–54) had high run-to-run variance (range 0.59) due to SGD noise. Day 55 replaced it with Linear Ridge CV (5-fold, N=4096 ProbeSet) which is 2.7× more stable (range 0.22). This revealed that many previous "G1 pass" results were false negatives — the model has persistent linear K leakage that the MLP was masking.

### Ridge Probe Conservative
Linear Ridge only measures **linear** K leakage. If the model has nonlinear K information (which Day 42 showed it does), the linear probe underestimates total leakage. MLP telemetry is retained as a secondary diagnostic.

---

## Days 56–58 — K-Orthogonalization

### Beta Explosion (Day 56)
Initial K-orthogonalization launched with no clamping. OLS regression beta grew to ~129 (unconstrained), injecting a massive K signal INTO the representation rather than removing it. Loss reached 1e+160. Hotfixed with clamping but revealed the fundamental instability of per-batch OLS beta estimation.

### Safeguard Overcorrection (Day 57)
Magnitude cap (η=0.15) + correlation floor (corr_min=0.01) + N_min=512 + per-epoch window reset collectively disabled the mechanism. Beta stayed near zero (~0.002), and 27% of active epochs were NO_OP due to insufficient samples.

### Global Gate Incompatibility (Day 58)
The global predictive gate (R_global ≥ 0.10) requires R_global — the per-batch correlation between Z·β and k₀ — to exceed 0.10. But:
- Population-level leakage is weak (R² ≈ 0.05, |r| ≈ 0.22)
- Per-batch (B=64) correlation noise completely swamps this signal
- R_global ≈ 0.01–0.02 across all epochs → gate never opens → 100% NO_OP

### Fundamental Chicken-and-Egg
K-ortho has a bootstrap problem: β must be accurate enough for the global gate to open, but β only becomes accurate after many corrections are applied. Without corrections, the EMA accumulates statistics from an uncorrected representation, which has different K-Z structure than what would exist post-correction.

### D=1 Bottleneck
The current model outputs D=1 for logit_residual_norm (single scalar per sample). This means:
- No cross-channel diversity for robust beta estimation
- Univariate safety guards are critical but provide no redundancy
- Any gate threshold must be well-calibrated for this single dimension

---

## Day 59 — Frozen Beta (OrthoStatSet)

### Representation Drift
Frozen beta from a pre-training ProbeSet does not track the model's evolving representation. As the encoder learns, the K-Z relationship changes, making the frozen correction vector systematically wrong. G1 went the wrong direction (−56%).

### No Feedback Loop
Unlike EMA-based approaches (Day 58), frozen beta provides no mechanism to detect or correct its own staleness during training.

---

## Day 60 — Epoch-Rolling K-Ortho

### Recompute Latency
Per-epoch ProbeSet forward pass adds computation. For large ProbeSet sizes, this becomes non-trivial.

### Wrong Correction Direction
Even with epoch-fresh beta and a Do-No-Harm gate, the correction increased G1 by 40%. The fundamental K-Z coupling in this architecture may require nonlinear correction, not linear subtraction.

---

## Day 61 — ProbeSet-Synced K-Ortho + Gradient Shield

### Mixed Results
ProbeSet-synced beta achieved G1=−14% on primary arm but shield-only arm showed organic clean behavior (G1 near zero without intervention). This made it unclear whether the mechanism was actively helping or the model was naturally clean.

### Simulated-Post Limitation
The simulated-post sanity check validates beta on the ProbeSet, but ProbeSet statistics may not represent the training distribution precisely. A correction that looks safe on the ProbeSet could harm generalization.

---

## Day 62 — Shield-Only (Aligned Measurement)

### 3-Seed Selection Artifact
ShieldOnly showed −59.3% G1 reduction on 3 seeds (47000, 49000, 49200). This appeared to be a breakthrough. However, Day 63 showed this was a **selection artifact** — the 3 seeds happened to be ones where K-leakage was dominant.

---

## Day 63 — E2E Validation (10 Seeds)

### Seed Dependence
Expanded to 10 seeds, ShieldOnly wins 5 but loses 5, with aggregate G1 54.2% *worse* than Control. The gradient shield helps when K-leakage is the dominant error source but hurts when the model is naturally clean (removing useful gradient signal).

### Topology Collapse
Seed 53000 causes topology collapse under the hard gradient shield. The K-collinear gradient component on this seed carries topology-relevant information that the shield destroys.

---

## Day 64 — Adaptive Soft Gradient Shielding (Mechanism-Family Closure)

### Dose-Response Falsification
Shield strength reduced from λ=1.0 → 0.50 → 0.25 across Days 63–64. Seed 53000 topology collapse persists at **all dosage levels**. This is not an over-strength problem — even a 25% partial removal of the K-collinear gradient disrupts topology.

### Mechanism Incompatibility (**Definitive**)
The failure is attributed to mechanism incompatibility, not instrumentation error:
- ✅ Alignment: 360/360 PASS (measurement is correct)
- ✅ Gating: Adaptive gate correctly turns OFF on clean seeds
- ✅ Batch safety: var_K skip prevents noisy projections
- ❌ Projection itself: Removing any fraction of K-collinear gradient destroys topology on certain seeds

### K-Collinear Gradient Contains Topology Signal
On seed 53000, the K-collinear component of the gradient is not purely "leakage" — it carries genuine information about slice structure. The model's scalar bottleneck (D=1) means K information and topology information are entangled in the same single dimension. Any projection that removes K-correlated signal also removes topology signal.

### Gradient Shielding Closed as Research Direction
Days 62–64 collectively demonstrate that gradient-component amputation on Z_g1 is architecturally incompatible:
- Hard λ=1.0 (Day 63): FAIL — seed-dependent, topology collapse
- Soft λ=0.50 (Day 64): FAIL — still seed-dependent, same collapse
- Softer λ=0.25 (Day 64): FAIL — still seed-dependent, same collapse
- Adaptive gate (Day 64): gate works, projection still harmful

**Future K-ortho work should avoid gradient-component amputation and instead test representation-space factorization (e.g., multi-dimensional Z with explicit K-channel separation) or auxiliary nuisance-routing architectures.**

---

## Day 65 — Split Residual + Nuisance Siphon

### Siphon Architecture Limitation
The `nn.Linear(1, 2)` split head has only 4 learnable parameters (2 weights + 2 biases). This is insufficient to meaningfully factorize K and topology information from a single scalar input. The split is essentially a linear transformation that cannot separate entangled signals.

### Siphon Not Siphoning
Both channels (z_topo, z_aux) converged to near-zero K correlation. The +29.9% G1 improvement came entirely from the `Corr(z_topo, K)²` decorrelation regularizer — the architectural split added no value over Day 66's pure regularization approach.

### Topology Collapse on Seed 53000
The siphon arm collapsed topology on seed 53000 (same failure mode as gradient shielding). K-removing interventions remain seed-53000-incompatible.

---

## Day 66 — Decorrelation-Only Regularization

### Global Correlation Penalty Limitation
Squared-Pearson `Corr(z_g1, K)²` penalizes ALL Z-K correlation, including legitimate topology-linked covariance (e.g., higher syndrome count naturally correlates with higher error probability). This cannot distinguish "shortcut K-dependence" from "physics-meaningful K-dependence."

### Underpowered
Median improvement +13.6% with adaptive hysteresis — well below 30% threshold. The gate protects topology but limits the decorrelation budget to levels too low for meaningful G1 reduction.

---

## Days 67–69 — Iso-K Local Ranking (ExactK)

### Pair Density at Extreme K
ExactK (ΔK=0) produces ~47 pairs/batch on average. At extreme K values (very low or very high syndrome count), few samples have the same K, resulting in 0 pairs. The ranking signal is concentrated in the mid-K regime where most samples cluster.

### Seed-Specific Adversariality
Seed 49200 is an irreducible adversarial case at d=5 (−155.6%). The seed's natural Z-K correlation opposes the ranking objective — any amount of iso-K pressure creates a conflicting gradient. No guard (StopOnRise, RatioCapped) can fix 49200 without damaging winners. This is a ~1-in-10 prevalence.

### Margin Sensitivity
The hinge margin parameter is critical: 0.50 (Day 68) caused chronic violations → +26.6%. Reducing to 0.30 (Day 69) → +31.2%. The optimal margin depends on the natural Z-gap distribution, which varies across seeds and distances.

### λ Decay Tuning
The decay schedule `0.85^(ep-8)` was hand-tuned. Different seeds may benefit from different decay rates. No automatic tuning is implemented.

### Forward-Pass-Only Limitation
ExactK is a forward-pass auxiliary loss (no backward gradient modification). This means it can only encourage ranking through loss gradients — it cannot prevent the model from learning K shortcuts via other loss components (BCE). The mechanism is effective but not air-tight.

---

## Day 70 — d=7 OOD Generalization

### OOD Performance Gap
ExactK_Tuned_Prod achieves +31.2% at d=5 but only +23.4% at d=7 (same hyperparameters). The ~8pp regression suggests the iso-K ranking signal is diluted in the higher-dimensional representation space (336 vs 120 detectors). Distance-specific tuning may close this gap.

### Seed Adversariality is Distance-Dependent
Seed 49200 is adversarial at d=5 (−155.6%) but beneficial at d=7 (+60.8%). Seeds 51000 (−40.4%) and 55000 (−584.1%) are adversarial at d=7 but fine at d=5. Adversarial behavior cannot be predicted from lower-distance results.

### Hyperparameter Transfer NOT Validated Beyond d=7
ExactK_Tuned_Prod has been tested on d=5 and d=7 only. Performance on d=3 (trivial) or d=9+ (extreme) is unknown. The margin, λ, and decay parameters may require adjustment for very different lattice sizes.

### EarlyCutoff Limitation
Hard cutoff (λ=0 after epoch 8) removes too much signal (+5.1% vs +23.4%). The gradual decay (0.85^(ep-8)) is strictly superior. EarlyCutoff is not recommended.

### Hardware Dependency
Day 70's d=7 experiment required a RunPod cloud GPU (RTX PRO 6000, 96GB VRAM, 188GB RAM). The experiment cannot be reproduced on the development laptop (RTX 4060, 8GB VRAM, 32GB RAM) due to memory constraints on the bipartite graph forward pass.

### Memory at Scale
While Day 70 showed stable memory (~1.9GB CPU, 16.2MB CUDA steady-state), the CUDA peak allocation was 13.4GB during ProbeSet evaluation. At d=9 or d=11, this could exceed available VRAM even on the 96GB card if N_probe is not reduced.

### No Multi-Distance Training
Day 70 trains separate models per seed (same as all prior days). Cross-distance training (train on d=3,5 → test on d=7) has not been evaluated. The bipartite graph structure changes between distances, requiring architecture modifications for cross-distance support.

---

## Days 71–72 — Checkpoint Selection (Selector v1–v5)

### Single-Epoch Selection Problem
Deployment requires a single checkpoint per seed, but epoch-level G1 at d=7 has ~10× more jitter than d=5. Any single-epoch selection is inherently noisy. Rolling-median smoothing (window=3) helps but cannot eliminate single-epoch spikes.

### SliceClean Noise at d=7
SliceClean (fraction of K-slices with AUROC ≥ 0.5) is noisy at d=7 because K-bins are sparser (336 detectors → wider K range → fewer samples per bin). Selectors v1–v5 using SliceClean as a survival filter had 40–80% TOPO_FAIL rates, blocking valid epochs.

### Selector Version Sensitivity
Results change significantly across selector versions (v1: +20.6%, v3: +2.5%, v5: +34.9%). The selector is a hyperparameter with large impact on the reported deployment metric. Any comparison must specify the exact selector version.

### min-G1 Fallback Cherry-Picking
Selectors v1–v3 use min-G1 as LEAKY/TOPO_FAIL fallback. This cherry-picks noise dips where G1 happens to be near zero, making both arms converge (~0.002). min-G1 is exploitable and should NOT be used as a primary metric.

---

## Day 73 — Selector v6

### drop_slice_floor Trade-off
Dropping SliceClean from survival eliminates TOPO_FAIL (0%) but removes a safety filter. If a model's topology genuinely collapses (SliceClean < 0.5), v6 will still select that epoch. The tg_roll ≥ −0.015 floor is the last line of defense, and it only catches catastrophic topology loss.

### Thresholds Not Cross-Validated
`tau_clean=0.025`, `tau_clean_hi=0.035`, `tg_floor=-0.015` were tuned on Day 70 data (10 seeds × 3 arms). No cross-validation on independent data was performed before Day 75. Day 75 holdout validated these thresholds but on the same physical regime (d=7, p=0.04).

### CLEAN/LEAKY Asymmetry Creates Metric Paradox
CLEAN pool maximizes tg_roll, LEAKY pool minimizes g1roll. Comparing selected G1 across pools is invalid. This "Asymmetric Selection Paradox" was discovered in Day 75 and led to Selected-G1 Δ% being permanently deprecated.

---

## Day 74 — v1.0 MLOps Hardening

### JSONL Replay Ordering
JSONL replay assumes epoch records are written in order. If a crash occurs mid-epoch, the partial line is silently dropped. No checksum per line — corruption mid-file may not be detected until KPI computation.

### Progressive Checkpointing Storage
Best-model checkpoints at epochs ≥ 6 overwrite the previous best. Only the final best is retained per seed. If the selection criterion changes after training, earlier checkpoints are lost and cannot be recovered.

### fsync Latency
`os.fsync()` per epoch adds I/O latency. On network-attached storage (e.g., RunPod volumes), this can be significant. No async flushing option.

---

## Day 75 — V1.0 Holdout Validation

### Asymmetric Selection Paradox
Selector v6 uses asymmetric objectives across CLEAN and LEAKY pools: CLEAN maximizes `tg_roll`, LEAKY minimizes `g1roll`. Comparing selected G1 between arms conflates two optimization targets. The "Selected-G1 Δ%" metric is **permanently deprecated** because:
- Control often selects LEAKY pool epochs with near-zero `g1roll`
- Prod often selects CLEAN pool epochs optimized for topology, not minimum G1
- Ratios of near-zero denominators explode

### KPI-A is Informational Only
KPI-A (median `tg_roll_selected` on CLEAN seeds) is informational, not a pass/fail criterion. Prod's tg_roll is −17% vs Control at d=7 — both arms produce strong topology once clean. This reflects Prod spending optimization budget on leakage reduction (ExactK), which slightly reduces topology gain. Not a physics failure.

### Receipt Schema Fragility
Original experiment receipts stored `tg_roll=0.0` (logging artifact). Day 75.3 added `extract_required_float()` to fail loudly on missing/None fields, but any future experiment must explicitly populate all canonical receipt fields (`tg_roll_selected`, `g1roll_selected`, `g1_inst_selected`, `spike_delta`).

### Holdout Scope
Day 75 validates on a single holdout configuration: d=7, p=0.04, X basis, correlated_crosstalk_like (corr=0.5). Generalization to d=3 (trivial), d=9+ (unknown), other noise models, or cross-p is not validated.

### 10-Seed Statistical Power
10 seeds provide limited statistical power. KPI-B (leaky cohort) has only ~6 leaky seeds, making the median improvement estimate sensitive to a single seed flip. Bootstrap CIs should be computed for publication.

