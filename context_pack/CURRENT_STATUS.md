# Current Status — QEC Noise Factory

**Last updated**: Day 75.3 (2026-03-01) -- V1.0 Release Closure: Science +45.0%, Safe Yield 80%, KPI-A (Prod tg_roll=0.0664 vs Ctrl=0.0800, -17%), KPI-B leaky +15.1% (83% wins), 0 violations. Selected-G1 Delta% deprecated (asymmetric selector). Receipt schema fixed (extract_required_float).

## What We're Building

**QEC Noise Factory** is a research pipeline for evaluating ML-based quantum error
correction decoders under **correlated noise**. The key challenge: real quantum
hardware has spatially correlated errors that break the independence assumption
used by standard MWPM decoders.

We generate synthetic QEC datasets (surface code, circuit-level noise), train
ML decoders, and benchmark them against MWPM baselines to measure the
"correlated noise gap" — how much worse MWPM becomes when it assumes independent
errors.

## Latest Benchmark Highlights

### Day 70 — ExactK d=7 OOD Generalization ✅ GENERALIZATION_PASS (+23.4%)

**ExactK_Tuned_Prod generalizes from d=5 → d=7.** Day 69's champion config (ΔK=0, margin=0.30, λ decay 0.85^(ep-8)) tested on d=7 (336 detectors, harder lattice). Result: **+23.4% median G1 reduction (0.0274 → 0.0210)** with ZERO topology collapses across all 10 seeds. Above OOD threshold (≥20%) but below standard PASS (≥30%) — expected at harder distance. EarlyCutoff arm (+5.1%) was too aggressive (λ=0 after ep 8 removes useful signal). Seed 49200 reversed at d=7: +60.8% (was -155.6% at d=5). Memory stable, 32min runtime. Alignment: 360/360 PASS. **ExactK_Tuned_Prod is the production config for all distances.**

### Day 69 — ExactK Tuned (margin↓ + λ decay) 🏆 PASS (+31.2%)

**First forward-pass-only intervention to cross the 30% PASS threshold on full 10-seed validation.** ExactK_Tuned (ΔK=0, margin=0.30, λ decay 0.85^(ep-8)): **+31.2% median G1 reduction (0.0154 → 0.0106)** with ZERO topology collapses across all 10 seeds. Key fix: margin calibration (0.50→0.30) transformed chronic hinge violations into effective ranking signal. λ decay prevents late-epoch gradient accumulation. Gated arm (+24.0%) was too conservative. Seed 49200 remains adversarial (-155.6%) but median crosses threshold. Alignment: 360/360 PASS. **ExactK_Tuned is a production candidate.**

### Day 68 — IsoKMargin_ExactK Phase 2 (10-seed E2E) ✅ PARTIAL (+26.6%)

Full 10-seed validation of ExactK (ΔK=0). 3 arms: Control, ExactK_Base (raw Z hinge), ExactK_SafeStd (detached-scale Z hinge). B=128, max_pairs=256. **ExactK_Base: +26.6% median G1 reduction (0.0154 → 0.0113) with ZERO topology collapses across ALL 10 seeds** — first forward-pass-only intervention to survive full 10-seed validation. SafeStd: ~0% aggregate (high variance). **Scale gaming NOT detected**: var_ratio ≈ 1.0, alpha_Z_scale ≈ 1.0. Phase 1 → Phase 2 regression (+50% → +26.6%) driven by easier seeds diluting median, not mechanism failure. Core signal preserved on hard seeds (53000: +46.0%, 50000: +88.9%). Alignment: 360/360 PASS. Day 69: tuning (λ/margin adjustment), not pivot.

### Day 67 — Iso-K Local Ranking ❌ FAIL_TOPOLOGY (NearK) / ✅ ExactK discovery

Pivot from global decorrelation to local ranking within iso-K bins. Hinge margin on batch-standardized Z_g1, forward-pass auxiliary only. 3 arms: Control, IsoKMargin_ExactK (ΔK=0, λ=0.10), IsoKMargin_NearK (|ΔK|≤1, λ=0.10). Phase 1 (4 hard seeds): **NearK collapses seed 53000** (drop collapse ctrl=0.068 → arm=-0.091), hard fail. **ExactK achieves +50.0% median G1 reduction (0.0224 → 0.0112) with NO topology collapse on any seed** — best safe forward-pass-only result in the project. ExactK wins on seed 53000 too (+81.4%). Key finding: exact K matching eliminates ALL K-confounding within pairs; NearK's ΔK=1 reintroduces the same confounding that global decorrelation suffers from. Alignment: 144/144 PASS.

### Day 66 — Decorrelation-Only Regularization ❌ FAIL_TOPOLOGY (B) / PARTIAL (C)

Decisive test of squared-Pearson decorrelation on Z_g1 (no split head, no gradient surgery). 3 arms: Control, FixedLowDecorOnly (λ=0.02), AdaptiveHysteresisDecorOnly (λ=0.05 + hysteresis gate). **Key result**: FixedLowDecorOnly collapses seed 53000 (med G1 = 0.0223, -14.5% worse). **AdaptiveHysteresisDecorOnly survives ALL 10 seeds including seed 53000** (+67.5% there) — first intervention since Day 62 to avoid seed 53000 collapse. But median improvement is only +13.6% (0.0195 → 0.0169), below 30% threshold. Gate-protected decorrelation is safe but underpowered — global correlation penalty cannot distinguish shortcut K from legitimate topology signal. Alignment: 360/360 PASS.

### Day 65 — Split Residual + Nuisance Siphon ❌ FAIL_TOPOLOGY

Pivot from gradient shielding to representation-space factorization. SplitResidualHead (nn.Linear(1,2) → z_topo + z_aux) after KCS. 3 arms: Control, SplitResidual2D, SplitResidual2D_Siphon (aux K-MSE λ=0.1 + decorrelation λ=0.05). Siphon arm: med G1 = 0.0137 vs Control 0.0195 (**+29.9% improvement**, just misses 30% threshold). Split-only: med G1 = 0.0288 (**-48% worse**). Topology collapse: Split/seed=50000, Siphon/seed=53000. **Key finding**: Siphon is not siphoning — both channels have near-zero K correlation; improvement comes from decorrelation loss regularizer, not architectural factorization. Alignment: 360/360 PASS.

### Day 64 — Adaptive Soft Gradient Shielding ❌ FAIL_TOPOLOGY

Soft projection (λ=0.50 and λ=0.25) + epoch hysteresis gate + batch var_K safety does **not fix** Day 63 failure. λ50: med G1 = 0.0215 (-10.3% worse), λ25: med G1 = 0.0243 (-24.6% worse). Seed 53000 topology collapse persists at ALL λ values. Gate mechanism works correctly (clean seeds unaffected) but K-collinear projection itself is architecturally flawed. **Gradient shielding (all variants) closed as non-viable.** Alignment: 360/360 PASS.

**Dose-response falsification**: Reducing shield strength from 1.0 → 0.5 → 0.25 did not remove the topology-collapse failure mode (seed 53000), falsifying the "over-strength only" hypothesis.

**Mechanism-family closure**: Because alignment, gating, and var(K) safety all passed while the same failure persisted, the failure is attributed to mechanism incompatibility (K-collinear gradient contains task-relevant topology signal in this scalar bottleneck), not instrumentation error.

**Pivot rationale**: Future work should avoid direct gradient-component amputation on Z_g1 and instead test representation-space factorization or auxiliary nuisance-routing architectures.

### Day 63 — E2E Validation (10 Seeds) ❌ FAIL — Production Freeze Denied

Day 62's 3-seed ShieldOnly success does **not generalize**. Expanded to 10 seeds: ShieldOnly med G1_raw = **0.0301** vs Control 0.0195 (54.2% *worse*). Shield wins on 5/10 seeds (47000, 49000, 51000, 53000, 55000) but loses on 5/10. Topology collapse on seed 53000. Alignment invariant 100% PASS (240/240). **Conclusion**: gradient shielding is seed-dependent — helps when K-leakage is dominant, hurts when model is naturally clean. Production freeze denied. Next: adaptive engagement or softer projection.

### Day 62 — Shield-Only vs Shield+Beta (Aligned Measurement) ✅ SUCCESS

**Shield-only wins.** Fixed Day 61 measurement bug with single-pass `evaluate_g1_aligned()`. ShieldOnly: med G1_raw = **0.0066** (59.3% reduction vs Control 0.0163), organic clean. ShieldPlusBeta: med G1_raw = 0.0102 but G1_post = 0.0184 (beta *worsens* post scores). Alignment invariant 100% PASS (252/252). **Conclusion**: `OrthoGradientShield` alone is the correct strategy — prevention > correction. No topology collapse.

### Day 61 — ProbeSet-Synced β + Gradient Shield ⚠️✅ MIXED (Breakthrough)

ProbeSet-synced β (exact honest_g1 ridge CV) + simulated-post sanity check + optional `OrthoGradientShield`. 3 arms: Control, Primary (β only), Primary+Shield (β + gradient shield). **Primary: FAIL** (-14%, β sanity check trivially passes by construction). **Primary+Shield: PASS (organic_clean)** — gradient shield removes K-collinear gradient component during backprop, preventing K-leakage from being learned. Med G1_raw_probe = 0.0102 ≤ 0.015. **Breakthrough**: After 6 failed attempts at post-hoc correction (Days 56–60), gradient-level prevention works. Day 62 should explore shield-only strategy.

### Day 60 — Epoch-Rolling K-Ortho + Do-No-Harm Gate ❌ FAIL

Epoch-rolling β recomputed every epoch from OrthoStatSet (N=4096) + Do-No-Harm gate (R²_probe ≤ 0.015 → β=0). 2 arms: Control, Primary (η=0.15). Gate works correctly — disengages for clean models (seed 49000, ep 7–12). β adapts per epoch (no stale-beta trap). But median G1_post = 0.0228 vs Control 0.0163 → **-40%** (worse, not better). Linear K-orthogonalization remains directionally unreliable after 5 attempts (Days 56–60). Next steps: nonlinear debiasing or architectural changes.

### Day 59 — Frozen-Beta K-Orthogonalization ❌ FAIL

Frozen per-epoch β from OrthoStatSet (N=4096, `torch.no_grad()`). 3 arms: Control, Primary (η=0.15), Gentle (η=0.08). Mechanism is stable and always active (0% NO_OP), but G1 **increased** -56% (Primary) / -193% (Gentle). eff_corr_ratio ≈ 0.012 — linear correction is counterproductive. No topology collapse. **Linear K-orthogonalization exhausted after 4 attempts (Days 56–59).** Leakage is nonlinear or architectural.

### Day 58 — EMA β + Global Predictive Gate ⚠️ INVALID

Cross-epoch EMA β (α=0.05) + global gate (R_global ≥ 0.10) + magnitude cap (η=0.15). Gate never opens: R_global ≈ 0.02 because per-batch (B=64) correlation is undetectable for weak signal (population R²≈0.05). 100% NO_OP across all arms. No topology collapse.

### Day 57 — Vector K-Orthogonalization ⚠️ INVALID

Vector β + magnitude cap (η=0.15) + corr-floor (0.01) + ramp schedule. Safeguards too conservative → mechanism nearly inert (β≈0). NO_OP 27% due to N_min=512 + window reset + ramp warmup. No topology collapse. Needs relaxed corr-floor with cross-epoch accumulation.

### Day 56 — K-Orthogonalization ❌ FAIL (seed-dependent)

Rolling-window K-ortho on Z_g1 (exact G1 probe input). When it works: 84–90% G1 reduction (2 seeds hit G1 ≤ 0.01). But inconsistent — hurts on seed 49200. Initial launch aborted due to beta explosion (β≈129), hotfixed with clamping.

### Day 55 — G1 Probe Stabilization ✅ PASS

Replaced MLP R² gate with Linear Ridge CV (5-fold, N=4096 ProbeSet). 2.7x more stable. Reveals real persistent K leakage (R²≈0.02–0.07) that MLP was masking with oscillation noise.

### Day 53 — Envelope Penalty + Checkpoint Selection ❌ FAIL

R² already >0.05 at epoch 1 across all seeds — early stop kills all runs at epoch 2 before envelope penalty can act. K correlation is a cold-start problem, not drift.

### Day 52 — Fast Warmup + 16 Epochs ❌ FAIL

A_ctrl regressed 6/6→4/6 with longer training. Sigma_ema fundamentally flawed.

### Day 51 — Warmup + Sigma EMA + Freeze ❌ FAIL

Warmup drops R² leakage from 0.57→0.03 on seed 49000 (near-threshold). But alpha stays stuck at init (0.500) with only 8 epochs → topology collapses (SliceC~0.45). Need more epochs or warmer alpha.

### Day 50 — Baseline-Centered Null-Space 🟡 G4 SOLVED

**First-ever 6/6 gate pass** on C_bin/seed=47000. Per-step detached baseline centering drops |scr| from 0.06–2.1 to **0.001–0.006** across ALL seeds. G4 is definitively closed. Remaining blocker: seed-dependent K-leakage (R²=0.00 on seed 47000 but 0.57 on seed 49000) and topology.

### Day 49.3 — Bias-Free + EMA + Shielded Null ❌ FAIL (catastrophic)

3-arm × 3-seed experiment. Bias-free head saturates at tanh clamp ceiling (+5.0). EMA converges to +5.0. KCS sigma→0 causes |scr| explosion (36–297). R²=0.78–0.97. DC offset is a backbone representation problem, not head bias.

### Day 49.2 — Debiased Null-Space ❌ FAIL (seed sensitivity)

Learnable per-K-bin baseline subtraction. Baseline learned b≈+0.06 (correct direction). But seed=49200 gives |scr|≈0.15 naturally, while seed=49000→1.18 and seed=49100→0.90. Scrambled residual varies **10× across seeds**. Both arms fail topology (SliceC=0.36).

### Day 49.1 — Pre-Clamp Null-Space + Bias Kill ❌ FAIL (hypothesis disproved)

2-arm A/B: pre-clamp null-space on z_pre vs post-clamp on z_raw. **Hypothesis disproved**: z_pre ≈ +1.0 is in linear tanh regime (grad ≈ 0.96), not saturated. Pre-clamp made |scr| worse (0.90→1.06) and collapsed topology (TG=-0.019). Bias is architectural — shared weights produce constant non-zero offset for scrambled inputs.

### Day 49 — Close G4: λ_null Ramp + L1 Penalty ❌ FAIL

4-arm sweep (A_ref, B1/B2/B3 with ramp+L1+bias kill). All arms show |scr|=1.06–1.20 (target ≤0.05). Per-bin scrambled means ~−1.18 — same systematic offset as Day 48. R² regressed to 0.255 on reference arm (seed sensitivity). Training unstable on B arms.

### Day 48 — Null-Space Alignment ❌ FAIL (negative result)

KCS(real stats) on scrambled creates -1.0 offset: |scr|=0.93–1.00 (from 0.20). Reverted. Lesson: null-space must target Z_raw, not Z_used.

### Day 47 — Fix Normalizer Gaming ❌ FAIL (5/6 gates!)

Arm B: R²=**0.000**✓, SliceC=**0.641**✓, drop=**0.155**✓, TG=**+0.096**✓, stable✓. Only |scr|=0.20 fails (target ≤0.05). Fixed gate check bug masking R²=0.0.

### Day 46 — Leakage Penalty + Exact-K Scrambler ❌ FAIL (progress)

Moment-matching penalty: R²=0.028, drop=**0.109**≥0.10 (G2b✓), TG=+0.068. New blocker: |scr|=1.42 (moment-matching conflicts with null-space constraint).

### Day 45 — Tempered GRL + Topology Preservation ❌ FAIL (progress)

Topology restored at λ=0.02: SliceClean=**0.634**≥0.62, TG=+0.026, stable training. But R²=0.035>0.01 and Δscr=0.040<0.10. Leakage-topology tradeoff identified.

### Day 44 — KCS + GRL ❌ FAIL (breakthrough on leakage)

KCS+GRL eliminated nonlinear K leakage: R²_MLP **0.112→0.003** (97% reduction). But adversary too aggressive — training unstable, topology signal collapsed (slice AUROC 0.613→0.587).

### Day 43 — Null-Space + α(K) ❌ FAIL

Null-space forcing works (|res_scr| 0.43→0.03, G2✓). TG positive both arms. But nonlinear K leakage persists (R²_MLP=0.28, G1✗).

### Day 42 — Residual Leakage Diagnostics ✅ ALL YES

Topology signal EXISTS (exact-K AUROC up to 0.71). Nonlinear K leakage (R²_gap=0.095) and heteroscedastic variance (corr=-0.813) mask it. Scrambler effective.

### Day 41 — Recombination Integrity + Alpha Sweep ❌ FAIL (genuine)

Fixed Day 40 wiring bug. Integrity gates ALL PASS. Alpha sweep: best α=1.0, TG=-0.004.

| α | AUROC_f | TG_f |
|---|---------|------|
| 0.0 | 0.528 | -0.009 |
| 1.0 | 0.533 | -0.004 |

### Day 40 — Prior+Residual Recombination ❌ FAIL

Explicit prior/residual split with iso-scrambler metrics. Both arms identical.

| Arm | AUROC_f | TG_final | iso_drop |
|-----|---------|----------|----------|
| A baseline | 0.509 | -0.095 | +0.011 |
| B recombined | 0.509 | -0.095 | +0.011 |

### Day 39 — BP Check-Node ❌ FAIL

| Arm | AUROC | TG |
|-----|-------|-----|
| A baseline | 0.511 | -0.052 |
| B BP check | 0.515 | -0.048 |

### Day 38 — Curriculum Transfer ❌ FAIL

Pretrained at p=0.02 (TG=+0.042 ✅), transferred to p=0.04. All 3 arms negative.

| Arm | Strategy | TG at p=0.04 |
|-----|----------|-------------|
| T0 | Scratch | -0.042 |
| T1 | Freeze MP | -0.035 |
| T2 | Partial + scrambler | -0.037 |

### Day 37.3 — P-Sweep ✅ PASS

Topology learnable at p≤0.02, not at p≥0.03. Best: p=0.02, TG=+0.045.

### Day 37 — Density Residualization ⚠️ MIXED

TopologyGain=+0.033 (PASS) but residual-K corr=−0.747 (FAIL).

### Day 36 — Density-Only Baseline ⚠️ CRITICAL

TopologyGain = −0.005 — FG v1 is WORSE than syndrome count at d=5.

## Known Issues / Blockers

| Issue | Status | Detail |
|-------|--------|--------|
| G4 |scr| ≤ 0.05 | [SOLVED] Day 50 | Baseline centering resolves systematic offset |
| Topology not learnable at p=0.04 | [CONFIRMED] Days 38-39 | Architecture ceiling — curriculum, scrambler, BP all fail |
| Topology learnable at p≤0.02 | [PASS] Day 37.3 | TG=+0.045 at p=0.02, d=5 |
| Density leakage at d=5 | [PARTIAL] Day 35 | Parity channel improves F1 but scrambler delta ~0.05 |
| FG v1 capacity ceiling | [CONFIRMED] Day 39 | Need architecture change (deeper BP, contrastive, wider) |
| Seed sensitivity | [MITIGATED] Day 69 | ExactK_Tuned handles all seeds; 49200 is irreducible adversarial case |
| **d=7 OOD generalization** | **[PASS]** Day 70 | ExactK_Tuned_Prod: +23.4% G1 reduction, 0 collapses |
