"""
Stable Reason Codes — Day 31.5

All gate reason codes as constants.  Never use ad-hoc strings.
"""

# ---------------------------------------------------------------------------
# Informativeness gates (Day 31 originals)
# ---------------------------------------------------------------------------
TRIVIAL_REGIME = "TRIVIAL_REGIME"
SATURATED_REGIME = "SATURATED_REGIME"
INSUFFICIENT_INFORMATIVE = "INSUFFICIENT_INFORMATIVE"
OK = "OK"

# ---------------------------------------------------------------------------
# Day 31.5: Correlation-mass gates
# ---------------------------------------------------------------------------
CORR_MASS_TOO_LOW = "CORR_MASS_TOO_LOW"          # k>2 mass ratio < threshold
CORR_TERMS_TOO_FEW = "CORR_TERMS_TOO_FEW"        # too few hyperedges

# ---------------------------------------------------------------------------
# Day 31.5: MWPM triviality probe
# ---------------------------------------------------------------------------
MWPM_TRIVIAL = "MWPM_TRIVIAL"                    # probe F1 > 0.995

# ---------------------------------------------------------------------------
# Day 31.5: p-grid data availability
# ---------------------------------------------------------------------------
P_GRID_MISSING_DATA = "P_GRID_MISSING_DATA"       # no shard data for selected p
P_GRID_GENERATED = "P_GRID_GENERATED"             # on-demand data generated

# ---------------------------------------------------------------------------
# Day 31.5: Candidate vs Selected separation
# ---------------------------------------------------------------------------
CANDIDATE_REJECTION_HIGH = "CANDIDATE_REJECTION_HIGH"
SELECTED_CONTAINS_TRIVIAL = "SELECTED_CONTAINS_TRIVIAL"
SELECTED_CONTAINS_SATURATED = "SELECTED_CONTAINS_SATURATED"
SELECTED_CORR_MASS_OK = "SELECTED_CORR_MASS_OK"

# ---------------------------------------------------------------------------
# Day 31.5: Per-p binning
# ---------------------------------------------------------------------------
P_BIN_LOW_SAMPLES = "P_BIN_LOW_SAMPLES"           # bin has < threshold samples
P_BIN_EMPTY = "P_BIN_EMPTY"                       # bin has 0 samples

# ---------------------------------------------------------------------------
# Day 31.5: Seeds + CI
# ---------------------------------------------------------------------------
ORACLE_VS_MISMATCH_INCONCLUSIVE = "ORACLE_VS_MISMATCH_INCONCLUSIVE"

# ---------------------------------------------------------------------------
# Day 31.5: GNN collapse guard
# ---------------------------------------------------------------------------
GNN_COLLAPSE_LOW = "GNN_COLLAPSE_LOW"             # pred_positive_rate < 0.5%
GNN_COLLAPSE_HIGH = "GNN_COLLAPSE_HIGH"           # pred_positive_rate > 95%

# ---------------------------------------------------------------------------
# Day 32: Factor-graph decoder gates
# ---------------------------------------------------------------------------
ERR_PROVENANCE_MISMATCH = "ERR_PROVENANCE_MISMATCH"  # DEM hash mismatch at load
FG_COLLAPSE_LOW = "FG_COLLAPSE_LOW"                  # factor-graph PPR < 0.5%
FG_COLLAPSE_HIGH = "FG_COLLAPSE_HIGH"                # factor-graph PPR > 95%
LEAKAGE_ZERO_SYNDROME = "LEAKAGE_ZERO_SYNDROME"      # model confident on all-zero
LEAKAGE_SHUFFLE_FAIL = "LEAKAGE_SHUFFLE_FAIL"        # performance unchanged after shuffle

# ---------------------------------------------------------------------------
# Day 33: Factor-graph v1 collapse guards
# ---------------------------------------------------------------------------
FG_COLLAPSE_TPR = "FG_COLLAPSE_TPR"                  # factor-graph TPR < 5%
FG_REVERSE_COLLAPSE = "FG_REVERSE_COLLAPSE"          # factor-graph FPR > 70%
METRIC_INTEGRITY_FAIL = "METRIC_INTEGRITY_FAIL"      # TPR=0 but F1>0 (key mapping bug)
HASH_MISSING = "HASH_MISSING"                        # required hash field empty

# ---------------------------------------------------------------------------
# Day 34: Ranking diagnostics + density scrambler
# ---------------------------------------------------------------------------
ERR_RANKING_COLLAPSE = "ERR_RANKING_COLLAPSE"        # AUROC < 0.65
ERR_DENSITY_LEAKAGE = "ERR_DENSITY_LEAKAGE"          # scrambler delta_auroc < 0.15
BRQL_FALLBACK = "BRQL_FALLBACK"                      # BRQL tau out of bounds, fell back

# ---------------------------------------------------------------------------
# Day 35: Local parity channel + density leakage gates
# ---------------------------------------------------------------------------
ERR_DENSITY_LEAKAGE_WARNING = "ERR_DENSITY_LEAKAGE_WARNING"  # scrambler delta < 0.10
ERR_DENSITY_LEAKAGE_FAIL = "ERR_DENSITY_LEAKAGE_FAIL"       # scrambler delta < 0.05
ERR_PARITY_NUMERICS = "ERR_PARITY_NUMERICS"                  # NaN in parity_agg

# ---------------------------------------------------------------------------
# Day 36: Density baseline + p-regime hygiene
# ---------------------------------------------------------------------------
ERR_TOPOLOGY_GAIN_WARN = "ERR_TOPOLOGY_GAIN_WARN"            # TopologyGain < 0.08
ERR_P_REGIME_MISMATCH = "ERR_P_REGIME_MISMATCH"              # shard p != target_p

# ---------------------------------------------------------------------------
# Day 37: Density residualization + truth gates
# ---------------------------------------------------------------------------
ERR_TOPOLOGYGAIN_FAIL = "ERR_TOPOLOGYGAIN_FAIL"              # TopologyGain < +0.02
ERR_ISODENSITY_AUROC_FAIL = "ERR_ISODENSITY_AUROC_FAIL"      # iso-density AUROC macro < 0.55
ERR_RESIDUAL_K_CORR_HIGH = "ERR_RESIDUAL_K_CORR_HIGH"        # |corr(residual, K)| >= 0.2

# ---------------------------------------------------------------------------
# Day 37.1: Regime lock + residual orthogonality guard
# ---------------------------------------------------------------------------
ERR_REGIME_LOCK_REQUIRED = "ERR_REGIME_LOCK_REQUIRED"        # data_source != generated
ERR_TARGET_P_MISSING = "ERR_TARGET_P_MISSING"                # p_used != target_p
ERR_STIM_REQUIRED_FOR_DECISION = "ERR_STIM_REQUIRED_FOR_DECISION"  # stim not installed

# ---------------------------------------------------------------------------
# Day 37.2: Metric integrity patch
# ---------------------------------------------------------------------------
ERR_AUROC_ORIENTATION_FLIPPED = "ERR_AUROC_ORIENTATION_FLIPPED"  # AUROC(1-p) > AUROC(p)
ERR_SCRAMBLER_K_MISMATCH = "ERR_SCRAMBLER_K_MISMATCH"            # K_clean != K_scrambled

# ---------------------------------------------------------------------------
# Day 37.3: Learnability map
# ---------------------------------------------------------------------------
ERR_TOPOLOGY_NOT_LEARNABLE = "ERR_TOPOLOGY_NOT_LEARNABLE"        # max TopologyGain <= 0

# Day 43: Null-space and recombination gates
ERR_NONLINEAR_K_LEAKAGE = "ERR_NONLINEAR_K_LEAKAGE"              # R²_MLP > 0.01
ERR_SCRAMBLER_RESIDUAL_NOT_ZERO = "ERR_SCRAMBLER_RESIDUAL_NOT_ZERO"  # mean(|res_scr|) >= 0.05
ERR_SIMPSON_INCONSISTENT = "ERR_SIMPSON_INCONSISTENT"              # TG_final < +0.01
ERR_RECOMBINATION_INCONSISTENT = "ERR_RECOMBINATION_INCONSISTENT"  # slice-global AUROC gap >= 0.06

# Day 44: KCS + GRL gates
ERR_KCS_STATS_EMPTY_BIN = "ERR_KCS_STATS_EMPTY_BIN"              # KCS bin has <2 samples
ERR_GRL_K_LEAKAGE = "ERR_GRL_K_LEAKAGE"                          # R²_MLP(K|Z_norm) > 0.01
ERR_KCS_TOPOLOGY_COLLAPSE = "ERR_KCS_TOPOLOGY_COLLAPSE"          # slice AUROC < 0.62
ERR_SIMPSON_ALIGNMENT_FAIL = "ERR_SIMPSON_ALIGNMENT_FAIL"         # |meanSlice - AUROC_final| > 0.03

# Day 53: Envelope penalty + checkpoint selection
ERR_LEAKAGE_DRIFT = "ERR_LEAKAGE_DRIFT"                          # R² increases over epochs
ERR_ENV_PENALTY_TOPOLOGY_HIT = "ERR_ENV_PENALTY_TOPOLOGY_HIT"    # SliceC/drop collapse when env penalty too strong
ERR_CHECKPOINT_SELECTION = "ERR_CHECKPOINT_SELECTION"             # best epoch logic/selection failure


