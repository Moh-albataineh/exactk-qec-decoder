# MLOps Policy — ExactK V1.0

## Overview

This document describes the production MLOps policy for ExactK V1.0 training and checkpoint selection.

---

## Training Pipeline

### Configuration

```
Model:           Factor-Graph v1 (bipartite message-passing)
Loss:            FocalLoss (γ=2) + ExactK iso-K hinge loss
ExactK params:   ΔK=0, λ=0.10, margin=0.30, decay=0.85^(ep−8)
Batch size:      256 effective (micro=64, grad_accum=4 for d=7)
Epochs:          12
Warmup:          5 epochs (no iso-K loss)
Active phase:    epochs 6–12
```

### JSONL Write-Ahead Logging (WAL)

Every epoch writes a JSON line to `{output_dir}/epoch_log.jsonl` with `os.fsync()`:

```python
class EpochLogger:
    def log_epoch(self, record: dict):
        line = json.dumps(record, default=_convert) + "\n"
        self.fh.write(line)
        self.fh.flush()
        os.fsync(self.fh.fileno())
```

**Why**: Crash-safe. Replay from JSONL to reconstruct full training history. No data loss beyond the in-progress epoch.

### Progressive Checkpointing

Best model checkpoint saved at epochs ≥ `active_epoch_min` (6). Only the current best is retained per seed.

---

## Selector v6 Policy

### Survival Filter

```python
surviving = [ep for ep in range(active_epoch_min, total_epochs)
             if tg_roll[ep] >= -0.015]
```

**drop_slice_floor mode**: SliceClean is NOT used for survival. Only `tg_roll ≥ −0.015`.

### Pool Assignment

| Pool | Criteria | Objective |
|------|----------|-----------|
| **CLEAN** | `g1roll ≤ 0.025` AND `g1_inst ≤ 0.035` | argmax(tg_roll) |
| **LEAKY** | Surviving epochs not in CLEAN | argmin(g1roll) |
| **TOPO_FAIL** | No surviving epochs | argmin(g1roll) from all active epochs |

### Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `tau_clean` | 0.025 | G1 rolling median threshold for CLEAN eligibility |
| `tau_clean_hi` | 0.035 | G1 instantaneous cap (spike protection) |
| `tg_floor` | −0.015 | Minimum topology gain for survival |
| `active_epoch_min` | 6 | First epoch eligible for selection |

---

## Receipt Schema

Every selection receipt **must** include these fields. Missing or `None` values raise `KeyError` via `extract_required_float()`.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `tg_roll_selected` | float | Topology gain (rolling) at selected epoch |
| `g1roll_selected` | float | G1 rolling median at selected epoch |
| `g1_inst_selected` | float | G1 instantaneous at selected epoch |
| `spike_delta` | float | `g1_inst - g1roll` at selected epoch |
| `selector_pool` | str | `"CLEAN"`, `"LEAKY"`, or `"TOPO_FAIL"` |
| `selector_version` | str | `"v6_drop_slice_floor"` |
| `chosen_epoch` | int | Selected epoch index |
| `seed` | int | Training seed |
| `arm` | str | `"Control"` or `"ExactK_Tuned_Prod"` |

### Fail-Loudly Policy

```python
def extract_required_float(receipt: dict, key: str, context: str = "") -> float:
    val = receipt.get(key)
    if val is None:
        raise KeyError(f"Required field '{key}' is missing or None{context}")
    return float(val)
```

**Never** use `.get(key, 0.0)` for critical KPI fields. Silent defaults caused the Day 75.2 KPI-A bug (`prod_median_tg_roll = 0.0`).

---

## V1.0 Release KPIs

| KPI | Definition | Target |
|-----|-----------|--------|
| Science Δ | Epoch-median G1, Prod vs Control | ≥ 20% |
| Safe Yield | % seeds in CLEAN pool | ≥ 80% |
| TOPO_FAIL | % seeds with no surviving epochs | ≤ 10% |
| KPI-A | median(tg_roll) Prod vs Control on CLEAN | informational |
| KPI-B | Epoch-median G1 improvement on leaky cohort | > 0 |
| KPI-C | Do-No-Harm: Prod G1 ≤ tau_clean_hi on clean Control seeds | 0 violations |
| Spike | max(spike_delta) on Prod CLEAN seeds | < 0.015 |

### Deprecation

**Selected-G1 Δ%** is permanently deprecated due to the Asymmetric Selection Paradox: CLEAN pool maximizes `tg_roll`, LEAKY pool minimizes `g1roll` — comparing selected G1 across pools is mathematically invalid.
