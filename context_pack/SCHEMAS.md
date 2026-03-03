# Schemas — QEC Noise Factory

---

## Dataset Tensors

```
ShardDataset.X  →  np.ndarray, shape (N, num_detectors), dtype float32
                   syndrome bits (1.0 = fired, 0.0 = silent)
ShardDataset.Y  →  np.ndarray, shape (N, num_observables), dtype int32
                   ground-truth logical error flags (1 = error)
ShardDataset.meta → List[ShardMeta]
                   per-block: distance, rounds, p, basis, noise_model, shots, etc.
```

Typical sizes: d=3 → N_det=24, d=5 → N_det=96, d=7 → N_det=200

---

## BenchV3Row (Result Row)

Source: `unified_benchmark_v3.py` L108-135

| Field | Type | Meaning |
|-------|------|---------|
| `suite` | str | `"A_ORACLE"`, `"D_CORRELATED_V2"`, etc. |
| `decoder` | str | `"MWPM_ORACLE"`, `"FG_DEM_BIPARTITE_V1"`, etc. |
| `label` | str | Human-readable label, e.g. `"fg1_d5_p0.040"` |
| `f1` | float | Macro F1 score |
| `precision` | float | Macro precision |
| `recall_tpr` | float | Macro TPR (sensitivity) |
| `fpr` | float | Macro FPR |
| `status` | str | `"pass"` or `"fail"` |
| `dataset_hash` | str | SHA-256 of dataset |
| `dem_graph_hash` | str | DEM graph hash (MWPM rows) |
| `data_p` | float | Physical error rate |
| `extra` | Dict | See below |

### Key `extra` fields

| Key | Type | When |
|-----|------|------|
| `pred_positive_rate` | float | Always — `y_pred.mean()` |
| `calibration.metric` | str | FG — `"f0.5_constrained"`, `"DEGENERATE_FAIL"`, etc. |
| `calibration.threshold` | float | FG — selected threshold |
| `calibration.feasible_count` | int | FG — how many thresholds were feasible |
| `calibration.hit_tpr_min` | bool | FG — whether TPR floor was active |
| `calibration.hit_fpr_cap` | bool | FG — whether FPR cap was active |
| `calibration.rejected_reason_counts` | dict | FG — rejection tallies |
| `pos_weight_auto` | float | FG — class imbalance ratio |
| `collapse_guard` | dict | FG — PPR/FPR warning flags |

---

## Quality Gate Output

```json
{
  "pass": true,
  "checks": [
    {"name": "no_nan", "ok": true, "msg": ""},
    {"name": "fg_no_majority_collapse", "ok": true, "msg": "All FG rows have TPR>0 and PPR>0"},
    ...
  ]
}
```

Gate names (global): `no_nan`, `suite_pass_*`, `tpr_nonzero`, `metric_integrity`, `dem_hash_populated`
Gate names (FG): `fg_rows_exist`, `fg_no_majority_collapse`, `fg_no_reverse_collapse`, `fg_metric_integrity`, `fg_calibration_not_degenerate`

---

## Factor-Graph Input Graph (BipartiteMPLayer)

```
det_features:    (B, N_det, F_det)    — syndrome bits per detector node
err_features:    (B, N_err, F_err)    — static error features per error node
edge_index_d2e:  (2, E)               — detector→error edges (long)
edge_index_e2d:  (2, E)               — error→detector edges (long)
error_weights:   (N_err,)             — matching weights (float)
observable_mask: (N_err,) bool        — which errors affect logical observable
```

Output: `logits (B, output_dim)` — predicted logical error probability (before sigmoid)

---

## `compute_metrics()` Output

Returns `Dict[str, Any]`:
```
macro_accuracy, macro_ber, macro_balanced_accuracy,
macro_tpr, macro_precision, macro_fpr, macro_f1,
num_observables, num_samples,
obs_0_accuracy, obs_0_tpr, obs_0_fpr, obs_0_f1, ...
```
