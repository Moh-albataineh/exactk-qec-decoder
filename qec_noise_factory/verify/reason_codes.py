from __future__ import annotations

# Keep reason codes as stable strings (do not rename lightly).
STATIC_SYNTAX_FAIL = "static_syntax_fail"
STATIC_IMPORT_FAIL = "static_import_fail"
SCHEMA_MISMATCH = "schema_mismatch"

SANDBOX_TIMEOUT = "heavy_run_timeout"
SANDBOX_OOM = "heavy_run_oom"
SANDBOX_SECURITY_VIOLATION = "sandbox_security_violation"
OUTPUT_QUOTA_EXCEEDED = "output_quota_exceeded"

MINI_RUN_INVALID = "mini_run_invalid"
MINI_RUN_NO_NOISE = "mini_run_no_noise"
MINI_RUN_RANDOM = "mini_run_random"
MINI_RUN_TOO_CLEAN = "mini_run_too_clean"

# Day 10 — Data Quality Gates
DOMAIN_INVALID_VALUES = "domain_invalid_values"
PATHOLOGICAL_DISTRIBUTION = "pathological_distribution"
QC_INCONCLUSIVE = "qc_inconclusive"

ACCEPTED = "accepted"
DUPLICATE_SAMPLE = "duplicate_sample"
PACK_QUOTA_FULL = "pack_quota_full"
ATTEMPT_SKIPPED = "attempt_skipped"

ALLOWED_REASON_CODES = {
    STATIC_SYNTAX_FAIL,
    STATIC_IMPORT_FAIL,
    SCHEMA_MISMATCH,
    SANDBOX_TIMEOUT,
    SANDBOX_OOM,
    SANDBOX_SECURITY_VIOLATION,
    OUTPUT_QUOTA_EXCEEDED,
    MINI_RUN_INVALID,
    MINI_RUN_NO_NOISE,
    MINI_RUN_RANDOM,
    MINI_RUN_TOO_CLEAN,
    ACCEPTED,
    DUPLICATE_SAMPLE,
    PACK_QUOTA_FULL,
    ATTEMPT_SKIPPED,
    # Day 10
    DOMAIN_INVALID_VALUES,
    PATHOLOGICAL_DISTRIBUTION,
    QC_INCONCLUSIVE,
}

def assert_reason_code(rc: str) -> None:
    if rc not in ALLOWED_REASON_CODES:
        raise ValueError(f"Unknown reason_code: {rc}")
