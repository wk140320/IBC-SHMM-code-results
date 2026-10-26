#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2 Controlled Semi-Synthetic Fault Injection Protocol (G6).
Strictly label-independent fault generator operating ONLY on copies of held-out
test-subject standardized observation streams.
"""

import hashlib
import numpy as np


FAULT_REGIMES = [
    {
        "regime_id": "R2_SparseSpike_2Ch_8Sigma",
        "fault_type": "sparse_partial_spike",
        "n_corrupted_channels": 2,
        "burst_len": 1,
        "magnitude_scale": 8.0,
        "event_rate": 0.015,
        "description": "Isolated 1-step partial-channel impulse on 2 of 6 channels (+/-8 train robust MAD scale, 1.5% step rate)",
    },
    {
        "regime_id": "R2_Burst4Step_SingleSensor3Ch_8Sigma",
        "fault_type": "sensor_grouped_burst",
        "n_corrupted_channels": 3,
        "burst_len": 4,
        "magnitude_scale": 8.0,
        "event_rate": 0.005,
        "description": "4-step consecutive burst on one 3-axis sensor (channels 0..2 or 3..5, +/-8 train robust MAD scale, 0.5% onset rate)",
    },
    {
        "regime_id": "R2_StepBiasDrift_20Step_2Ch_4Sigma",
        "fault_type": "step_bias_drift",
        "n_corrupted_channels": 2,
        "burst_len": 20,
        "magnitude_scale": 4.0,
        "event_rate": 0.002,
        "description": "20-step sustained step bias offset on 2 of 6 channels (+/-4 train robust MAD scale)",
    },
    {
        "regime_id": "R2_SensorFreeze_10Step_2Ch",
        "fault_type": "sample_and_hold_freeze",
        "n_corrupted_channels": 2,
        "burst_len": 10,
        "magnitude_scale": 0.0,
        "event_rate": 0.003,
        "description": "10-step sample-and-hold sensor freeze + 6-sigma spike on 2 of 6 channels",
    },
]


def inject_prefrozen_faults(Y_clean_norm, robust_mad_scale_norm, regime_cfg, seed_base=20261023, subject_id_int=1):
    """
    Injects deterministic, label-independent faults onto a copy of `Y_clean_norm`.
    Never reads activity or transition labels.
    Returns:
      - `Y_corrupted`: (T, m) float array
      - `fault_mask`: (T, m) bool array (True where corrupted)
      - `manifest_entry`: dict with SHA256s, coverage rate, and exact event count
    """
    T_len, m = Y_clean_norm.shape
    Y_out = Y_clean_norm.copy()
    fault_mask = np.zeros((T_len, m), dtype=bool)

    reg_hash_int = int(hashlib.sha256(regime_cfg["regime_id"].encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_base + 1009 * int(subject_id_int) + (reg_hash_int % 999983))

    rate = float(regime_cfg["event_rate"])
    b_len = int(regime_cfg["burst_len"])
    mag = float(regime_cfg["magnitude_scale"])
    f_type = regime_cfg["fault_type"]
    n_ch = int(regime_cfg["n_corrupted_channels"])

    t = 15  # leave initial warmup steps clean
    n_events = 0
    while t < T_len - b_len - 1:
        if rng.random() < rate:
            n_events += 1
            if f_type == "sensor_grouped_burst":
                # Choose either first 3 axes (0,1,2) or second 3 axes (3,4,5)
                sensor_grp = 0 if rng.random() < 0.5 else 1
                chs = np.array([3 * sensor_grp, 3 * sensor_grp + 1, 3 * sensor_grp + 2], dtype=np.int64)
            else:
                chs = rng.choice(m, size=min(n_ch, m), replace=False)

            signs = rng.choice([-1.0, 1.0], size=len(chs))
            for dt in range(b_len):
                idx_t = t + dt
                fault_mask[idx_t, chs] = True
                if f_type == "sample_and_hold_freeze":
                    # Freeze at t-1 plus initial spike at dt==0
                    if dt == 0:
                        Y_out[idx_t, chs] = Y_clean_norm[t - 1, chs] + signs * 6.0 * robust_mad_scale_norm[chs]
                    else:
                        Y_out[idx_t, chs] = Y_out[t, chs]
                else:
                    Y_out[idx_t, chs] = Y_clean_norm[idx_t, chs] + signs * mag * robust_mad_scale_norm[chs]
            t += b_len + 10  # minimum spacing between injected events
        else:
            t += 1

    clean_sha = hashlib.sha256(np.round(Y_clean_norm, 8).tobytes()).hexdigest()
    corr_sha = hashlib.sha256(np.round(Y_out, 8).tobytes()).hexdigest()
    step_any_fault = np.any(fault_mask, axis=1)

    return Y_out, fault_mask, {
        "regime_id": regime_cfg["regime_id"],
        "subject_id": subject_id_int,
        "n_injected_events": int(n_events),
        "corrupted_steps_count": int(np.sum(step_any_fault)),
        "corrupted_steps_pct": float(100.0 * np.mean(step_any_fault)),
        "corrupted_channel_elements_pct": float(100.0 * np.mean(fault_mask)),
        "clean_quantized_sha256": clean_sha,
        "corrupted_quantized_sha256": corr_sha,
    }
