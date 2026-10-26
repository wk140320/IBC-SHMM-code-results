#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HARTH (UCI id=779) Free-Living Dual-Accelerometer Adapter.
Loads 22 subject CSVs (back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label),
supports causal decimation and Leave-One-Subject-Out (LOSO) / Prefix-Calibrated
evaluation without leakage, preserving exact physical timestamps in seconds.
"""

import os
import glob
import numpy as np
from pathlib import Path

REPO_ROOT = Path(os.environ.get("IBC_SHMM_REPO_ROOT", Path(__file__).resolve().parents[1]))
UPG_DIR = str(REPO_ROOT)
HARTH_EXT = os.path.join(UPG_DIR, "01_Official_Real_Datasets/HARTH/extracted")

HARTH_CORE6_MAP = {
    1: 0,   # walking
    3: 1,   # shuffling (standing with leg movement)
    4: 2,   # stairs (ascending) / dynamic stairs
    5: 2,   # stairs (descending) -> merged into stairs mode in 6-mode compact representation
    6: 3,   # standing
    7: 4,   # sitting
    8: 5,   # lying
}

HARTH_NATIVE12_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]
HARTH_NATIVE12_MAP = {code: idx for idx, code in enumerate(HARTH_NATIVE12_CODES)}
HARTH_NATIVE12_NAMES = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_asc",
    5: "stairs_desc",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling_sit",
    14: "cycling_stand",
    130: "cycling_sit_inact",
    140: "cycling_stand_inact",
}


def load_harth_subjects(causal_stride=5):
    """
    Loads all 22 HARTH subject CSV files at causal stride `causal_stride` from 50 Hz,
    attaching exact physical timestamps in seconds (`onset_sec`, `end_sec`, `center_sec`).
    """
    csv_files = sorted(glob.glob(os.path.join(HARTH_EXT, "**/*.csv"), recursive=True))
    subjects = {}
    fs_dec = 50.0 / float(causal_stride)
    for cp in csv_files:
        bn = os.path.basename(cp)
        subj_id = bn.replace(".csv", "")
        xs = []
        lbls_native = []
        with open(cp, "r", encoding="utf-8") as f:
            header = [c.strip() for c in f.readline().strip().split(",")]
            col_idx = [header.index(c) for c in ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]]
            lbl_idx = header.index("label")
            row_num = 0
            for line in f:
                if not line.strip():
                    continue
                if row_num % causal_stride == 0:
                    parts = line.strip().split(",")
                    xs.append([float(parts[i]) for i in col_idx])
                    lbls_native.append(int(float(parts[lbl_idx])))
                row_num += 1

        Y_raw = np.array(xs, dtype=np.float64)
        nat_arr = np.array(lbls_native, dtype=np.int32)
        T_len = Y_raw.shape[0]
        native12_labels = np.array([HARTH_NATIVE12_MAP.get(int(v), -1) for v in nat_arr], dtype=np.int32)

        switch_events = []
        for t in range(1, T_len):
            if native12_labels[t] != native12_labels[t - 1] and native12_labels[t] >= 0 and native12_labels[t - 1] >= 0:
                t_sec = float(t) / fs_dec
                switch_events.append({
                    "subject_id": subj_id,
                    "event_type": "annotated_activity_switch",
                    "from_mode": int(native12_labels[t - 1]),
                    "to_mode": int(native12_labels[t]),
                    "onset_sample": int(t),
                    "end_sample": int(t),
                    "center_sample": int(t),
                    "onset_sec": t_sec,
                    "end_sec": t_sec,
                    "center_sec": t_sec,
                })

        subjects[subj_id] = {
            "subject_id": subj_id,
            "Y_raw": Y_raw,
            "native_codes": nat_arr,
            "native12_labels": native12_labels,
            "basic_labels": native12_labels,
            "reset_indices": [0],
            "trans_events": switch_events,
            "sampling_rate_hz": fs_dec,
        }
    return subjects
