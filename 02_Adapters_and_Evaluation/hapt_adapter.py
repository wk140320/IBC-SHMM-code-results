#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAPT (UCI id=341) Raw Inertial Sequence & Real Postural Transition Adapter.
Reconstructs chronological 50 Hz 6-axis inertial trials from RawData/,
enforces experiment boundary resets, separates Basic Activities (1..6) from
Postural Transitions (7..12), and extracts ground-truth transition events
with exact 50 Hz physical timestamps in seconds (`onset_sec`, `end_sec`, `center_sec`).
"""

import os
import glob
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(os.environ.get("IBC_SHMM_REPO_ROOT", Path(__file__).resolve().parents[1]))
UPG_DIR = str(REPO_ROOT)
HAPT_EXT = os.path.join(UPG_DIR, "01_Official_Real_Datasets/HAPT/extracted")
RAW_FS_HZ = 50.0

BASIC_ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

TRANSITION_ACTIVITY_NAMES = {
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}

# Explicit directional mapping (from_basic_mode_0idx, to_basic_mode_0idx) for IDs 7..12
TRANSITION_DIRECTION_MAP = {
    7: (4, 3),   # STAND_TO_SIT: 5 -> 4 (0-idx: 4 -> 3)
    8: (3, 4),   # SIT_TO_STAND: 4 -> 5 (0-idx: 3 -> 4)
    9: (3, 5),   # SIT_TO_LIE:   4 -> 6 (0-idx: 3 -> 5)
    10: (5, 3),  # LIE_TO_SIT:   6 -> 4 (0-idx: 5 -> 3)
    11: (4, 5),  # STAND_TO_LIE: 5 -> 6 (0-idx: 4 -> 5)
    12: (5, 4),  # LIE_TO_STAND: 6 -> 5 (0-idx: 5 -> 4)
}

TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27]
VAL_SUBJECTS = [28, 29, 30]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]


def load_hapt_raw_trials():
    """
    Loads all 61 raw 50 Hz 6-axis trials from HAPT RawData/ and attaches
    sample-level native labels (0=unlabeled, 1..6=basic, 7..12=postural transition)
    plus explicit transition interval events with exact 50 Hz timestamps in seconds.
    """
    raw_dir = os.path.join(HAPT_EXT, "RawData")
    labels_by_exp = {}
    with open(os.path.join(raw_dir, "labels.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split()]
            if len(parts) == 5:
                exp_id, uid, act_id, s_idx, e_idx = parts
                labels_by_exp.setdefault(exp_id, []).append((uid, act_id, s_idx, e_idx))

    acc_files = sorted(glob.glob(os.path.join(raw_dir, "acc_exp*_user*.txt")))
    trials = []
    for af in acc_files:
        bn = os.path.basename(af)
        p = bn.replace(".txt", "").split("_")
        exp_id = int(p[1].replace("exp", ""))
        user_id = int(p[2].replace("user", ""))
        gf = os.path.join(raw_dir, f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt")
        Y_acc = np.loadtxt(af, dtype=np.float64)
        Y_gyro = np.loadtxt(gf, dtype=np.float64)
        assert Y_acc.shape == Y_gyro.shape and Y_acc.shape[1] == 3
        Y_raw = np.hstack([Y_acc, Y_gyro])  # (T, 6)
        T = Y_raw.shape[0]

        native_labels = np.zeros(T, dtype=np.int32)
        basic_labels = np.full(T, -1, dtype=np.int32)  # 0..5 for IDs 1..6, -1 otherwise
        all12_labels = np.full(T, -1, dtype=np.int32)  # 0..11 for IDs 1..12, -1 otherwise

        trans_events = []
        direct_changes = []
        intervals = sorted(labels_by_exp.get(exp_id, []), key=lambda x: x[2])
        prev_basic_mode = None
        prev_basic_end = None

        for _, act_id, s_1based, e_1based in intervals:
            s0 = max(0, s_1based - 1)
            e0 = min(T, e_1based)
            native_labels[s0:e0] = act_id
            all12_labels[s0:e0] = act_id - 1
            if 1 <= act_id <= 6:
                b_mode = act_id - 1
                basic_labels[s0:e0] = b_mode
                if prev_basic_mode is not None and prev_basic_mode != b_mode:
                    has_intervening_trans = any(
                        (7 <= a_id <= 12) and (max(0, st - 1) >= prev_basic_end - 5 and min(T, en) <= s0 + 5)
                        for _, a_id, st, en in intervals
                    )
                    if not has_intervening_trans:
                        direct_changes.append({
                            "exp_id": exp_id,
                            "user_id": user_id,
                            "event_type": "direct_labeled_change",
                            "from_mode": int(prev_basic_mode),
                            "to_mode": int(b_mode),
                            "raw_50hz_onset_sample": int(s0),
                            "raw_50hz_end_sample": int(s0),
                            "onset_sample": int(s0),
                            "end_sample": int(s0),
                            "center_sample": int(s0),
                            "onset_sec": float(s0) / RAW_FS_HZ,
                            "end_sec": float(s0) / RAW_FS_HZ,
                            "center_sec": float(s0) / RAW_FS_HZ,
                        })
                prev_basic_mode = b_mode
                prev_basic_end = e0
            elif 7 <= act_id <= 12:
                from_m, to_m = TRANSITION_DIRECTION_MAP[act_id]
                end_samp = max(s0, e0 - 1)
                trans_events.append({
                    "exp_id": exp_id,
                    "user_id": user_id,
                    "event_type": "annotated_postural_transition",
                    "transition_id": int(act_id),
                    "transition_name": TRANSITION_ACTIVITY_NAMES[act_id],
                    "from_mode": int(from_m),
                    "to_mode": int(to_m),
                    "raw_50hz_onset_sample": int(s0),
                    "raw_50hz_end_sample": int(end_samp),
                    "onset_sample": int(s0),
                    "end_sample": int(end_samp),
                    "center_sample": int((s0 + end_samp) // 2),
                    "onset_sec": float(s0) / RAW_FS_HZ,
                    "end_sec": float(end_samp) / RAW_FS_HZ,
                    "center_sec": 0.5 * float(s0 + end_samp) / RAW_FS_HZ,
                    "duration_samples": int(e0 - s0),
                })

        trials.append({
            "exp_id": exp_id,
            "user_id": user_id,
            "Y_raw": Y_raw,
            "native_labels": native_labels,
            "basic_labels": basic_labels,
            "all12_labels": all12_labels,
            "trans_events": trans_events,
            "direct_changes": direct_changes,
        })
    return trials


def build_subject_concatenated_streams(trials, subject_ids):
    """
    Groups trials by subject_id in chronological exp_id order, preserving
    both raw 50 Hz sample offsets and exact 50 Hz timestamps in seconds.
    """
    by_subj = {}
    for tr in sorted(trials, key=lambda x: (x["user_id"], x["exp_id"])):
        uid = tr["user_id"]
        if uid not in subject_ids:
            continue
        by_subj.setdefault(uid, []).append(tr)

    out = {}
    for uid, tr_list in sorted(by_subj.items()):
        y_list, b_list, a12_list, nat_list = [], [], [], []
        reset_indices = []
        trans_ev_all = []
        direct_ch_all = []
        offset = 0
        trial_ids = []
        for tr in tr_list:
            reset_indices.append(offset)
            trial_ids.append(tr["exp_id"])
            T_tr = tr["Y_raw"].shape[0]
            y_list.append(tr["Y_raw"])
            b_list.append(tr["basic_labels"])
            a12_list.append(tr["all12_labels"])
            nat_list.append(tr["native_labels"])
            offset_sec = float(offset) / RAW_FS_HZ
            for ev in tr["trans_events"]:
                ev_c = dict(ev)
                ev_c["raw_50hz_onset_sample"] = int(ev.get("raw_50hz_onset_sample", ev["onset_sample"]) + offset)
                ev_c["raw_50hz_end_sample"] = int(ev.get("raw_50hz_end_sample", ev["end_sample"]) + offset)
                ev_c["onset_sample"] += offset
                ev_c["end_sample"] += offset
                ev_c["center_sample"] += offset
                ev_c["onset_sec"] = float(ev_c["raw_50hz_onset_sample"]) / RAW_FS_HZ
                ev_c["end_sec"] = float(ev_c["raw_50hz_end_sample"]) / RAW_FS_HZ
                ev_c["center_sec"] = 0.5 * (ev_c["onset_sec"] + ev_c["end_sec"])
                trans_ev_all.append(ev_c)
            for dc in tr["direct_changes"]:
                dc_c = dict(dc)
                dc_c["raw_50hz_onset_sample"] = int(dc.get("raw_50hz_onset_sample", dc["onset_sample"]) + offset)
                dc_c["raw_50hz_end_sample"] = int(dc.get("raw_50hz_end_sample", dc["end_sample"]) + offset)
                dc_c["onset_sample"] += offset
                dc_c["end_sample"] += offset
                dc_c["center_sample"] += offset
                dc_c["onset_sec"] = float(dc_c["raw_50hz_onset_sample"]) / RAW_FS_HZ
                dc_c["end_sec"] = float(dc_c["raw_50hz_end_sample"]) / RAW_FS_HZ
                dc_c["center_sec"] = dc_c["onset_sec"]
                direct_ch_all.append(dc_c)
            offset += T_tr
        out[uid] = {
            "subject_id": uid,
            "trial_ids": trial_ids,
            "Y_raw": np.vstack(y_list),
            "basic_labels": np.concatenate(b_list),
            "all12_labels": np.concatenate(a12_list),
            "native_labels": np.concatenate(nat_list),
            "reset_indices": reset_indices,
            "trans_events": trans_ev_all,
            "direct_changes": direct_ch_all,
            "sampling_rate_hz": RAW_FS_HZ,
        }
    return out


def fit_train_scaler_and_slds_params(train_streams, num_modes=6, label_key="basic_labels", reg_cov=1e-3):
    """
    Fits channel standardization (mean, std, robust MAD scale) AND linear-Gaussian SLDS
    parameters (A_j, b_j, C_j=I_6, Q_j, R_j, T, pi_init, mu_eq_x, Sigma_eq_x)
    STRICTLY on training subjects' labeled segments, respecting reset_indices.
    """
    all_y_train = np.vstack([s["Y_raw"] for s in train_streams.values()])
    mean_tr = np.mean(all_y_train, axis=0)
    std_tr = np.std(all_y_train, axis=0)
    std_tr = np.where(std_tr < 1e-8, 1.0, std_tr)

    all_y_norm = (all_y_train - mean_tr) / std_tr
    med_norm = np.median(all_y_norm, axis=0)
    mad_norm = 1.4826 * np.median(np.abs(all_y_norm - med_norm), axis=0)
    mad_norm = np.where(mad_norm < 1e-6, 1.0, mad_norm)

    m = all_y_train.shape[1]
    d = m

    A_list = np.zeros((num_modes, d, d), dtype=np.float64)
    b_list = np.zeros((num_modes, d), dtype=np.float64)
    C_list = np.repeat(np.eye(m, d, dtype=np.float64)[None, :, :], num_modes, axis=0)
    Q_list = np.zeros((num_modes, d, d), dtype=np.float64)
    R_list = np.zeros((num_modes, m, m), dtype=np.float64)
    mu_eq_x = np.zeros((num_modes, d), dtype=np.float64)
    Sigma_eq_x = np.zeros((num_modes, d, d), dtype=np.float64)
    T_counts = np.full((num_modes, num_modes), 1.0, dtype=np.float64)
    mode_sample_counts = np.zeros(num_modes, dtype=np.int64)

    y_by_mode = {j: [] for j in range(num_modes)}
    pairs_by_mode = {j: ([], []) for j in range(num_modes)}
    diff2_by_mode = {j: [] for j in range(num_modes)}

    for s in train_streams.values():
        Y_n = (s["Y_raw"] - mean_tr) / std_tr
        lbls = s[label_key]
        resets = set(s["reset_indices"])
        T_len = Y_n.shape[0]
        for t in range(T_len):
            k = int(lbls[t])
            if 0 <= k < num_modes:
                y_by_mode[k].append(Y_n[t])
                mode_sample_counts[k] += 1
                if t > 0 and (t not in resets):
                    k_prev = int(lbls[t - 1])
                    if 0 <= k_prev < num_modes:
                        T_counts[k_prev, k] += 1.0
                        if k_prev == k:
                            pairs_by_mode[k][0].append(Y_n[t - 1])
                            pairs_by_mode[k][1].append(Y_n[t])
                            if t > 1 and ((t - 1) not in resets) and int(lbls[t - 2]) == k:
                                diff2_by_mode[k].append(Y_n[t] - 2.0 * Y_n[t - 1] + Y_n[t - 2])

    for j in range(num_modes):
        if len(y_by_mode[j]) >= 10:
            Yj = np.array(y_by_mode[j], dtype=np.float64)
            mu_j = np.mean(Yj, axis=0)
            cov_j = np.cov(Yj, rowvar=False) + reg_cov * np.eye(d)
        else:
            mu_j = np.zeros(d, dtype=np.float64)
            cov_j = np.eye(d, dtype=np.float64)
        mu_eq_x[j] = mu_j
        Sigma_eq_x[j] = 0.5 * (cov_j + cov_j.T)

        X_prev_list, X_curr_list = pairs_by_mode[j]
        if len(X_prev_list) >= 20:
            Xp = np.array(X_prev_list, dtype=np.float64)
            Xc = np.array(X_curr_list, dtype=np.float64)
            Xp_c = Xp - mu_j
            Xc_c = Xc - mu_j
            gram = Xp_c.T @ Xp_c + 10.0 * np.eye(d)
            A_j = (Xc_c.T @ Xp_c) @ np.linalg.inv(gram)
            eigvals = np.linalg.eigvals(A_j)
            max_ev = float(np.max(np.abs(eigvals)))
            if max_ev > 0.96:
                A_j = A_j * (0.96 / max_ev)
            b_j = mu_j - A_j @ mu_j
            res_dyn = Xc - (Xp @ A_j.T + b_j)
            cov_res = np.cov(res_dyn, rowvar=False) + reg_cov * np.eye(d)
        else:
            A_j = 0.85 * np.eye(d)
            b_j = mu_j - A_j @ mu_j
            cov_res = 0.1 * np.eye(d)

        if len(diff2_by_mode[j]) >= 20:
            D2 = np.array(diff2_by_mode[j], dtype=np.float64)
            R_est = np.cov(D2, rowvar=False) / 6.0 + reg_cov * np.eye(m)
        else:
            R_est = 0.35 * cov_res + reg_cov * np.eye(m)

        Q_est = 0.65 * cov_res + reg_cov * np.eye(d)
        A_list[j] = A_j
        b_list[j] = b_j
        Q_list[j] = 0.5 * (Q_est + Q_est.T)
        R_list[j] = 0.5 * (R_est + R_est.T)

    T_mat = T_counts / np.sum(T_counts, axis=1, keepdims=True)
    min_off = 0.002 / max(1, num_modes - 1)
    T_mat = np.maximum(T_mat, min_off)
    T_mat = T_mat / np.sum(T_mat, axis=1, keepdims=True)

    pi_init = mode_sample_counts.astype(np.float64) + 1.0
    pi_init = pi_init / np.sum(pi_init)

    return {
        "scaler_mean": mean_tr,
        "scaler_std": std_tr,
        "robust_mad_scale_norm": mad_norm,
        "num_modes": int(num_modes),
        "state_dim": int(d),
        "obs_dim": int(m),
        "A": A_list,
        "b": b_list,
        "C": C_list,
        "Q": Q_list,
        "R": R_list,
        "T_mat": T_mat,
        "pi_init": pi_init,
        "mu_eq_x": mu_eq_x,
        "Sigma_eq_x": Sigma_eq_x,
        "mode_sample_counts": mode_sample_counts.tolist(),
    }
