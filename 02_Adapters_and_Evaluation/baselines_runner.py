#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Baselines, Unconfounded One-Factor-at-a-Time Attribution Ablations &
Prefix-Adaptable Task-Competitor Runner (Gates G3, G4, G5, G6 & Round-20 P0-A, P0-B, P1-E).
Vectorized with `numpy.lib.stride_tricks.sliding_window_view` for fast,
100% strictly causal sliding-window feature extraction and Hampel pre-despiking.
"""

import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from strict_online_evaluator import run_strict_online_switching_filter


SWITCHING_METHODS_LIST = [
    "Standard-SKF-NoEq",
    "Standard-SKF",
    "Sat-SKF",
    "Huber-SKF-SingleLayer",
    "StudentT-IMM-1Step-NoEq",
    "StudentT-IMM-1Step",
    "StudentT-IMM-VB-ELBO",
    "Coord-StudentT-IMM-ELBO",
    "StudentT-IMM-VB-Cal",
    "Coord-StudentT-IMM",
    "IBC-PureMarginal-Ab",
    "IBC-SHMM-LegacySat",
    "IBC-SHMM-PrevStep",
    "IBC-SHMM-ObsWarmStart",
    "IBC-SHMM",
    "Hampel-ValTuned+Standard-SKF",
    "Hampel31-Fixed6+Standard-SKF",
    "Hampel-ValTuned+IBC-SHMM",
]

ABLATION_ONLY_METHODS_LIST = [
    "IBC-Ab1a-UnclipOnly",
    "IBC-Ab1b-UnclipPlusDyn050",
    "IBC-Ab1c-UnclipPlusDyn050Eq008",
    "IBC-Ab1-UnclipModeScoreOnly",
    "IBC-Ab2-UnclipPlusThresholds",
    "IBC-Ab3-PreM3StateWeight",
    "IBC-NoSubspace-Ab",
    "IBC-NoEqAnchor-Ab",
    "IBC-NoTempGate-Ab",
]

ABLATION_ATTRIBUTION_METHODS_LIST = [
    "IBC-SHMM-LegacySat",
    "IBC-Ab1a-UnclipOnly",
    "IBC-Ab1b-UnclipPlusDyn050",
    "IBC-Ab1c-UnclipPlusDyn050Eq008",
    "IBC-Ab1-UnclipModeScoreOnly",
    "IBC-Ab2-UnclipPlusThresholds",
    "IBC-SHMM",
    "IBC-Ab3-PreM3StateWeight",
    "IBC-NoSubspace-Ab",
    "IBC-NoEqAnchor-Ab",
    "IBC-NoTempGate-Ab",
    "IBC-PureMarginal-Ab",
    "IBC-SHMM-PrevStep",
    "IBC-SHMM-ObsWarmStart",
]

TASK_COMPETITORS_LIST = [
    "Supervised-Gaussian-HMM-Forward",
    "Causal-Window-RandomForest",
    "Causal-Window-HistGBDT",
]


def causal_hampel_filter_multichannel(Y_in, window_len=7, threshold_sigma=6.0, reset_indices=None):
    """
    Strictly causal sliding-window Hampel despiker over `[max(seg_start, t-L+1) : t+1]`.
    Never reads future samples `t+1..T`. Vectorized over main segment body via sliding_window_view.
    """
    T_len, m = Y_in.shape
    Y_out = Y_in.copy()
    replaced_mask = np.zeros((T_len, m), dtype=bool)
    resets = sorted(list(set(reset_indices or [0])))
    if 0 not in resets:
        resets = [0] + resets
    seg_bounds = resets + [T_len]

    for s_idx in range(len(seg_bounds) - 1):
        s_start = seg_bounds[s_idx]
        s_end = seg_bounds[s_idx + 1]
        seg_len = s_end - s_start
        if seg_len <= 2:
            continue

        warm_end = min(s_end, s_start + window_len - 1)
        for t in range(s_start + 2, warm_end):
            win = Y_in[s_start : t + 1]
            med = np.median(win, axis=0)
            mad = np.maximum(1.4826 * np.median(np.abs(win - med[None, :]), axis=0), 1e-6)
            bad = np.abs(Y_in[t] - med) / mad > threshold_sigma
            if np.any(bad):
                Y_out[t, bad] = med[bad]
                replaced_mask[t, bad] = True

        if seg_len >= window_len:
            seg = Y_in[s_start:s_end]
            wins = sliding_window_view(seg, window_shape=window_len, axis=0)
            meds = np.median(wins, axis=-1)
            mads = np.maximum(1.4826 * np.median(np.abs(wins - meds[:, :, None]), axis=-1), 1e-6)
            y_body = seg[window_len - 1 :]
            bad_body = (np.abs(y_body - meds) / mads) > threshold_sigma
            idx_t_start = s_start + window_len - 1
            Y_out[idx_t_start:s_end] = np.where(bad_body, meds, y_body)
            replaced_mask[idx_t_start:s_end] = bad_body

    return Y_out, replaced_mask


def extract_causal_window_features(Y_norm, window_len=10, reset_indices=None):
    """
    Extracts strictly causal past-window statistical features over `[max(seg_start, t-L+1) : t+1]`
    for each timestamp `t` (total dimension = 6 * m = 36 features for m=6).
    """
    T_len, m = Y_norm.shape
    feats = np.zeros((T_len, 6 * m), dtype=np.float64)
    resets = sorted(list(set(reset_indices or [0])))
    if 0 not in resets:
        resets = [0] + resets
    seg_bounds = resets + [T_len]

    for s_idx in range(len(seg_bounds) - 1):
        s_start = seg_bounds[s_idx]
        s_end = seg_bounds[s_idx + 1]
        seg_len = s_end - s_start
        if seg_len == 0:
            continue

        warm_end = min(s_end, s_start + window_len - 1)
        for t in range(s_start, warm_end):
            win = Y_norm[s_start : t + 1]
            y_curr = Y_norm[t]
            w_mean = np.mean(win, axis=0)
            w_std = np.std(win, axis=0)
            w_min = np.min(win, axis=0)
            w_max = np.max(win, axis=0)
            y_diff = (Y_norm[t] - Y_norm[t - 1]) if t > s_start else np.zeros(m, dtype=np.float64)
            feats[t] = np.concatenate([y_curr, w_mean, w_std, w_min, w_max, y_diff])

        if seg_len >= window_len:
            seg = Y_norm[s_start:s_end]
            wins = sliding_window_view(seg, window_shape=window_len, axis=0)
            y_curr_b = seg[window_len - 1 :]
            w_mean_b = np.mean(wins, axis=-1)
            w_std_b = np.std(wins, axis=-1)
            w_min_b = np.min(wins, axis=-1)
            w_max_b = np.max(wins, axis=-1)
            if window_len >= 2:
                y_diff_b = seg[window_len - 1 :] - seg[window_len - 2 : -1]
            else:
                y_diff_b = np.vstack([np.zeros((1, m)), seg[1:] - seg[:-1]])
            feats[s_start + window_len - 1 : s_end] = np.hstack([
                y_curr_b, w_mean_b, w_std_b, w_min_b, w_max_b, y_diff_b
            ])

    return feats


def fit_task_competitors_on_train(train_streams, model_params, label_key="basic_labels", window_len=10, max_train_samples=50000, seed=20261015):
    """
    Fits the 3 task competitors (`Supervised-Gaussian-HMM-Forward`,
    `Causal-Window-RandomForest`, `Causal-Window-HistGBDT`) strictly on training subjects.
    Also caches `X_fit` and `y_fit` so that 15%-Prefix Calibration (P1-E) can fairly
    adapt all 3 task competitors using the target subject's prefix.
    """
    mean_tr = model_params["scaler_mean"]
    std_tr = model_params["scaler_std"]
    M = int(model_params["num_modes"])

    X_tr_list = []
    y_tr_list = []
    for s in train_streams.values():
        Y_n = (s["Y_raw"] - mean_tr) / std_tr
        F_s = extract_causal_window_features(Y_n, window_len=window_len, reset_indices=s["reset_indices"])
        lbls = s[label_key]
        valid = (lbls >= 0) & (lbls < M)
        if np.any(valid):
            X_tr_list.append(F_s[valid])
            y_tr_list.append(lbls[valid])

    X_tr = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)

    f_dim = X_tr.shape[1]
    hmm_mu = np.zeros((M, f_dim), dtype=np.float64)
    hmm_var = np.ones((M, f_dim), dtype=np.float64)
    for j in range(M):
        idx_j = (y_tr == j)
        if np.sum(idx_j) >= 10:
            hmm_mu[j] = np.mean(X_tr[idx_j], axis=0)
            hmm_var[j] = np.var(X_tr[idx_j], axis=0) + 1e-2

    rng = np.random.default_rng(seed)
    if len(y_tr) > max_train_samples:
        sel = rng.choice(len(y_tr), size=max_train_samples, replace=False)
        X_fit = X_tr[sel]
        y_fit = y_tr[sel]
    else:
        X_fit = X_tr
        y_fit = y_tr

    rf_clf = RandomForestClassifier(
        n_estimators=60,
        max_depth=14,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf_clf.fit(X_fit, y_fit)

    gbdt_clf = HistGradientBoostingClassifier(
        max_iter=60,
        max_depth=8,
        learning_rate=0.1,
        random_state=seed,
    )
    gbdt_clf.fit(X_fit, y_fit)

    return {
        "window_len": int(window_len),
        "hmm_mu": hmm_mu,
        "hmm_var": hmm_var,
        "T_mat": model_params["T_mat"],
        "pi_init": model_params["pi_init"],
        "rf_clf": rf_clf,
        "gbdt_clf": gbdt_clf,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "n_train_samples_total": int(len(y_tr)),
        "n_train_samples_tree_fit": int(len(y_fit)),
    }


def adapt_task_competitors_with_prefix(base_comp_bundle, Y_prefix_norm, y_prefix_labels, seed=20261024):
    """
    Fairly adapts all 3 supervised task competitors (`Supervised-Gaussian-HMM-Forward`,
    `Causal-Window-RandomForest`, `Causal-Window-HistGBDT`) using the target subject's
    15% supervised prefix (`Y_prefix_norm`, `y_prefix_labels`), matching the 0.75/0.25
    target-prefix parameter adaptation provided to switching state-space filters (Closing P1-E).
    """
    w_len = base_comp_bundle["window_len"]
    M = len(base_comp_bundle["pi_init"])
    F_pref = extract_causal_window_features(Y_prefix_norm, window_len=w_len, reset_indices=[0])
    valid_p = (y_prefix_labels >= 0) & (y_prefix_labels < M)

    if not np.any(valid_p):
        return base_comp_bundle

    X_p = F_pref[valid_p]
    y_p = y_prefix_labels[valid_p]

    hmm_mu_adapt = base_comp_bundle["hmm_mu"].copy()
    hmm_var_adapt = base_comp_bundle["hmm_var"].copy()
    for j in range(M):
        idx_j = (y_p == j)
        if np.sum(idx_j) >= 10:
            hmm_mu_adapt[j] = 0.75 * np.mean(X_p[idx_j], axis=0) + 0.25 * hmm_mu_adapt[j]

    # Combine training subset with upweighted target prefix samples for RF and HistGBDT
    X_tr_base = base_comp_bundle["X_fit"]
    y_tr_base = base_comp_bundle["y_fit"]
    X_comb = np.vstack([X_tr_base, X_p])
    y_comb = np.concatenate([y_tr_base, y_p])
    # Weight prefix samples so total prefix weight is ~20% of training weight
    w_prefix_val = max(5.0, 0.20 * float(len(y_tr_base)) / max(1.0, float(len(y_p))))
    sample_weights = np.concatenate([
        np.ones(len(y_tr_base), dtype=np.float64),
        np.full(len(y_p), w_prefix_val, dtype=np.float64),
    ])

    rf_adapt = RandomForestClassifier(
        n_estimators=60,
        max_depth=14,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    rf_adapt.fit(X_comb, y_comb, sample_weight=sample_weights)

    gbdt_adapt = HistGradientBoostingClassifier(
        max_iter=60,
        max_depth=8,
        learning_rate=0.1,
        random_state=seed,
    )
    gbdt_adapt.fit(X_comb, y_comb, sample_weight=sample_weights)

    return {
        "window_len": w_len,
        "hmm_mu": hmm_mu_adapt,
        "hmm_var": hmm_var_adapt,
        "T_mat": base_comp_bundle["T_mat"],
        "pi_init": base_comp_bundle["pi_init"],
        "rf_clf": rf_adapt,
        "gbdt_clf": gbdt_adapt,
        "prefix_adapted": True,
        "n_prefix_valid_samples": int(len(y_p)),
    }


def run_task_competitor_online(Y_norm, comp_bundle, method_name, reset_indices=None):
    """
    Runs one of the 3 task competitors in a strictly causal online pass.
    """
    T_len, m = Y_norm.shape
    M = len(comp_bundle["pi_init"])
    w_len = comp_bundle["window_len"]
    feats = extract_causal_window_features(Y_norm, window_len=w_len, reset_indices=reset_indices)

    if method_name == "Supervised-Gaussian-HMM-Forward":
        hmm_mu = comp_bundle["hmm_mu"]
        hmm_var = comp_bundle["hmm_var"]
        T_mat = comp_bundle["T_mat"]
        pi_init = comp_bundle["pi_init"]
        resets = set(reset_indices or [0])
        resets.add(0)

        filtered_mode = np.zeros(T_len, dtype=np.int32)
        filtered_probs = np.zeros((T_len, M), dtype=np.float64)
        pred_prior_probs = np.zeros((T_len, M), dtype=np.float64)
        gamma_prev = pi_init.copy()

        log_emiss = np.zeros((T_len, M), dtype=np.float64)
        for j in range(M):
            diff = feats - hmm_mu[j][None, :]
            log_emiss[:, j] = -0.5 * np.sum((diff ** 2) / hmm_var[j][None, :] + np.log(2.0 * np.pi * hmm_var[j][None, :]), axis=1)

        for t in range(T_len):
            if t in resets:
                pi_pred = pi_init.copy()
            else:
                pi_pred = gamma_prev @ T_mat
                pi_pred = np.maximum(pi_pred, 1e-12)
                pi_pred /= np.sum(pi_pred)
            pred_prior_probs[t] = pi_pred
            lp = np.log(pi_pred) + 0.15 * log_emiss[t]
            lp -= logsumexp(lp)
            gamma_t = np.exp(lp)
            gamma_t /= np.sum(gamma_t)
            filtered_probs[t] = gamma_t
            filtered_mode[t] = int(np.argmax(gamma_t))
            gamma_prev = gamma_t

        return {
            "filtered_mode": filtered_mode,
            "filtered_probs": filtered_probs,
            "pred_prior_probs": pred_prior_probs,
            "filtered_state": Y_norm.copy(),
            "pred_obs": Y_norm.copy(),
        }

    elif method_name == "Causal-Window-RandomForest":
        clf = comp_bundle["rf_clf"]
    elif method_name == "Causal-Window-HistGBDT":
        clf = comp_bundle["gbdt_clf"]
    else:
        raise ValueError(f"Unknown task competitor: {method_name}")

    probs_raw = clf.predict_proba(feats)
    probs = np.zeros((T_len, M), dtype=np.float64)
    for idx_c, c_label in enumerate(clf.classes_):
        if 0 <= int(c_label) < M:
            probs[:, int(c_label)] = probs_raw[:, idx_c]
    filtered_mode = np.argmax(probs, axis=1).astype(np.int32)
    return {
        "filtered_mode": filtered_mode,
        "filtered_probs": probs,
        "pred_prior_probs": probs,
        "filtered_state": Y_norm.copy(),
        "pred_obs": Y_norm.copy(),
    }


def run_any_method_online(
    Y_norm,
    model_params,
    comp_bundle,
    method_name,
    reset_indices=None,
    val_tuned_hampel_hp=(5, 6.0),
):
    """
    Unified dispatcher for all switching state-space methods, single-factor/component ablations,
    and causal-window task competitors.
    """
    t0 = time.perf_counter()
    if method_name in TASK_COMPETITORS_LIST:
        out = run_task_competitor_online(Y_norm, comp_bundle, method_name, reset_indices=reset_indices)
    elif method_name == "Hampel-ValTuned+Standard-SKF":
        w_l, tau_s = val_tuned_hampel_hp
        Y_desp, _ = causal_hampel_filter_multichannel(Y_norm, window_len=w_l, threshold_sigma=tau_s, reset_indices=reset_indices)
        out = run_strict_online_switching_filter(Y_desp, model_params, method="Standard-SKF", reset_indices=reset_indices, reset_mode="pre_obs_equilibrium")
    elif method_name == "Hampel31-Fixed6+Standard-SKF":
        Y_desp, _ = causal_hampel_filter_multichannel(Y_norm, window_len=31, threshold_sigma=6.0, reset_indices=reset_indices)
        out = run_strict_online_switching_filter(Y_desp, model_params, method="Standard-SKF", reset_indices=reset_indices, reset_mode="pre_obs_equilibrium")
    elif method_name == "Hampel-ValTuned+IBC-SHMM":
        w_l, tau_s = val_tuned_hampel_hp
        Y_desp, _ = causal_hampel_filter_multichannel(Y_norm, window_len=w_l, threshold_sigma=tau_s, reset_indices=reset_indices)
        out = run_strict_online_switching_filter(Y_desp, model_params, method="IBC-SHMM", reset_indices=reset_indices, reset_mode="pre_obs_equilibrium")
    elif method_name == "IBC-SHMM-ObsWarmStart":
        out = run_strict_online_switching_filter(Y_norm, model_params, method="IBC-SHMM-ObsWarmStart", reset_indices=reset_indices, reset_mode="obs_warmstart")
    else:
        out = run_strict_online_switching_filter(Y_norm, model_params, method=method_name, reset_indices=reset_indices, reset_mode="pre_obs_equilibrium")

    elapsed_s = time.perf_counter() - t0
    out["elapsed_sec"] = float(elapsed_s)
    out["ms_per_step"] = float(1000.0 * elapsed_s / max(1, Y_norm.shape[0]))
    return out
