#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Strict Online Evaluator, Pre-Observation Reset Wrapper, Exact Continuous-Time (±0.5s / ±1.0s)
One-to-One Hungarian Transition Event Matcher, and Future-Perturbation Causality Verifier
(Gates G4 / CAUSAL-P1-4 / Round-20 P0-A, P0-B, P1-D).

Implements the unconfounded, strict one-factor-at-a-time attribution chain:
  - Step 0 (`IBC-SHMM-LegacySat`):
      Legacy thresholds (4.2, 5.5, 6.5, nu0=3.2, z_gate=2.65, alpha=0.48), scalar Tier-3 weight,
      M3 w_dof_raw, and q_sat-clipped mode score:
      l = -0.5*ld_dyn - 0.45*rho_sat(u2_dyn) - 0.16*rho_sat(u2_eq) - 0.02*ld_eq.
  - Step 1a (`IBC-Ab1a-UnclipOnly` — Strictly 1 factor vs Step 0):
      Replaces ONLY min(u^2, q_sat) -> u^2 in rho_dyn and rho_eq; keeps exact coefficients
      (-0.5, -0.45, -0.16, -0.02), legacy thresholds, scalar Tier-3, and M3 w_dof_raw.
  - Step 1b (`IBC-Ab1b-UnclipPlusDyn050` — Strictly 1 factor vs Step 1a):
      Changes ONLY dynamic Student-t coefficient -0.45 -> -0.50; keeps -0.16*rho_unclip(u2_eq) - 0.02*ld_eq.
  - Step 1c (`IBC-Ab1c-UnclipPlusDyn050Eq008` — Strictly 1 factor vs Step 1b):
      Changes ONLY equilibrium Student-t coefficient -0.16 -> -0.08 (-0.5*w_eq); keeps -0.02*ld_eq.
  - Step 1d (`IBC-Ab1-UnclipModeScoreOnly` — Strictly 1 factor vs Step 1c):
      Drops ONLY -0.02*ld_eq (completing repaired mode-score form -0.5*ld_dyn - 0.50*rho_unclip_dyn - 0.08*rho_unclip_eq),
      while still keeping legacy thresholds, scalar Tier-3, and M3 w_dof_raw.
  - Step 2 (`IBC-Ab2-UnclipPlusThresholds` — Strictly 1 factor vs Step 1d):
      Updates ONLY thresholds (4.2,5.5,6.5,nu0=3.2,z_gate=2.65,alpha=0.48) -> (3.0,5.0,3.8,nu0=4.0,z_gate=3.2,alpha=0.55),
      keeping scalar Tier-3 and M3 w_dof_raw.
  - Step 3 (`IBC-SHMM` — Strictly 1 factor vs Step 2):
      Updates ONLY Tier-3 weight from scalar to 0.5/0.5 coordinate-global hybrid weight (keeping M3 w_dof_raw).
  - Orthogonal M3 State-Weight Check (`IBC-Ab3-PreM3StateWeight` vs `IBC-SHMM` — Strictly 1 factor):
      Identical to `IBC-SHMM` in mode score, thresholds, and Tier-3 0.5/0.5 hybrid weight;
      differs ONLY by omitting min(1, q_sat/u2_dyn) in w_dof_raw.
  - Component & Causality Ablations on `IBC-SHMM`:
      * `IBC-NoSubspace-Ab`: Disables Tier-2 clean-subspace projection (\hat{C}_t = {1..m})
      * `IBC-NoEqAnchor-Ab`: Sets equilibrium anchor weight w_eq = 0.0
      * `IBC-NoTempGate-Ab`: Disables transition & likelihood tempering (eta_t = 1.0)
      * `IBC-PureMarginal-Ab`: Unscaled Gaussian marginal log-likelihood on \hat{C}_t
      * `IBC-SHMM-PrevStep`: Strictly F_{t-1}-measurable transition gate T_{t-1}(eta_{t-1})
      * `IBC-SHMM-ObsWarmStart`: Observation-projected segment reset warm-start
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import chi2


IBC_FAMILY_METHODS = (
    "IBC-SHMM",
    "IBC-SHMM-PrevStep",
    "IBC-SHMM-ObsWarmStart",
    "IBC-SHMM-LegacySat",
    "IBC-PureMarginal-Ab",
    "IBC-Ab1a-UnclipOnly",
    "IBC-Ab1b-UnclipPlusDyn050",
    "IBC-Ab1c-UnclipPlusDyn050Eq008",
    "IBC-Ab1-UnclipModeScoreOnly",
    "IBC-Ab2-UnclipPlusThresholds",
    "IBC-Ab3-PreM3StateWeight",
    "IBC-NoSubspace-Ab",
    "IBC-NoEqAnchor-Ab",
    "IBC-NoTempGate-Ab",
)


def compute_schur_marginal_precision_batch(R, R_inv, omega_mat, eye_m):
    """
    Exact 3D batch Schur-complement marginal precision matrix W_{t,marg}^{(j)}(omega_t)
    across all M modes simultaneously.
    """
    if np.all(omega_mat >= 0.999999):
        return R_inv
    w = np.clip(omega_mat, 0.0, 1.0)
    d_sqrt = np.sqrt(w)
    diag_R = np.diagonal(R, axis1=1, axis2=2)
    M_mat = (d_sqrt[:, :, None] * R * d_sqrt[:, None, :]) + (diag_R * (1.0 - w))[:, :, None] * eye_m
    M_inv = np.linalg.inv(M_mat + 1e-12 * eye_m)
    return d_sqrt[:, :, None] * M_inv * d_sqrt[:, None, :]


def run_strict_online_switching_filter(
    Y_norm,
    model_params,
    method="IBC-SHMM",
    reset_indices=None,
    reset_mode="pre_obs_equilibrium",
    eq_weight_override=None,
):
    r"""
    Strictly causal online switching state-space filter enforcing `reset_mode="pre_obs_equilibrium"`
    and supporting both main baseline comparison and unconfounded one-factor-at-a-time attribution.
    """
    T_len, m = Y_norm.shape
    M = int(model_params["num_modes"])
    d = int(model_params["state_dim"])
    A = model_params["A"]
    b = model_params["b"]
    C = model_params["C"]
    Q = model_params["Q"]
    R = model_params["R"]
    T_mat = model_params["T_mat"]
    pi_init = model_params["pi_init"]
    mu_eq_x = model_params["mu_eq_x"]
    Sigma_eq_x = model_params["Sigma_eq_x"]

    eye_d = np.eye(d, dtype=np.float64)[None, :, :]
    eye_m = np.eye(m, dtype=np.float64)[None, :, :]
    is_identity_C = bool(d == m and np.allclose(C, eye_m))

    R_inv = np.linalg.inv(R + 1e-9 * eye_m)
    C_mu_eq = mu_eq_x.copy() if is_identity_C else np.einsum("mij,mj->mi", C, mu_eq_x)
    Sigma_eq_y = (Sigma_eq_x + R) if is_identity_C else (np.einsum("mij,mjk,mlk->mil", C, Sigma_eq_x, C) + R)
    Sigma_eq_y_inv = np.linalg.inv(Sigma_eq_y + 1e-9 * eye_m)
    _, logdet_Sigma_eq_y = np.linalg.slogdet(2.0 * np.pi * Sigma_eq_y)
    CRC_full = R_inv if is_identity_C else np.einsum("mji,mjk,mkl->mil", C, R_inv, C)

    reset_set = set(reset_indices or [0])
    reset_set.add(0)

    if eq_weight_override is not None:
        w_eq = float(eq_weight_override)
    elif method in (
        "Standard-SKF-NoEq",
        "StudentT-IMM-1Step-NoEq",
        "StudentT-IMM-VB-ELBO",
        "Coord-StudentT-IMM-ELBO",
        "IBC-PureMarginal-Ab",
        "IBC-NoEqAnchor-Ab",
    ):
        w_eq = 0.0
    elif method in ("StudentT-IMM-VB-Cal", "Coord-StudentT-IMM"):
        w_eq = 0.08
    elif method in IBC_FAMILY_METHODS:
        w_eq = 0.16
    else:
        w_eq = 0.15

    filtered_mode = np.zeros(T_len, dtype=np.int32)
    filtered_probs = np.zeros((T_len, M), dtype=np.float64)
    pred_prior_probs = np.zeros((T_len, M), dtype=np.float64)
    filtered_state = np.zeros((T_len, d), dtype=np.float64)
    pred_obs = np.zeros((T_len, m), dtype=np.float64)
    clean_channel_mask = np.ones((T_len, m), dtype=bool)

    gamma_prev = pi_init.copy()
    mu_prev = mu_eq_x.copy()
    Sigma_prev = Sigma_eq_x.copy()
    eta_prev = 1.0

    # Explicit Factor Flags for Unconfounded One-Factor-at-a-Time Attribution:
    use_legacy_thresholds = method in (
        "IBC-SHMM-LegacySat",
        "IBC-Ab1a-UnclipOnly",
        "IBC-Ab1b-UnclipPlusDyn050",
        "IBC-Ab1c-UnclipPlusDyn050Eq008",
        "IBC-Ab1-UnclipModeScoreOnly",
    )
    use_legacy_tier3_scalar = method in (
        "IBC-SHMM-LegacySat",
        "IBC-PureMarginal-Ab",
        "IBC-Ab1a-UnclipOnly",
        "IBC-Ab1b-UnclipPlusDyn050",
        "IBC-Ab1c-UnclipPlusDyn050Eq008",
        "IBC-Ab1-UnclipModeScoreOnly",
        "IBC-Ab2-UnclipPlusThresholds",
    )
    # Note: Step 0, Step 1a..1d, Step 2, and Step 3 (IBC-SHMM) ALL use M3 w_dof_raw;
    # ONLY `IBC-Ab3-PreM3StateWeight` disables the q_sat factor in w_dof_raw!
    use_m3_sixth_order_w_raw = (method != "IBC-Ab3-PreM3StateWeight")
    use_clean_subspace_proj = (method != "IBC-NoSubspace-Ab")
    use_temp_gate = (method != "IBC-NoTempGate-Ab")

    if use_legacy_thresholds:
        c_clean, c_burst, c_iso, nu_clean, nu_0, z_gate, anom_alpha = 4.2, 5.5, 6.5, 24.0, 3.2, 2.65, 0.48
    else:
        c_clean, c_burst, c_iso, nu_clean, nu_0, z_gate, anom_alpha = 3.0, 5.0, 3.8, 24.0, 4.0, 3.2, 0.55

    q95_m = float(chi2.ppf(0.95, df=m))
    q_sat = max(22.0, float(chi2.ppf(0.999, df=m)))
    c_hub = 1.345
    nu_st = 4.0

    for t in range(T_len):
        y_t = Y_norm[t]
        is_reset = (t in reset_set)

        if is_reset:
            pi_pred_base = pi_init.copy()
            if reset_mode == "pre_obs_equilibrium":
                mu_pred_base = mu_eq_x.copy()
            else:
                mu_pred_base = np.repeat(y_t[None, :d], M, axis=0)
            Sigma_pred_base = Sigma_eq_x.copy()
            eta_prev = 1.0
        else:
            if method == "IBC-SHMM-PrevStep" and use_temp_gate:
                T_pre = eta_prev * T_mat + (1.0 - eta_prev) * np.eye(M)
            else:
                T_pre = T_mat

            pi_pred_base = gamma_prev @ T_pre
            pi_pred_base = np.maximum(pi_pred_base, 1e-12)
            pi_pred_base /= np.sum(pi_pred_base)

            W_ij = (gamma_prev[:, None] * T_pre) / pi_pred_base[None, :]
            mu_mix = W_ij.T @ mu_prev
            diff_m = mu_prev[:, None, :] - mu_mix[None, :, :]
            cov_add = diff_m[:, :, :, None] * diff_m[:, :, None, :]
            Sigma_mix = np.einsum("ij,ijkl->jkl", W_ij, Sigma_prev[:, None, :, :] + cov_add)
            mu_pred_base = np.einsum("mij,mj->mi", A, mu_mix) + b
            Sigma_pred_base = np.einsum("mij,mjk,mlk->mil", A, Sigma_mix, A) + Q

        pred_prior_probs[t] = pi_pred_base

        if is_identity_C:
            r_dyn = y_t[None, :] - mu_pred_base
            S_dyn = 0.5 * (Sigma_pred_base + np.swapaxes(Sigma_pred_base, 1, 2)) + R
        else:
            r_dyn = y_t[None, :] - np.einsum("mij,mj->mi", C, mu_pred_base)
            S_raw = np.einsum("mij,mjk,mlk->mil", C, Sigma_pred_base, C) + R
            S_dyn = 0.5 * (S_raw + np.swapaxes(S_raw, 1, 2))

        S_dyn_inv = np.linalg.inv(S_dyn + 1e-9 * eye_m)
        diag_S = np.maximum(np.diagonal(S_dyn, axis1=1, axis2=2), 1e-12)
        z_std = np.abs(r_dyn) / np.sqrt(diag_S)

        z_min_ch = np.min(z_std, axis=0)
        z_med_min = float(np.median(z_min_ch))
        z_max_min = float(np.max(z_min_ch))
        z_anom = max(z_med_min, anom_alpha * z_max_min)

        if use_temp_gate and z_anom > z_gate:
            eta_t = max(0.01, float((z_gate / z_anom) ** 3))
        else:
            eta_t = 1.0

        clean_idx = np.where(z_min_ch <= c_clean)[0]
        if len(clean_idx) == 0:
            clean_idx = np.array([int(np.argmin(z_min_ch))], dtype=np.int64)
        clean_mask = np.zeros(m, dtype=bool)
        clean_mask[clean_idx] = True
        clean_channel_mask[t] = clean_mask

        uses_unified_imm = (
            (method in IBC_FAMILY_METHODS)
            and (method not in ("IBC-PureMarginal-Ab", "IBC-SHMM-PrevStep", "IBC-NoTempGate-Ab"))
        )
        if uses_unified_imm and (not is_reset) and eta_t < 0.999999:
            T_uni = eta_t * T_mat + (1.0 - eta_t) * np.eye(M)
            pi_eff = gamma_prev @ T_uni
            pi_eff = np.maximum(pi_eff, 1e-12)
            pi_eff /= np.sum(pi_eff)
            W_ij = (gamma_prev[:, None] * T_uni) / pi_eff[None, :]
            mu_mix = W_ij.T @ mu_prev
            diff_m = mu_prev[:, None, :] - mu_mix[None, :, :]
            cov_add = diff_m[:, :, :, None] * diff_m[:, :, None, :]
            Sigma_mix = np.einsum("ij,ijkl->jkl", W_ij, Sigma_prev[:, None, :, :] + cov_add)
            mu_pred = np.einsum("mij,mj->mi", A, mu_mix) + b
            Sigma_pred = np.einsum("mij,mjk,mlk->mil", A, Sigma_mix, A) + Q
            if is_identity_C:
                r_dyn = y_t[None, :] - mu_pred
                S_dyn = 0.5 * (Sigma_pred + np.swapaxes(Sigma_pred, 1, 2)) + R
            else:
                r_dyn = y_t[None, :] - np.einsum("mij,mj->mi", C, mu_pred)
                S_raw = np.einsum("mij,mjk,mlk->mil", C, Sigma_pred, C) + R
                S_dyn = 0.5 * (S_raw + np.swapaxes(S_raw, 1, 2))
            S_dyn_inv = np.linalg.inv(S_dyn + 1e-9 * eye_m)
            diag_S = np.maximum(np.diagonal(S_dyn, axis1=1, axis2=2), 1e-12)
            z_std = np.abs(r_dyn) / np.sqrt(diag_S)
        else:
            pi_eff = pi_pred_base
            mu_pred = mu_pred_base
            Sigma_pred = Sigma_pred_base

        u2_full_vec = np.einsum("mi,mij,mj->m", r_dyn, S_dyn_inv, r_dyn)
        r_eq_mat = y_t[None, :] - C_mu_eq
        u2_eq_vec = np.einsum("mi,mij,mj->m", r_eq_mat, Sigma_eq_y_inv, r_eq_mat)
        _, logdet_S_vec = np.linalg.slogdet(2.0 * np.pi * S_dyn)
        P_prior_inv_all = np.linalg.inv(Sigma_pred + 1e-9 * eye_d)
        P_mu_all = np.einsum("mij,mj->mi", P_prior_inv_all, mu_pred)

        if method in ("Standard-SKF-NoEq", "Standard-SKF"):
            K_post = P_prior_inv_all + CRC_full
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_term = np.einsum("mij,j->mi", R_inv, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, R_inv, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + Cy_term)
            log_ev = -0.5 * (u2_full_vec + logdet_S_vec) - w_eq * 0.5 * (u2_eq_vec + logdet_Sigma_eq_y)

        elif method == "Sat-SKF":
            scale_sat = np.minimum(1.0, q_sat / np.maximum(u2_full_vec, 1e-12))
            K_post = P_prior_inv_all + scale_sat[:, None, None] * CRC_full
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_base = np.einsum("mij,j->mi", R_inv, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, R_inv, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + scale_sat[:, None] * Cy_base)
            log_ev = -0.5 * (np.minimum(u2_full_vec, q_sat) + logdet_S_vec) - w_eq * 0.5 * np.minimum(u2_eq_vec, q_sat)

        elif method in ("StudentT-IMM-1Step-NoEq", "StudentT-IMM-1Step"):
            w_st = np.minimum(1.0, (nu_st + m) / (nu_st + u2_full_vec))
            K_post = P_prior_inv_all + w_st[:, None, None] * CRC_full
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_base = np.einsum("mij,j->mi", R_inv, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, R_inv, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + w_st[:, None] * Cy_base)
            log_ev = -0.5 * (nu_st + m) * np.log(1.0 + u2_full_vec / nu_st) - 0.5 * logdet_S_vec - w_eq * 0.5 * (nu_st + m) * np.log(1.0 + u2_eq_vec / nu_st)

        elif method == "Huber-SKF-SingleLayer":
            w_hub = np.where(z_std <= c_hub, 1.0, c_hub / np.maximum(z_std, 1e-12))
            d_hub = np.sqrt(w_hub)
            W_batch = d_hub[:, :, None] * R_inv * d_hub[:, None, :]
            K_post = P_prior_inv_all + (W_batch if is_identity_C else np.einsum("mji,mjk,mkl->mil", C, W_batch, C))
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_term = np.einsum("mij,j->mi", W_batch, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, W_batch, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + Cy_term)
            u2_hub = np.sum(np.where(z_std <= c_hub, 0.5 * z_std ** 2, c_hub * z_std - 0.5 * c_hub ** 2), axis=1)
            log_ev = -u2_hub - 0.5 * logdet_S_vec - w_eq * 0.5 * np.minimum(u2_eq_vec, q_sat)

        elif method in ("StudentT-IMM-VB-ELBO", "StudentT-IMM-VB-Cal"):
            lam_vb = np.ones(M, dtype=np.float64)
            mu_curr = mu_pred.copy()
            Sigma_curr = Sigma_pred.copy()
            CRy_all = np.einsum("mij,j->mi", R_inv, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, R_inv, y_t)
            for _ in range(4):
                K_post = P_prior_inv_all + lam_vb[:, None, None] * CRC_full
                Sigma_curr = np.linalg.inv(K_post + 1e-9 * eye_d)
                res_curr = (y_t[None, :] - mu_curr) if is_identity_C else (y_t[None, :] - np.einsum("mij,mj->mi", C, mu_curr))
                quad_r = np.einsum("mi,mij,mj->m", res_curr, R_inv, res_curr)
                tr_term = np.einsum("mij,mji->m", CRC_full, Sigma_curr)
                lam_vb = (nu_st + m) / (nu_st + quad_r + tr_term)
                mu_curr = np.einsum("mij,mj->mi", Sigma_curr, P_mu_all + lam_vb[:, None] * CRy_all)
            mu_post = mu_curr
            Sigma_post = 0.5 * (Sigma_curr + np.swapaxes(Sigma_curr, 1, 2))
            log_ev = -0.5 * (nu_st + m) * np.log(1.0 + u2_full_vec / nu_st) - 0.5 * logdet_S_vec - w_eq * (nu_st + m) * np.log(1.0 + u2_eq_vec / nu_st)

        elif method in ("Coord-StudentT-IMM-ELBO", "Coord-StudentT-IMM"):
            w_coord = np.minimum(1.0, (nu_st + 1.0) / (nu_st + z_std ** 2))
            W_batch = compute_schur_marginal_precision_batch(R, R_inv, w_coord, eye_m)
            K_post = P_prior_inv_all + (W_batch if is_identity_C else np.einsum("mji,mjk,mkl->mil", C, W_batch, C))
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_term = np.einsum("mij,j->mi", W_batch, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, W_batch, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + Cy_term)
            log_ev = -0.5 * np.sum((nu_st + 1.0) * np.log(1.0 + (z_std ** 2) / nu_st), axis=1) - 0.5 * logdet_S_vec - w_eq * 0.5 * (nu_st + m) * np.log(1.0 + u2_eq_vec / nu_st)

        else:
            # IBC-SHMM Family & Strict One-Factor-at-a-Time Attribution Variants
            is_pure_marg = (method == "IBC-PureMarginal-Ab")
            nu_eff_clean = nu_0 if is_pure_marg else nu_clean
            use_dof_scale = (not is_pure_marg)

            u2_excess = np.maximum(0.0, u2_full_vec - q95_m)
            if use_m3_sixth_order_w_raw:
                w_dof_raw = ((nu_0 + m) / (nu_0 + m + u2_excess)) * np.minimum(1.0, q_sat / np.maximum(u2_full_vec, 1e-12))
            else:
                w_dof_raw = (nu_0 + m) / (nu_0 + m + u2_full_vec)

            if z_med_min > c_burst:
                # Tier 1: multi-channel burst (including even-m half-channel boundary |O_t^{true}| = m/2)
                excess_b = np.maximum(0.0, z_std - c_burst) / c_burst
                omega_mat = w_dof_raw[:, None] / (1.0 + excess_b ** 2)
                u2_eff_dyn = u2_full_vec
                u2_eff_eq = u2_eq_vec
                ld_eff_dyn = logdet_S_vec
                ld_eff_eq = logdet_Sigma_eq_y
            elif use_clean_subspace_proj and (z_max_min > c_iso):
                # Tier 2: partial-channel outlier -> extract threshold-selected clean subspace \hat{C}_t
                n_c = len(clean_idx)
                dof_scale = (float(m) / float(n_c)) if use_dof_scale else 1.0
                if n_c == m:
                    u2_clean_raw = u2_full_vec
                    ld_CC = logdet_S_vec
                    u2_eq_clean_raw = u2_eq_vec
                    ld_eq_CC = logdet_Sigma_eq_y
                else:
                    eye_c = np.eye(n_c, dtype=np.float64)[None, :, :]
                    r_c = r_dyn[:, clean_idx]
                    S_cc = S_dyn[:, clean_idx[:, None], clean_idx[None, :]]
                    S_cc_inv = np.linalg.inv(S_cc + 1e-9 * eye_c)
                    _, ld_CC = np.linalg.slogdet(2.0 * np.pi * S_cc)
                    u2_clean_raw = np.einsum("mi,mij,mj->m", r_c, S_cc_inv, r_c)

                    r_eq_c = r_eq_mat[:, clean_idx]
                    Seq_cc = Sigma_eq_y[:, clean_idx[:, None], clean_idx[None, :]]
                    Seq_cc_inv = np.linalg.inv(Seq_cc + 1e-9 * eye_c)
                    _, ld_eq_CC = np.linalg.slogdet(2.0 * np.pi * Seq_cc)
                    u2_eq_clean_raw = np.einsum("mi,mij,mj->m", r_eq_c, Seq_cc_inv, r_eq_c)

                u2_eff_dyn = u2_clean_raw * dof_scale
                ld_eff_dyn = ld_CC * dof_scale
                u2_eff_eq = u2_eq_clean_raw * dof_scale
                ld_eff_eq = ld_eq_CC * dof_scale

                u2_clean_excess = np.maximum(0.0, (u2_clean_raw * float(m) / float(n_c)) - q95_m)
                w_dof_clean = (nu_eff_clean + m) / (nu_eff_clean + m + u2_clean_excess)
                excess_f = np.maximum(0.0, z_std - c_clean) / c_clean
                w_fault = w_dof_raw[:, None] / (1.0 + excess_f ** 2)
                omega_mat = np.where(clean_mask[None, :], w_dof_clean[:, None], w_fault)
            else:
                # Tier 3: nominal inlier
                if use_legacy_tier3_scalar:
                    w_clean_all = (nu_eff_clean + m) / (nu_eff_clean + m + u2_excess)
                    omega_mat = np.repeat(w_clean_all[:, None], m, axis=1)
                else:
                    w_coord_clean = (nu_clean + 1.0) / (nu_clean + z_std ** 2)
                    w_dof_clean = (nu_0 + m) / (nu_0 + u2_full_vec)
                    omega_mat = np.minimum(1.0, 0.5 * w_coord_clean + 0.5 * w_dof_clean[:, None])
                u2_eff_dyn = u2_full_vec
                u2_eff_eq = u2_eq_vec
                ld_eff_dyn = logdet_S_vec
                ld_eff_eq = logdet_Sigma_eq_y

            W_batch = compute_schur_marginal_precision_batch(R, R_inv, omega_mat, eye_m)
            K_post = P_prior_inv_all + (W_batch if is_identity_C else np.einsum("mji,mjk,mkl->mil", C, W_batch, C))
            Sigma_post = np.linalg.inv(K_post + 1e-9 * eye_d)
            Sigma_post = 0.5 * (Sigma_post + np.swapaxes(Sigma_post, 1, 2))
            Cy_term = np.einsum("mij,j->mi", W_batch, y_t) if is_identity_C else np.einsum("mji,mjk,k->mi", C, W_batch, y_t)
            mu_post = np.einsum("mij,mj->mi", Sigma_post, P_mu_all + Cy_term)

            # Strict One-Factor-at-a-Time Mode Quasi-Evidence Decomposition (P0-A):
            if is_pure_marg:
                log_ev = -0.5 * ld_eff_dyn - 0.5 * u2_eff_dyn
            elif method == "IBC-SHMM-LegacySat":
                # Step 0: q_sat-clipped + (-0.5, -0.45, -w_eq=-0.16, -0.02)
                rho_dyn = (nu_0 + m) * np.log(1.0 + np.minimum(u2_eff_dyn, q_sat) / nu_0)
                rho_eq = (nu_0 + m) * np.log(1.0 + np.minimum(u2_eff_eq, q_sat) / nu_0)
                log_ev = -0.5 * ld_eff_dyn - 0.45 * rho_dyn - w_eq * rho_eq - 0.02 * ld_eff_eq
            elif method == "IBC-Ab1a-UnclipOnly":
                # Step 1a (Strictly 1 factor vs Step 0): ONLY replace min(u^2, q_sat) -> u^2; keep (-0.5, -0.45, -0.16, -0.02)
                rho_dyn = (nu_0 + m) * np.log(1.0 + u2_eff_dyn / nu_0)
                rho_eq = (nu_0 + m) * np.log(1.0 + u2_eff_eq / nu_0)
                log_ev = -0.5 * ld_eff_dyn - 0.45 * rho_dyn - w_eq * rho_eq - 0.02 * ld_eff_eq
            elif method == "IBC-Ab1b-UnclipPlusDyn050":
                # Step 1b (Strictly 1 factor vs Step 1a): ONLY change dynamic coeff -0.45 -> -0.50; keep (-0.16, -0.02)
                rho_dyn = (nu_0 + m) * np.log(1.0 + u2_eff_dyn / nu_0)
                rho_eq = (nu_0 + m) * np.log(1.0 + u2_eff_eq / nu_0)
                log_ev = -0.5 * ld_eff_dyn - 0.50 * rho_dyn - w_eq * rho_eq - 0.02 * ld_eff_eq
            elif method == "IBC-Ab1c-UnclipPlusDyn050Eq008":
                # Step 1c (Strictly 1 factor vs Step 1b): ONLY change eq coeff -w_eq (-0.16) -> -0.5*w_eq (-0.08); keep -0.02*ld_eq
                rho_dyn = (nu_0 + m) * np.log(1.0 + u2_eff_dyn / nu_0)
                rho_eq = (nu_0 + m) * np.log(1.0 + u2_eff_eq / nu_0)
                log_ev = -0.5 * ld_eff_dyn - 0.50 * rho_dyn - 0.5 * w_eq * rho_eq - 0.02 * ld_eff_eq
            else:
                # Step 1d (`IBC-Ab1-UnclipModeScoreOnly`), Step 2 (`IBC-Ab2-UnclipPlusThresholds`),
                # Step 3 (`IBC-SHMM`), and component ablations: drop -0.02*ld_eq
                rho_dyn = (nu_0 + m) * np.log(1.0 + u2_eff_dyn / nu_0)
                rho_eq = (nu_0 + m) * np.log(1.0 + u2_eff_eq / nu_0)
                log_ev = -0.5 * ld_eff_dyn - 0.50 * rho_dyn - 0.5 * w_eq * rho_eq

        if (method in IBC_FAMILY_METHODS) and (method != "IBC-PureMarginal-Ab") and use_temp_gate:
            tau_temp = 0.52 + 0.48 * eta_t
        else:
            tau_temp = 1.0

        log_post = np.log(np.maximum(pi_eff, 1e-15)) + tau_temp * log_ev
        log_post -= logsumexp(log_post)
        gamma_t = np.exp(log_post)
        gamma_t /= np.sum(gamma_t)

        filtered_probs[t] = gamma_t
        filtered_mode[t] = int(np.argmax(gamma_t))
        x_comb = gamma_t @ mu_post
        filtered_state[t] = x_comb
        if is_identity_C:
            pred_obs[t] = x_comb
        else:
            pred_obs[t] = gamma_t @ np.einsum("mij,mj->mi", C, mu_post)

        gamma_prev = gamma_t
        mu_prev = mu_post
        Sigma_prev = Sigma_post
        eta_prev = eta_t

    return {
        "filtered_mode": filtered_mode,
        "filtered_probs": filtered_probs,
        "pred_prior_probs": pred_prior_probs,
        "filtered_state": filtered_state,
        "pred_obs": pred_obs,
        "clean_channel_mask": clean_channel_mask,
    }


def extract_predicted_mode_switches(filtered_mode, reset_indices=None, min_dwell=2, sampling_rate_hz=5.0):
    resets = set(reset_indices or [0])
    T_len = len(filtered_mode)
    events = []
    if T_len == 0:
        return events

    curr_confirmed_mode = int(filtered_mode[0])
    candidate_mode = curr_confirmed_mode
    candidate_onset = 0
    candidate_count = 1
    fs = float(sampling_rate_hz)

    for t in range(1, T_len):
        if t in resets:
            curr_confirmed_mode = int(filtered_mode[t])
            candidate_mode = curr_confirmed_mode
            candidate_onset = t
            candidate_count = 1
            continue

        m_t = int(filtered_mode[t])
        if m_t != curr_confirmed_mode:
            if m_t == candidate_mode:
                candidate_count += 1
            else:
                candidate_mode = m_t
                candidate_onset = t
                candidate_count = 1
            if candidate_count >= min_dwell:
                events.append({
                    "from_mode": int(curr_confirmed_mode),
                    "to_mode": int(candidate_mode),
                    "onset_sample": int(candidate_onset),
                    "confirm_sample": int(t),
                    "onset_sec": float(candidate_onset) / fs,
                    "confirm_sec": float(t) / fs,
                })
                curr_confirmed_mode = candidate_mode
        else:
            candidate_mode = curr_confirmed_mode
            candidate_count = 0

    return events


def match_transition_events_hungarian(
    pred_events,
    true_events,
    tolerance_samples=None,
    require_direction=True,
    total_duration_hours=1.0,
    tolerance_sec=None,
    sampling_rate_hz=None,
):
    """
    One-to-One Hungarian bipartite transition event matcher.
    When `tolerance_sec` and `sampling_rate_hz` are provided (or when events carry `onset_sec`),
    matches strictly by physical continuous time in seconds (`dist_sec <= tolerance_sec + 1e-9`),
    eliminating discrete stride-rounding distortion (Closing Round-20 P1-D).
    """
    n_p = len(pred_events)
    n_t = len(true_events)
    if n_t == 0 and n_p == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": np.nan, "recall": np.nan, "f1": np.nan,
            "mean_abs_delay_samples": np.nan, "mean_signed_delay_samples": np.nan,
            "mean_abs_delay_sec": np.nan, "mean_signed_delay_sec": np.nan,
            "false_switches_per_hour": 0.0,
            "matched_pairs": [],
        }
    if n_t == 0:
        return {
            "tp": 0, "fp": n_p, "fn": 0,
            "precision": 0.0, "recall": np.nan, "f1": 0.0,
            "mean_abs_delay_samples": np.nan, "mean_signed_delay_samples": np.nan,
            "mean_abs_delay_sec": np.nan, "mean_signed_delay_sec": np.nan,
            "false_switches_per_hour": float(n_p / max(total_duration_hours, 1e-6)),
            "matched_pairs": [],
        }
    if n_p == 0:
        return {
            "tp": 0, "fp": 0, "fn": n_t,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mean_abs_delay_samples": np.nan, "mean_signed_delay_samples": np.nan,
            "mean_abs_delay_sec": np.nan, "mean_signed_delay_sec": np.nan,
            "false_switches_per_hour": 0.0,
            "matched_pairs": [],
        }

    use_sec_matching = (tolerance_sec is not None)
    fs = float(sampling_rate_hz) if sampling_rate_hz is not None else 5.0

    INF_COST = 1e9
    cost_mat = np.full((n_p, n_t), INF_COST, dtype=np.float64)
    for i, pe in enumerate(pred_events):
        p_t = pe["onset_sample"]
        p_sec = float(pe["onset_sec"]) if "onset_sec" in pe else (float(p_t) / fs)
        for j, te in enumerate(true_events):
            if require_direction and (pe["to_mode"] != te["to_mode"]):
                continue
            if use_sec_matching:
                t_start_sec = float(te["onset_sec"]) if "onset_sec" in te else (float(te.get("onset_sample", te["center_sample"])) / fs)
                t_end_sec = float(te["end_sec"]) if "end_sec" in te else (float(te.get("end_sample", te["center_sample"])) / fs)
                t_center_sec = float(te["center_sec"]) if "center_sec" in te else (0.5 * (t_start_sec + t_end_sec))
                if p_sec < t_start_sec:
                    dist_sec = t_start_sec - p_sec
                elif p_sec > t_end_sec:
                    dist_sec = p_sec - t_end_sec
                else:
                    dist_sec = 0.0
                if dist_sec <= float(tolerance_sec) + 1e-9:
                    cost_mat[i, j] = dist_sec + 1e-3 * abs(p_sec - t_center_sec)
            else:
                t_start = te.get("onset_sample", te["center_sample"])
                t_end = te.get("end_sample", te["center_sample"])
                if p_t < t_start:
                    dist = t_start - p_t
                elif p_t > t_end:
                    dist = p_t - t_end
                else:
                    dist = 0
                if dist <= (tolerance_samples if tolerance_samples is not None else 25):
                    cost_mat[i, j] = float(dist) + 1e-3 * abs(p_t - te["center_sample"])

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost_mat[r, c] < INF_COST * 0.1:
            pe = pred_events[r]
            te = true_events[c]
            p_t = int(pe["onset_sample"])
            t_t = int(te.get("onset_sample", te["center_sample"]))
            signed_delay_samp = p_t - t_t
            p_sec = float(pe["onset_sec"]) if "onset_sec" in pe else (float(p_t) / fs)
            t_sec = float(te["onset_sec"]) if "onset_sec" in te else (float(t_t) / fs)
            signed_delay_sec = p_sec - t_sec
            matched_pairs.append({
                "pred_idx": int(r),
                "true_idx": int(c),
                "pred_onset": p_t,
                "true_onset": t_t,
                "true_end": int(te.get("end_sample", te["center_sample"])),
                "pred_onset_sec": round(p_sec, 4),
                "true_onset_sec": round(t_sec, 4),
                "pred_from_to": [int(pe["from_mode"]), int(pe["to_mode"])],
                "true_from_to": [int(te["from_mode"]), int(te["to_mode"])],
                "exact_pair_match": bool(pe["from_mode"] == te["from_mode"] and pe["to_mode"] == te["to_mode"]),
                "signed_delay_samples": int(signed_delay_samp),
                "abs_delay_samples": int(abs(signed_delay_samp)),
                "signed_delay_sec": float(signed_delay_sec),
                "abs_delay_sec": float(abs(signed_delay_sec)),
            })

    tp = len(matched_pairs)
    fp = n_p - tp
    fn = n_t - tp
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    abs_delays_samp = [m["abs_delay_samples"] for m in matched_pairs]
    sig_delays_samp = [m["signed_delay_samples"] for m in matched_pairs]
    abs_delays_sec = [m["abs_delay_sec"] for m in matched_pairs]
    sig_delays_sec = [m["signed_delay_sec"] for m in matched_pairs]

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mean_abs_delay_samples": float(np.mean(abs_delays_samp)) if abs_delays_samp else np.nan,
        "mean_signed_delay_samples": float(np.mean(sig_delays_samp)) if sig_delays_samp else np.nan,
        "mean_abs_delay_sec": float(np.mean(abs_delays_sec)) if abs_delays_sec else np.nan,
        "mean_signed_delay_sec": float(np.mean(sig_delays_sec)) if sig_delays_sec else np.nan,
        "false_switches_per_hour": float(fp / max(total_duration_hours, 1e-6)),
        "matched_pairs": matched_pairs,
    }


def compute_subject_macro_f1(y_true, y_pred, valid_mask, class_ids):
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    K = len(class_ids)
    conf = np.zeros((K, K), dtype=np.int64)
    for idx_t, c_t in enumerate(class_ids):
        m_t = (yt == c_t)
        if not np.any(m_t):
            continue
        for idx_p, c_p in enumerate(class_ids):
            conf[idx_t, idx_p] = int(np.sum(m_t & (yp == c_p)))

    f1_present = []
    f1_all_zero_div = []
    per_class = {}
    for idx_c, c_id in enumerate(class_ids):
        tp = conf[idx_c, idx_c]
        fn = int(np.sum(conf[idx_c, :])) - tp
        fp = int(np.sum(conf[:, idx_c])) - tp
        support = tp + fn
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_all_zero_div.append(f1)
        if support > 0:
            f1_present.append(f1)
        per_class[int(c_id)] = {
            "support": int(support),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1) if support > 0 else None,
        }

    acc = float(np.mean(yt == yp)) if len(yt) > 0 else np.nan
    return {
        "n_valid_samples": int(len(yt)),
        "accuracy": acc,
        "macro_f1_present_classes": float(np.mean(f1_present)) if f1_present else np.nan,
        "macro_f1_all_classes_zerodiv0": float(np.mean(f1_all_zero_div)) if f1_all_zero_div else np.nan,
        "n_present_classes": len(f1_present),
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }
