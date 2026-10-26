# IBC-SHMM: code and aggregated results

This repository contains the KBS-oriented IBC-SHMM experiment code, protocol records, selected aggregate results, and the figures used in the current KBS review manuscript. LaTeX and Word manuscript files, author/contact forms, cover letters, internal review records, raw sensor data, subject-level metrics, event-level matches, and private execution logs are intentionally not included.

## Evidence status

The originally frozen protocol (`00_Protocol/study_protocol.yaml`) describes the preregistered `IBC-SHMM-LegacySat` variant. The repaired `IBC-SHMM` variant and its associated threshold/mode-score changes are recorded in the separate amendment (`protocol_amendment_v2_repaired_m3.yaml`) as post-hoc diagnostic/exploratory, after examination of external IMU streams. Results for the repaired method must therefore be interpreted as exploratory; this repository does not claim independent confirmatory validation or a successful reproduction run on a clean external machine.

The result table is a KBS-filtered subset of the frozen v7 result matrix. It excludes the Neurocomputing-only rows and the internal source-path column. The exact source matrix SHA-256 before this transformation is recorded in `03_Results/README.md`. The included data are aggregate metrics; per-subject and event-level rows are deliberately omitted.

## Datasets

The code expects the official UCI HAPT (dataset 341) and HARTH (dataset 779) archives. The archives are not mirrored here. Use the helper below to download from UCI, check the expected SHA-256 values recorded in the provenance table, and extract into the expected local layout:

```bash
python 01_Official_Real_Datasets/download_datasets.py
```

The UCI catalog lists both source datasets under CC BY 4.0: [HAPT, DOI 10.24432/C54G7M](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) and [HARTH, DOI 10.24432/C5NC90](https://archive.ics.uci.edu/dataset/779/harth). Please cite the dataset creators and follow the dataset records' attribution terms. Raw archives remain local and are excluded by `.gitignore`.

## Environment

Reference environment recorded in the experiment materials: Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3. Install the listed dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction workflow

From the repository root, after the dataset downloader succeeds:

```bash
python 02_Adapters_and_Evaluation/tests/run_g3_g4_verification.py
python 02_Adapters_and_Evaluation/run_r0_r1_r2_experiments.py --recompute-all --no-cache
```

The first step runs the adapter/fairness and future-perturbation causality checks and writes local gate reports consumed by the experiment runner. The second runs the zero-cache R0/R1/R2 and ablation suite. It is computationally intensive, especially the HARTH full-stream evaluation. Generated subject-level outputs and run logs are ignored by Git; do not force-add them to a public commit.

The runner and adapters use the repository root by default. Set `IBC_SHMM_REPO_ROOT` only if you place the expected data/protocol/result folders elsewhere.

## Repository contents

- `00_Protocol/`: frozen protocol and the explicitly post-hoc amendment.
- `01_Official_Real_Datasets/`: provenance and download helper; no raw participant records.
- `02_Adapters_and_Evaluation/`: adapters, online evaluator, baselines, fault protocol, experiment runner, and gate test source.
- `03_Results/`: filtered aggregate result matrix and its provenance notes.
- `04_Figures/`: the five KBS figures from the current review package.

## Licensing

No license is included for the author's code or derived result artifacts pending the author's explicit license choice. Public visibility alone does not grant reuse rights. Third-party dataset licensing is separate; see the official UCI pages linked above.
