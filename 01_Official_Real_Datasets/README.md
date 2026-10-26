# Dataset acquisition

Raw HAPT and HARTH archives are not stored in this repository. Run `python 01_Official_Real_Datasets/download_datasets.py` from the repository root. The helper fetches only from the official UCI URLs, verifies each archive SHA-256 against the published provenance table, then extracts to local ignored directories.

Dataset records and attribution:

- HAPT: Reyes-Ortiz et al. (2015), UCI dataset 341, DOI [10.24432/C54G7M](https://doi.org/10.24432/C54G7M), CC BY 4.0.
- HARTH: Logacjov et al. (2021), UCI dataset 779, DOI [10.24432/C5NC90](https://doi.org/10.24432/C5NC90), CC BY 4.0; related dataset paper, DOI [10.3390/s21237853](https://doi.org/10.3390/s21237853).

See `data_provenance.csv` for the access dates, archive sizes, and expected SHA-256 values used by this experiment package. Dataset terms and attribution are governed by their official records.
