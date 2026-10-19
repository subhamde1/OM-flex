# om-flex

Replication code and computational materials for the manuscript **“Speed-for-Flexibility Contracts for AI Data-Center Interconnection under Private Information and Operational Uncertainty.”**

## Purpose

This repository is maintained to support **transparency, reproducibility, and independent verification** of the computational analyses reported in the paper. It contains the executable Python programs used for the synthetic experiments, numerical checks, public-trace calculations, and sustained-capacity robustness analysis, together with the software requirements, trace protocol, and integrity information needed to reproduce the reported results.

The repository is a replication resource for the paper; it is not operational grid-control software and does not constitute engineering certification of electrical flexibility.

## Repository contents

| File | Purpose | Manuscript mapping |
|---|---|---|
| `synthetic_replication.py` | Reproduces the synthetic model, calibration, scheduling example, sensitivity analyses, finite-menu calculations, and numerical verification checks. | Appendices E–F |
| `trace_compute.py` | Downloads and verifies the public Alibaba GPU 2020 task archive when absent, applies the declared filters and transformations, solves the scheduling problems, and produces the public-trace menu and reliability results. | Appendix G |
| `sustained_capacity_stress.py` | Implements the sustained-capacity stress robustness experiment using the trace-processing functions in `trace_compute.py`. | Appendix H |
| `synthetic_results.json` | Reference JSON output from the synthetic replication program. | Synthetic computational results |
| `replication/workload_distortion.py` | Exact rational workload regimes, distortion comparisons, geometric pooling, and independent optimization diagnostics. | Appendix I; Sections 7.7–7.8 and 8.14–8.15 |
| `replication/workload_distortion_results.json` | Stored synthetic results and verification diagnostics for the new workload-distortion study. | Appendix I |
| `replication/build_workload_figures.py` | Generates Figures 17–18 as vector PDFs and 600-dpi PNGs from the synthetic results. | Figures 17–18 |
| `computed/trace_results.json` | Stored reference output for the existing eight public-trace protocols. | Appendix G |
| `computed/stress_results.json` | Stored reference output for the existing sustained-capacity stress analysis. | Appendix H |
| `requirements.txt` | Python package versions used for replication, figure generation, and PDF verification. | Computational environment |
| `trace_protocol.md` | Concise statement of the public-trace construction and sustained-stress protocol implemented by the code. | Appendices G–H |
| `SHA256SUMS.txt` | SHA-256 checksums for the original core files, new workload-study files, documentation, and stored outputs. | Integrity verification |

## Software environment

The replication programs were prepared for:

- Python 3.12.14
- NumPy 2.3.5
- SciPy 1.17.0
- Matplotlib 3.10.8
- PyMuPDF 1.26.6 (used by PDF verification in the complete manuscript package)

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the synthetic study

Run:

```bash
python synthetic_replication.py > synthetic_results.json
```

The script is self-contained: all model inputs are declared in the program, and the synthetic simulation uses seed `20260907`. It does not require an external dataset.

## Reproducing the new workload-distortion study

Keep the following three files together in the repository's `replication/` subdirectory:

- `workload_distortion.py`
- `workload_distortion_results.json`
- `build_workload_figures.py`

From the repository root, run:

```bash
python replication/workload_distortion.py
python replication/build_workload_figures.py
```

The first command writes `replication/workload_distortion_results.json`. It uses exact rational arithmetic for the structural calculations and separate numerical optimizations and scheduling LPs for diagnostics. The second writes `Pic17.pdf`, `Pic17.png`, `Pic18.pdf`, and `Pic18.png` to the repository root. Keep the `replication/` folder structure because the figure program resolves its input relative to that folder.

This study is fully synthetic. It introduces no company dataset and uses no Alibaba 2023 or 2026 trace. Its examples establish possibilities under the stated model assumptions rather than estimating empirical prevalence. Finite numerical checks support the analytical proofs and do not replace them.

The original three root-level programs and `synthetic_results.json` remain byte-identical to commit `97589d3beb17d6b0349534359213aa5a0f311de1`. The new files are additions to that replication record.

## Reproducing the public-trace study

Run:

```bash
python trace_compute.py
```

The program downloads the Alibaba GPU 2020 task archive **only if it is absent**, stores it under `raw/`, verifies its SHA-256 digest, applies the declared cohort filters and scheduling assumptions, and writes the cumulative trace results to:

```text
computed/trace_results.json
```

The source archive used by the program is:

```text
https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz
```

Integrity values enforced by `trace_compute.py`:

```text
Archive SHA-256:
cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5

Extracted pai_task_table.csv SHA-256:
6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de
```

The public trace remains subject to its publisher's terms. This repository does **not** claim ownership of the Alibaba data and should not be interpreted as redistributing it as original study data.

## Reproducing the sustained-capacity stress analysis

Run the sustained-stress program from the same directory as `trace_compute.py`:

```bash
python sustained_capacity_stress.py > computed/stress_results.json
```

Run `trace_compute.py` first so that `computed/` exists. The stress program imports `trace_compute.py`, uses the same trace cohort, fits the stress design and reliability caps using the training period, and then applies the fitted design unchanged to the holdout period. This is follow-up robustness using the previously examined holdout, rather than a new confirmatory validation sample.

## Public-trace interpretation

The Alibaba trace supplies recorded workload timing and resource-request information used to construct the model-based scheduling exercises. It does **not** directly measure electrical headroom, private flexibility cost, contractual service-level agreements, project economics, or hardware-level electrical feasibility. Accordingly, the trace exercise is an operational robustness analysis under the assumptions documented in the manuscript and `trace_protocol.md`, not a field demonstration of the proposed interconnection contract.

## Integrity check

After merging this update into the existing repository, verify the original files and the new additions with:

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows, the equivalent hashes can be checked with `Get-FileHash -Algorithm SHA256` in PowerShell.

## Citation

Please cite the associated manuscript when using this repository.

## License and third-party data

No open-source license is granted for this repository. The repository's original code and documentation remain subject to applicable copyright law unless permission is provided separately by the author.

The Alibaba GPU 2020 trace is third-party data and remains governed by the terms established by its publisher. This repository does not claim ownership of, or grant rights to, that dataset.
