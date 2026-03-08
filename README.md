# Economic Time-Series Explainability with TCN, Grad-CAM, Event Warping, and C-DEW

This repository contains the code used to reproduce our experiments on VIX forecasting and explainability with:

- TCN / CNN ensembles with RevIN
- Grad-CAM-based temporal attribution
- Post-hoc event/control analysis
- Event-warping metrics (DTW, DTDW, embedding-DTW, weighted DTW)
- C-DEW (Concept-weighted Dynamic Time Warping)
- Multi-concept dashboard analysis

The goal of this repository is to provide a clean and reproducible research codebase for explainable financial time-series modeling.

---

## Repository Structure

```text
.
project_root/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ README.md
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_train_tcn_revin.ipynb
│  ├─ 03_xai_gradcam.ipynb
│  ├─ 04_posthoc_analysis.ipynb
│  └─ 05_metrics_cdew.ipynb
├─ scripts/
│  ├─ run_experiment.py
│  ├─ run_posthoc.py
│  ├─ run_metrics.py
│  └─ run_cdew.py
├─ src/
│  └─ vix_xai/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ models.py
│     ├─ training.py
│     ├─ eval.py
│     ├─ xai.py
│     ├─ utils.py
│     ├─ experiments.py
│     ├─ event_warping.py
│     ├─ posthoc.py
│     ├─ metrics.py
│     └─ concepts.py
├─ tests/
│  ├─ test_data.py
│  ├─ test_event_warping.py
│  └─ test_smoke_train.py
├─ outputs/                        
├─ vix_tcn_revin_xai_plus.py       
├─ posthoc_analysis_v2.py          
├─ metrics_over_time_v2.py         # shim
├─ cdew_concepts_v2.py             # shim
└─ event_warping.py                # shim
