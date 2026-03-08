# VIX TCN + XAI + Event-Warping + C-DEW

## 프로젝트 구조

```
├── pyproject.toml
├── README.md
├── src/
│   └── vix_xai/            ← 핵심 패키지
│       ├── __init__.py
│       ├── config.py        # Config, set_seed, get_device
│       ├── data.py          # load_frame, build_dataloaders, SequenceDataset
│       ├── models.py        # RevIN, TCNEnsemble, CNNEnsemble
│       ├── training.py      # EarlyStopping, train_model
│       ├── eval.py          # evaluate_level_rmse, compute_baselines
│       ├── xai.py           # TimeSeriesGradCAMRegression, extract_multivariate_embeddings
│       ├── utils.py         # plot_*, save/load_model_bundle
│       ├── experiments.py   # search_cnn_config_under_budget, run_experiment_suite
│       ├── event_warping.py # DTW, DTDW, WDTW, embedding DTW
│       ├── posthoc.py       # Post-hoc CAM analysis, matching, deletion tests
│       ├── metrics.py       # DTW metrics over time, AUC, retrieval@k
│       └── concepts.py      # C-DEW, TCAV CV, ConceptDash
├── scripts/
│   ├── run_experiment.py    # Step 1: 모델 학습
│   ├── run_posthoc.py       # Step 2: Post-hoc 분석
│   ├── run_metrics.py       # Step 3: DTW 메트릭
│   └── run_cdew.py          # Step 4: C-DEW + Concept Dashboard
├── tests/
│   ├── test_event_warping.py
│   └── test_smoke_train.py
├── data/raw/
│   ├── timeseries_data.csv      ← 생성 필요 (아래 참조)
│   ├── timeseries_data2.csv     ← 티커 메타데이터
│   └── create_synthetic_data.py ← 합성 데이터 생성기
│
│   # 루트 레벨 호환 shim (기존 노트북 호환용)
├── vix_tcn_revin_xai_plus.py
├── event_warping.py
├── posthoc_analysis_v2.py
├── metrics_over_time_v2.py
└── cdew_concepts_v2.py
```

## 설치

```bash
pip install -e .
# 또는 의존성만:
pip install torch numpy pandas scikit-learn matplotlib scipy tqdm
```

## 실행 순서

### 0. 데이터 준비

합성 데이터로 테스트하려면:
```bash
cd <project_root>
python data/raw/create_synthetic_data.py
```

실제 데이터가 있으면 `data/raw/timeseries_data.csv`에 배치하세요.
필수 컬럼: `날짜`, `VIX`, `SPX`, `Gold`, `WTI`, `DXY`, `KOSPI` 등

### 1. 테스트 실행

```bash
# event_warping 단위 테스트 (torch 불필요)
python tests/test_event_warping.py

# 모델 학습 스모크 테스트
python tests/test_smoke_train.py
```

### 2. 모델 학습 (번들 생성)

```bash
# Quick 테스트 (10 epochs, TCN만)
python scripts/run_experiment.py --csv-path data/raw/timeseries_data.csv --quick

# 풀 실험 (300 epochs, TCN+CNN, 다중 target_mode)
python scripts/run_experiment.py --csv-path data/raw/timeseries_data.csv --epochs 300
```

결과: `outputs/bundles/best_model_bundle.pt`, `outputs/bundles/best_tcn_bundle.pt`

### 3. Post-hoc 분석 (번들 필요)

```bash
python scripts/run_posthoc.py --csv-path data/raw/timeseries_data.csv
```

### 4. DTW 메트릭 분석 (번들 필요)

```bash
python scripts/run_metrics.py --csv-path data/raw/timeseries_data.csv
```

### 5. C-DEW + ConceptDash (번들 필요)

```bash
# C-DEW만
python scripts/run_cdew.py --csv-path data/raw/timeseries_data.csv

# C-DEW + Concept Dashboard
python scripts/run_cdew.py --csv-path data/raw/timeseries_data.csv --run-dashboard
```

### GPU 사용

```bash
python scripts/run_experiment.py --device cuda:0 --csv-path data/raw/timeseries_data.csv
```

## 기존 노트북 호환

프로젝트 루트에 있는 shim 파일들이 기존 `import` 구문을 지원합니다:

```python
# 기존 방식 (여전히 동작)
import vix_tcn_revin_xai_plus as vtrx
import event_warping as ew
import posthoc_analysis_v2 as ph
import metrics_over_time_v2 as met
import cdew_concepts_v2 as cdew_mod

# 새로운 방식
import vix_xai as vtrx
from vix_xai.metrics import run_metrics_over_time
from vix_xai.concepts import run_cdew_analysis
```
