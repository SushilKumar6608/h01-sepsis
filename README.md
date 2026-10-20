# H01 v2 - Sepsis Early Warning: Clinical Event Sequence Anomaly Detection

Unsupervised sepsis early-warning system built on the PhysioNet/Computing in
Cardiology Challenge 2019 dataset (40,336 patients across two hospitals).
The core model is trained **only on non-sepsis patients**: sepsis presentations
are detected as anomalous reconstructions rather than learned from positive
labels.

This repository contains two layers built in sequence:

1. **ML core** (`src/`) - LSTM Autoencoder trained on Set A non-sepsis patients.
2. **Integration layer** (`integration/`) - HL7v2 ingestion, canonical LOINC
   mapping, FHIR parity checking, and a scoring API that wraps the model behind
   a REST endpoint returning FHIR R4 RiskAssessment resources.

> **Synthetic / public data only.** All data is from the public PhysioNet
> challenge. No real patient records are used or referenced anywhere in this
> project.

---

## Results

| Model | AUC-ROC (Hospital A) | AUC-ROC (Hospital B) |
|---|---|---|
| LSTM Autoencoder (reconstruction error) | **0.926** | **0.762** |
| SAHP Neural TPP (log-likelihood of timing) | 0.58 | 0.53 |

The LSTM AE outperforms the timing-based approach. Sepsis patients receive
denser, more regular monitoring - their measurement timing looks normal. The
drop from Set A to Set B reflects distribution shift: normalisation statistics
were computed on Set A non-sepsis windows only.

**SHAP top drivers** (features most associated with elevated reconstruction
error): WBC, BUN, Platelets, Creatinine, MAP.

---

## Architecture

```
HL7v2 ORU^R01 message
        |
        v
POST /ingest/hl7v2          # integration/ingest/api.py
        |
        |- parse (integration/ingest/hl7v2_parser.py)
        |- store raw + SHA-256 dedup (integration/ingest/raw_store.py)
        |- map to LOINC (integration/canonical/mapper.py)
        |
        v
POST /score?control_id=...  # same api.py
        |
        |- re-fetch raw from SQLite
        |- re-parse + re-map
        |- build (72, 14) feature sequence (integration/scoring/scorer.py)
        |- LSTM AE reconstruction error (src/lstm_ae.py)
        |- risk level: LOW / MODERATE / HIGH
        |
        v
FHIR R4 RiskAssessment (integration/scoring/fhir_risk_assessment.py)
```

---

## ML Core (`src/`)

### Model

**LSTM Autoencoder** - encodes a 72-hour sliding window of 14 vital/lab
features into a compact latent representation and reconstructs it. Reconstruction
error is used as the anomaly score.

| Hyperparameter | Value |
|---|---|
| Architecture | LSTM Autoencoder |
| Input features | 14 (7 vitals, 7 labs) |
| Sequence length | 72 hours |
| Hidden dim | 64 |
| Layers | 2 |
| Dropout | 0.2 |

**Features (in model order):**
HR, O2Sat, Resp, SBP, MAP, Temp, Glucose, DBP, Creatinine, BUN, Potassium,
WBC, Platelets, Hgb

### Risk thresholds (tuned on Set A test windows)

| Risk level | Reconstruction error |
|---|---|
| LOW | < 0.10 |
| MODERATE | 0.10 - 0.25 |
| HIGH | >= 0.25 |

### Preprocessing

- Forward/backward fill within each patient time series, then global median
  imputation for remaining gaps.
- Z-score normalisation using mean and standard deviation computed on Set A
  non-sepsis training windows (`data/processed/mean_A.npy`,
  `data/processed/std_A.npy`).
- 72-hour sliding windows with 1-hour steps. Labels are patient-level: every
  window from a sepsis patient is treated as positive.

### Training

```
conda activate D:\conda-envs\h01-sepsis
python src/train.py
```

### Evaluation

```
python src/evaluate.py
```

---

## Integration Layer (`integration/`)

Built on top of the trained model to demonstrate a realistic clinical
integration path. All data remains synthetic; the HL7v2 messages used in
tests are hand-crafted.

### Phase 1 - HL7v2 ingestion

`POST /ingest/hl7v2` accepts a raw HL7v2 ORU^R01 message (plain text body),
parses it, deduplicates by SHA-256 hash, stores it in SQLite, and returns a
summary including the `control_id` needed for scoring.

- Parser: `integration/ingest/hl7v2_parser.py`
- Storage: `integration/ingest/raw_store.py` (SQLAlchemy / SQLite)
- Idempotent: re-submitting the same message returns HTTP 409 with the
  existing record's ID.

### Phase 2 - Canonical mapping

Maps local hospital observation codes to LOINC codes using a YAML code map
(`integration/canonical/code_map.yaml`). Unmapped codes are preserved with a
warning so no data is silently dropped.

- Mapper: `integration/canonical/mapper.py`

### Phase 3 - FHIR parity checker

Validates that a FHIR R4 Observation resource carries an equivalent
observation to the parsed HL7v2 source. Reports coverage gaps and mismatches.

- Checker: `integration/fhir/parity_checker.py`

### Phase 4 - Scoring pipeline

`POST /score?control_id=<id>` retrieves the raw HL7v2 by control_id,
re-parses and re-maps it, builds a (72, 14) feature sequence, runs the LSTM
AE, and returns a **FHIR R4 RiskAssessment** resource.

- Sequence builder + scorer: `integration/scoring/scorer.py`
- FHIR output: `integration/scoring/fhir_risk_assessment.py`

**Example RiskAssessment response:**

```json
{
  "resourceType": "RiskAssessment",
  "id": "MSG001",
  "status": "final",
  "subject": { "reference": "Patient/P001" },
  "encounter": { "reference": "Encounter/MSG001" },
  "occurrenceDateTime": "2025-03-15T08:00:00+00:00",
  "basis": [{ "reference": "Device/lstm-ae-sepsis-lstm_ae_best.pt" }],
  "prediction": [{
    "outcome": { "text": "Sepsis risk - MODERATE" },
    "probabilityDecimal": 0.1423,
    "rationale": "LSTM Autoencoder reconstruction error; thresholds: LOW < 0.10, MODERATE 0.10-0.25, HIGH >= 0.25"
  }],
  "note": [{ "text": "Features used: 14; features imputed: 0; model: lstm_ae_best.pt" }]
}
```

---

## Setup

### Prerequisites

- Anaconda / Miniconda
- Python 3.10+
- CUDA optional (CPU inference is fast enough for the model size)

### Create environment

```bash
conda create -p D:\conda-envs\h01-sepsis python=3.10 -y
conda activate D:\conda-envs\h01-sepsis
pip install -r requirements.txt
```

### Download data

Data is from the PhysioNet/CinC 2019 Challenge (public):
https://physionet.org/content/challenge-2019/1.0.0/

Extract Set A to `data/raw/training_setA/` and Set B to
`data/raw/training_setB/`.

---

## Reproduction - ML core

Run these scripts in order from the repo root (Anaconda Prompt):

```
1. python src/load_data.py
2. python src/eda.py
3. python src/preprocessing.py
4. python src/train.py
5. python src/evaluate.py
6. python src/shap_analysis.py
7. python src/neural_tpp.py
8. python src/evaluate_tpp.py
```

---

## Running the API

```bash
cd D:\Egna projekt\h01-sepsis
set DATABASE_URL=sqlite:///integration/h01v2_dev.db
python -m uvicorn integration.ingest.api:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

**Ingest a message:**
```bash
curl -X POST http://localhost:8000/ingest/hl7v2 \
  -H "Content-Type: text/plain" \
  --data-binary @path/to/message.hl7
```

**Score it:**
```bash
curl -X POST "http://localhost:8000/score?control_id=MSG001"
```

---

## Tests

```bash
conda activate D:\conda-envs\h01-sepsis
pytest integration/tests/ -v
```

72 tests across 4 phases:

| Test file | Tests | Covers |
|---|---|---|
| `test_phase1.py` | 24 | HL7v2 parsing, raw storage, ingest endpoint |
| `test_phase2.py` | 12 | LOINC mapping, unmapped code handling |
| `test_phase3.py` | 12 | FHIR parity checker |
| `test_phase4.py` | 24 | Sequence builder, scorer thresholds, FHIR output shape, real-model integration |

Phase 4 integration tests (`TestScorerIntegration`) run automatically when
`models/lstm_ae_best.pt`, `data/processed/mean_A.npy`, and
`data/processed/std_A.npy` are present; otherwise they are skipped.

---

## Repository structure

```
h01-sepsis/
|- src/                        # ML core (read-only from integration layer)
|  |- lstm_ae.py               # LSTM Autoencoder model definition
|  |- preprocessing.py         # feature order, imputation, normalisation
|  |- train.py
|  |- evaluate.py
|  |- shap_analysis.py
|  |- neural_tpp.py
|  |- evaluate_tpp.py
|- integration/
|  |- ingest/
|  |  |- api.py                # FastAPI app (POST /ingest/hl7v2, POST /score)
|  |  |- hl7v2_parser.py
|  |  |- raw_store.py          # SQLAlchemy ORM + dedup
|  |- canonical/
|  |  |- mapper.py             # local code -> LOINC mapping
|  |  |- code_map.yaml
|  |  |- db.py                 # SQLAlchemy session factory
|  |- fhir/
|  |  |- parity_checker.py
|  |- scoring/
|  |  |- scorer.py             # sequence builder + LSTM AE wrapper
|  |  |- fhir_risk_assessment.py
|  |- tests/
|  |  |- test_phase1.py
|  |  |- test_phase2.py
|  |  |- test_phase3.py
|  |  |- test_phase4.py
|- models/
|  |- lstm_ae_best.pt          # trained weights (not tracked in git)
|- data/
|  |- raw/                     # PhysioNet challenge data (not tracked in git)
|  |- processed/
|  |  |- mean_A.npy            # Set A non-sepsis training mean
|  |  |- std_A.npy             # Set A non-sepsis training std
|- notebooks/
|  |- 01_exploration.ipynb
|- requirements.txt
```

---

## Context

Built as a portfolio project for a healthcare Forward Deployed AI Engineer
interview. The project demonstrates:

- Unsupervised anomaly detection applied to clinical time series.
- A production-style integration layer: standards-based ingestion (HL7v2),
  canonical coding (LOINC/UCUM), interoperability output (FHIR R4), and a
  fully-tested REST API.
- Cross-hospital generalisation analysis and honest reporting of distribution
  shift (Set A -> Set B AUC drop from 0.926 to 0.762).

The approach follows on from MSc thesis work on unsupervised anomaly detection
in industrial IoT sequences, adapted to clinical settings.
