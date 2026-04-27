# Clinical Event Sequence Anomaly Detection for Sepsis Early Warning

Unsupervised anomaly detection on ICU clinical event sequences using LSTM Autoencoder
and Neural Temporal Point Process (SAHP). Trained exclusively on non-sepsis patients —
no sepsis cases seen during training.

**Dataset:** PhysioNet/Computing in Cardiology Challenge 2019
**Patients:** 40,336 ICU patients across two hospital systems
**Task:** Flag patients likely to develop sepsis before clinical onset

---

## Results

| Model | Hospital A AUC-ROC | Hospital B AUC-ROC | Approach |
|---|---|---|---|
| LSTM Autoencoder | **0.926** | **0.762** | Reconstruction error on vital signs + labs |
| SAHP Neural TPP | 0.58 | 0.53 | Log-likelihood of measurement timing patterns |

**Key finding:** Value-based anomaly detection substantially outperforms timing-based
detection in ICU settings. Sepsis patients' measurement sequences are denser and more
regular — clinicians respond to deterioration by increasing monitoring frequency, making
timing patterns counterintuitively normal. This contrasts with industrial IoT settings
where event timing is highly informative.

The cross-hospital AUC drop (0.926 → 0.762) reflects genuine distributional shift
between Hospital A and Hospital B — different monitoring equipment, lab protocols, and
patient populations. The model is normalized on Hospital A statistics only.

---

## SHAP Feature Importance

Top features driving sepsis detection (LSTM AE, Hospital A):

| Rank | Feature | Clinical meaning |
|---|---|---|
| 1 | WBC | White blood cell count — primary infection marker |
| 2 | BUN | Blood urea nitrogen — kidney dysfunction |
| 3 | Platelets | Dropping platelets signal coagulation failure |
| 4 | Creatinine | Kidney failure marker consistent with BUN |
| 5 | MAP | Mean arterial pressure — septic shock criterion |

The model independently learned clinically validated sepsis markers (Sepsis-3 criteria)
without any supervision — purely from reconstruction error patterns.

---

## Methodology

    Raw .psv files (40,336 patients)
             |
             v
    Feature selection (14 features: 7 vitals + 7 labs)
             |
             v
    Imputation (forward-fill → backward-fill → global medians)
             |
             v
    Normalization (z-score, computed on non-sepsis train set only)
             |
             v
    Sliding window segmentation (72h windows, step=1h)
             |
             v
    +---------------------------+    +---------------------------+
    |   LSTM Autoencoder        |    |   SAHP (Neural TPP)       |
    |   Train on normal only    |    |   Train on normal only    |
    |   Reconstruction error    |    |   Event timing likelihood |
    |   = anomaly score         |    |   = anomaly score         |
    +---------------------------+    +---------------------------+
             |                                |
        AUC-ROC 0.926                   AUC-ROC 0.58
             |
             v
    SHAP Explainability
    (WBC, BUN, Platelets, Creatinine, MAP)

### Why unsupervised?

Sepsis labels are expensive, inconsistent across hospitals, and only available
retrospectively. An unsupervised model trained only on normal patients generalizes
better across hospital systems and requires no labelled data at deployment time.

---

## Dataset

Source: PhysioNet Challenge 2019 — https://physionet.org/content/challenge-2019/1.0.0/

Reyna et al., "Early Prediction of Sepsis from Clinical Data", Critical Care Medicine, 2020.

- Hospital A: 20,336 patients | 8.8% sepsis prevalence
- Hospital B: 20,000 patients | 5.7% sepsis prevalence
- Features: 8 vital signs, 26 lab values, 6 demographics
- Median ICU stay: ~39 hours

**Labelling:** Patient-level — all windows from a sepsis patient are labelled positive,
reflecting the clinical task of flagging at-risk patients early, not detecting the
exact onset hour.

---

## Project Structure

    h01-sepsis/
    +-- data/
    |   +-- raw/              (original .psv files, not tracked in git)
    |   +-- processed/        (normalized numpy arrays)
    |   +-- tpp/              (TPP event sequences)
    +-- src/
    |   +-- preprocessing.py       (feature engineering + windowing)
    |   +-- lstm_ae.py             (LSTM Autoencoder model)
    |   +-- train_lstm_ae.py       (training script)
    |   +-- evaluate_lstm_ae.py    (AUC-ROC evaluation)
    |   +-- explain_lstm_ae.py     (SHAP explainability)
    |   +-- tpp_preprocessing.py   (TPP event sequence conversion)
    |   +-- tpp_data_converter.py  (EasyTPP format converter)
    |   +-- train_tpp.py           (SAHP training via EasyTPP)
    |   +-- evaluate_tpp.py        (TPP evaluation)
    +-- configs/
    |   +-- sahp.yaml              (EasyTPP training config)
    +-- models/                    (saved model checkpoints)
    +-- notebooks/
    |   +-- 01_exploration.ipynb
    |   +-- lstm_ae_results.png
    |   +-- shap_feature_importance.png
    |   +-- tpp_results.png
    |   +-- dashboard.html
    +-- README.md

---

## Reproducing Results

Step 1 — Setup environment

    conda create -n h01-sepsis python=3.10 -y
    conda activate h01-sepsis
    pip install torch numpy pandas scikit-learn matplotlib seaborn shap jupyter tqdm easy-tpp

Step 2 — Download dataset from PhysioNet Challenge 2019 and extract to data/raw/

Step 3 — Run pipeline in order

    python src/preprocessing.py
    python src/train_lstm_ae.py
    python src/evaluate_lstm_ae.py
    python src/explain_lstm_ae.py
    python src/tpp_preprocessing.py
    python src/tpp_data_converter.py
    python src/train_tpp.py
    python src/evaluate_tpp.py

---

## Relevance to Healthcare AI

This project directly addresses priorities stated by Swedish healthcare providers for 2026:

- Early warning: flags sepsis risk hours before clinical onset
- Explainability: SHAP identifies which lab values drove the alert
- No labelled data required: trained only on normal patient sequences
- Cross-hospital generalization: evaluated on two distinct hospital systems

Sepsis affects approximately 50 million people annually and carries 20-30% mortality.
Every hour of delayed treatment increases mortality by approximately 7%.

---

## Connection to Industrial IoT Thesis

This project extends the methodology from an MSc thesis on unsupervised anomaly
detection in industrial IoT door event sequences (ASSA ABLOY Entrance Systems, 2026)
to clinical ICU event sequences — same architecture (LSTM AE + Neural TPP), new domain,
stronger societal impact. The contrasting TPP results between domains reveal a fundamental
difference: industrial event timing is highly structured and anomaly-revealing, while
clinical measurement timing reflects clinician response rather than patient physiology.

---

## Author

Sushil | MSc AI and Automation, University West, Sweden | 2026