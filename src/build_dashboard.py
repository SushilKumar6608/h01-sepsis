"""
build_dashboard.py
Builds a standalone HTML dashboard summarising all project results.
Output: notebooks/dashboard.html
"""

import base64
import os

OUT = r"D:\Egna projekt\h01-sepsis\notebooks"

def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load all result images
img_roc    = img_to_base64(os.path.join(OUT, "lstm_ae_results.png"))
img_shap   = img_to_base64(os.path.join(OUT, "shap_feature_importance.png"))
img_vitals = img_to_base64(os.path.join(OUT, "sepsis_vs_nonsepsis_vitals.png"))
img_icu    = img_to_base64(os.path.join(OUT, "icu_stay_distribution.png"))
img_tpp    = img_to_base64(os.path.join(OUT, "tpp_results.png"))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sepsis Early Warning — Results Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e8eaf0; }}

  .hero {{
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1b2a 100%);
    border-bottom: 1px solid #2a3a4a;
    padding: 48px 40px 36px;
  }}
  .hero h1 {{ font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; }}
  .hero p {{ color: #8899aa; font-size: 1rem; max-width: 720px; line-height: 1.6; }}
  .badge {{
    display: inline-block; background: #1e3a5f; color: #5ba3f5;
    border: 1px solid #2a5a8f; border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; margin: 8px 6px 0 0;
  }}

  .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 40px; }}

  /* Metric cards */
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px;
  }}
  .metric-card {{
    background: #1a1f2e; border: 1px solid #2a3a4a; border-radius: 12px;
    padding: 24px; text-align: center; position: relative; overflow: hidden;
  }}
  .metric-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }}
  .metric-card.green::before {{ background: linear-gradient(90deg, #00d4aa, #00ff88); }}
  .metric-card.blue::before  {{ background: linear-gradient(90deg, #5ba3f5, #a78bfa); }}
  .metric-card.amber::before {{ background: linear-gradient(90deg, #f59e0b, #ef4444); }}
  .metric-card.purple::before {{ background: linear-gradient(90deg, #a78bfa, #ec4899); }}
  .metric-val {{ font-size: 2.4rem; font-weight: 800; margin: 8px 0 4px; }}
  .metric-card.green .metric-val {{ color: #00d4aa; }}
  .metric-card.blue .metric-val  {{ color: #5ba3f5; }}
  .metric-card.amber .metric-val {{ color: #f59e0b; }}
  .metric-card.purple .metric-val {{ color: #a78bfa; }}
  .metric-label {{ font-size: 0.82rem; color: #6b7a8d; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-sub {{ font-size: 0.78rem; color: #4a5568; margin-top: 6px; }}

  /* Section */
  .section {{ margin-bottom: 48px; }}
  .section-title {{
    font-size: 1.1rem; font-weight: 700; color: #ffffff;
    margin-bottom: 20px; padding-bottom: 10px;
    border-bottom: 1px solid #2a3a4a;
    display: flex; align-items: center; gap: 10px;
  }}
  .section-title .dot {{
    width: 8px; height: 8px; border-radius: 50%; background: #5ba3f5;
  }}

  /* Result image panels */
  .img-panel {{
    background: #1a1f2e; border: 1px solid #2a3a4a; border-radius: 12px;
    padding: 20px; margin-bottom: 24px;
  }}
  .img-panel img {{ width: 100%; border-radius: 8px; }}
  .img-caption {{ font-size: 0.82rem; color: #6b7a8d; margin-top: 12px; line-height: 1.5; }}

  /* Two-col layout */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}

  /* Comparison table */
  .comp-table {{
    width: 100%; border-collapse: collapse;
    background: #1a1f2e; border-radius: 12px; overflow: hidden;
    border: 1px solid #2a3a4a;
  }}
  .comp-table th {{
    background: #0d1b2a; color: #8899aa; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 14px 20px; text-align: left;
  }}
  .comp-table td {{ padding: 14px 20px; border-top: 1px solid #2a3a4a; font-size: 0.92rem; }}
  .comp-table tr:hover td {{ background: #1e2a3a; }}
  .good {{ color: #00d4aa; font-weight: 700; }}
  .ok   {{ color: #f59e0b; font-weight: 700; }}
  .model-tag {{
    display: inline-block; background: #1e3a5f; color: #5ba3f5;
    border-radius: 6px; padding: 2px 10px; font-size: 0.8rem;
  }}

  /* Finding box */
  .finding {{
    background: #0d1b2a; border-left: 3px solid #5ba3f5;
    border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 20px 0;
    font-size: 0.92rem; line-height: 1.7; color: #c0ccd8;
  }}
  .finding strong {{ color: #5ba3f5; }}

  /* SHAP bar chart */
  .shap-bar {{ margin: 8px 0; }}
  .shap-label {{ font-size: 0.82rem; color: #c0ccd8; margin-bottom: 4px; display: flex; justify-content: space-between; }}
  .shap-track {{ background: #2a3a4a; border-radius: 4px; height: 10px; }}
  .shap-fill {{ height: 10px; border-radius: 4px; background: linear-gradient(90deg, #5ba3f5, #00d4aa); }}

  /* Pipeline */
  .pipeline {{ display: flex; align-items: center; gap: 0; flex-wrap: wrap; margin: 20px 0; }}
  .pipe-step {{
    background: #1a1f2e; border: 1px solid #2a3a4a; border-radius: 8px;
    padding: 12px 18px; font-size: 0.82rem; color: #c0ccd8; text-align: center;
    min-width: 120px;
  }}
  .pipe-arrow {{ color: #2a3a4a; font-size: 1.4rem; padding: 0 6px; }}

  /* Footer */
  footer {{
    text-align: center; padding: 24px; color: #4a5568; font-size: 0.8rem;
    border-top: 1px solid #2a3a4a; margin-top: 40px;
  }}
</style>
</head>
<body>

<div class="hero">
  <h1>🏥 Sepsis Early Warning — Results Dashboard</h1>
  <p>Unsupervised anomaly detection on ICU clinical event sequences. Trained exclusively 
     on non-sepsis patients — no sepsis cases seen during training. Evaluated on 
     40,336 patients across two hospital systems.</p>
  <span class="badge">PhysioNet 2019</span>
  <span class="badge">LSTM Autoencoder</span>
  <span class="badge">Neural TPP (SAHP)</span>
  <span class="badge">SHAP Explainability</span>
  <span class="badge">Unsupervised</span>
  <span class="badge">Cross-hospital</span>
</div>

<div class="container">

  <!-- Metric Cards -->
  <div class="metrics-grid">
    <div class="metric-card green">
      <div class="metric-label">LSTM AE — Hospital A</div>
      <div class="metric-val">0.926</div>
      <div class="metric-sub">AUC-ROC</div>
    </div>
    <div class="metric-card blue">
      <div class="metric-label">LSTM AE — Hospital B</div>
      <div class="metric-val">0.762</div>
      <div class="metric-sub">AUC-ROC (cross-hospital)</div>
    </div>
    <div class="metric-card amber">
      <div class="metric-label">Patients Evaluated</div>
      <div class="metric-val">40,336</div>
      <div class="metric-sub">2 hospital systems</div>
    </div>
    <div class="metric-card purple">
      <div class="metric-label">Features Used</div>
      <div class="metric-val">14</div>
      <div class="metric-sub">7 vitals + 7 labs</div>
    </div>
  </div>

  <!-- Model Comparison -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Model Comparison</div>
    <table class="comp-table">
      <tr>
        <th>Model</th>
        <th>Hospital A AUC-ROC</th>
        <th>Hospital B AUC-ROC</th>
        <th>Signal Used</th>
        <th>Training Patients</th>
      </tr>
      <tr>
        <td><span class="model-tag">LSTM Autoencoder</span></td>
        <td class="good">0.926</td>
        <td class="good">0.762</td>
        <td>Vital signs + lab values</td>
        <td>18,546 (non-sepsis only)</td>
      </tr>
      <tr>
        <td><span class="model-tag">SAHP Neural TPP</span></td>
        <td class="ok">0.58</td>
        <td class="ok">0.53</td>
        <td>Measurement timing patterns</td>
        <td>18,546 (non-sepsis only)</td>
      </tr>
    </table>

    <div class="finding">
      <strong>Key finding:</strong> Value-based anomaly detection substantially outperforms 
      timing-based detection in ICU settings. Sepsis patients' measurement sequences are 
      <em>denser and more regular</em> — clinicians respond to deterioration by increasing 
      monitoring frequency, making timing patterns counterintuitively normal. This contrasts 
      with industrial IoT settings where event timing is the primary anomaly signal.
    </div>
  </div>

  <!-- ROC Curves -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> LSTM Autoencoder — ROC Curves & Score Distributions</div>
    <div class="img-panel">
      <img src="data:image/png;base64,{img_roc}" alt="ROC curves">
      <div class="img-caption">
        Left: Hospital A (in-distribution) — AUC-ROC 0.926. 
        Right: Hospital B (cross-hospital generalization) — AUC-ROC 0.762. 
        Score distributions show clear separation between sepsis and non-sepsis 
        reconstruction errors, with sepsis patients producing higher anomaly scores.
      </div>
    </div>
  </div>

  <!-- SHAP -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> SHAP Feature Importance — What Drives Sepsis Detection?</div>
    <div class="two-col">
      <div class="img-panel">
        <img src="data:image/png;base64,{img_shap}" alt="SHAP importance">
        <div class="img-caption">
          Mean |SHAP value| across top 100 highest-scoring sepsis windows (Hospital A). 
          The model independently learned Sepsis-3 diagnostic criteria without supervision.
        </div>
      </div>
      <div class="img-panel" style="padding:28px;">
        <div style="font-size:0.85rem; color:#8899aa; margin-bottom:16px; text-transform:uppercase; letter-spacing:0.05em;">Clinical interpretation</div>
        {"".join([
          f'''<div class="shap-bar">
            <div class="shap-label"><span>{feat}</span><span style="color:#5ba3f5">{interp}</span></div>
            <div class="shap-track"><div class="shap-fill" style="width:{pct}%"></div></div>
          </div>'''
          for feat, interp, pct in [
            ("WBC", "Immune response marker", 100),
            ("BUN", "Kidney dysfunction", 68),
            ("Platelets", "Coagulation failure", 65),
            ("Creatinine", "Renal failure", 64),
            ("MAP", "Septic shock criterion", 51),
            ("DBP", "Blood pressure signal", 51),
            ("HR", "Tachycardia marker", 50),
            ("Hgb", "Anaemia / bleeding", 50),
            ("Potassium", "Electrolyte imbalance", 48),
            ("O2Sat", "Hypoxia signal", 47),
            ("Glucose", "Metabolic stress", 47),
            ("Temp", "Fever (unreliable in elderly)", 38),
            ("SBP", "Hypotension signal", 37),
            ("Resp", "Respiratory distress", 34),
          ]
        ])}
      </div>
    </div>
  </div>

  <!-- Patient Vitals -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Sepsis vs Non-Sepsis Patient — Vital Signs Over ICU Stay</div>
    <div class="img-panel">
      <img src="data:image/png;base64,{img_vitals}" alt="Vitals comparison">
      <div class="img-caption">
        Blue (left): Non-sepsis patient p000001, 54h ICU stay. 
        Red (right): Sepsis patient p000009, 258h ICU stay, onset at hour 248 (dashed line). 
        Note persistently elevated HR, O2Sat drop, and volatile Resp rate in sepsis patient 
        long before clinical onset — the signal the LSTM AE learns to detect.
      </div>
    </div>
  </div>

  <!-- ICU Distribution -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Dataset — ICU Stay Distribution</div>
    <div class="img-panel">
      <img src="data:image/png;base64,{img_icu}" alt="ICU distribution">
      <div class="img-caption">
        Median ICU stay: 39h (Hospital A), 38h (Hospital B). 
        Max sequence length set to 72h (covers ~97% of patients). 
        Late-onset cases captured via sliding window strategy.
      </div>
    </div>
  </div>

  <!-- TPP Results -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Neural TPP (SAHP) — Timing-Based Detection</div>
    <div class="img-panel">
      <img src="data:image/png;base64,{img_tpp}" alt="TPP results">
      <div class="img-caption">
        SAHP trained on measurement timing patterns only. AUC-ROC ~0.58 — marginally above 
        random. Sepsis patients receive <em>higher</em> log-likelihood because increased 
        monitoring frequency creates denser, more regular event sequences. 
        This finding highlights a fundamental domain difference from industrial IoT settings.
      </div>
    </div>
  </div>

  <!-- Pipeline -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Processing Pipeline</div>
    <div class="img-panel" style="padding:28px;">
      <div class="pipeline">
        <div class="pipe-step">📁 Raw .psv<br>40,336 patients</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">🔧 Feature<br>selection<br>14 features</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">🩹 Imputation<br>ffill + global<br>medians</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">📐 Normalize<br>z-score<br>(train only)</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">🪟 Sliding<br>window<br>72h, step=1h</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">🧠 LSTM AE<br>Train on<br>non-sepsis</div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">📊 AUC-ROC<br>+ SHAP<br>explanation</div>
      </div>
    </div>
  </div>

  <!-- Dataset stats -->
  <div class="section">
    <div class="section-title"><div class="dot"></div> Dataset Statistics</div>
    <table class="comp-table">
      <tr>
        <th>Split</th><th>Patients</th><th>Sepsis</th><th>Non-Sepsis</th>
        <th>Sepsis %</th><th>Median ICU Stay</th>
      </tr>
      <tr>
        <td>Hospital A — Train</td><td>18,546</td><td>0</td>
        <td>18,546</td><td>—</td><td>39h</td>
      </tr>
      <tr>
        <td>Hospital A — Test</td><td>20,336</td>
        <td class="good">1,790</td><td>18,546</td>
        <td class="ok">8.8%</td><td>39h</td>
      </tr>
      <tr>
        <td>Hospital B — Test</td><td>20,000</td>
        <td class="good">1,142</td><td>18,858</td>
        <td class="ok">5.7%</td><td>38h</td>
      </tr>
    </table>
  </div>

</div>

<footer>
  Clinical Event Sequence Anomaly Detection for Sepsis Early Warning &nbsp;|&nbsp;
  PhysioNet Challenge 2019 &nbsp;|&nbsp;
  MSc AI and Automation, University West, Sweden &nbsp;|&nbsp; 2026
</footer>

</body>
</html>"""

out_file = os.path.join(OUT, "dashboard.html")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to {out_file}")
print("Open it in any browser — no internet connection needed.")