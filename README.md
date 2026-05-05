# SDP_SynergyLens

SDP_SynergyLens is a Flask workspace for drug-combination synergy screening. The migrated UI adds the useful experience ideas from the reference project while keeping this repository's model artifacts, feature construction, data format, and API contracts as the source of truth.

## What The App Includes

| Area | Current behavior |
| --- | --- |
| Prediction | Autocomplete valid drugs, choose a valid cell line, call `/api/predict`, and show score, label, confidence-style boundary note, safety note, history, and report download. |
| Explainability | Call `/api/explain` and render SHAP-style chart, positive/negative contributor cards, totals, feature table, readable summary, and safety note. |
| Chat | Dynamic prediction chat uses `/api/chat`; Gemini keys fail over safely, then local fallback answers are used only after all configured keys fail. |
| Project Assistant | Browser-only static helper for project facts, CSV format, score bands, endpoints, SHAP usage, and safety limits. |
| Drug Info | `/api/drug_info/<id>` returns local drug name and optional PubChem metadata; blocked PubChem calls return renderable `n/a` metadata instead of breaking the UI. |
| Batch | `/api/batch` still expects current CSV columns and returns CSV. The UI adds staged upload, validation, sample CSV download, stats, horizontal table scroll, and output download. |
| Transparency | Additive helper endpoints power backend status, system summary, and deployed artifact transparency sections. |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Gemini Configuration

Gemini is optional. The chat endpoint reads keys from:

- comma-separated `GEMINI_API_KEYS`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- optional local `.gemini_keys.local`

Keys are tried in that order, placeholders and duplicates are ignored, and the built-in chat guide is used if every Gemini key fails. Set `GEMINI_MODEL_NAME` to choose the model; `GEMINI_MODEL` is still accepted for older local configs. Keys are not returned to the frontend or printed in logs. `.env` and `.gemini_keys.local` are ignored by Git.

## API Reference

### UI helper endpoints

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Backend readiness badge with drug/cell/feature counts, model status, SHAP availability, chat backend label, and errors. |
| `GET /api/cell-lines` | Valid cell-line options for dropdowns and validation. |
| `GET /api/system-summary` | Current route list, prediction flow, and score thresholds. |
| `GET /api/model-performance-summary` | Deployed model/data counts. Performance metrics are marked unavailable unless present in current repo artifacts. |

### Existing core endpoints

| Route | Contract |
| --- | --- |
| `GET /api/drugs?q=<query>` | Returns drug autocomplete records. `limit=all` is an additive UI helper for local validation. |
| `POST /api/predict` | JSON body: `drug1_id`, `drug2_id`, `cell_line`, optional `cancer_type`. Returns score, label, level, color, and display names. |
| `POST /api/explain` | Same body as predict. Returns `features`, `prediction`, and `base_value` when explanation support is available. |
| `POST /api/chat` | Predict body plus `question`. Returns answer/context plus safe backend metadata such as `used_fallback` and `attempted_key_count`. |
| `GET /api/drug_info/<drug_id>` | Returns local name, structure URL, and PubChem metadata when reachable. |
| `POST /api/batch` | Multipart CSV upload under `file`; returns a CSV download. |

## Batch CSV Format

Required columns:

```csv
drug1_id,drug2_id,cell_line
```

Optional column:

```csv
cancer_type
```

Example:

```csv
drug1_id,drug2_id,cell_line,cancer_type
740,752,OVCAR-3,Ovarian Cancer
```

The frontend can normalize a few old-style aliases before upload, but the backend remains on the current SDP_SynergyLens format.

## Score Bands

| Score | Label |
| --- | --- |
| `> 10` | Strong Synergy |
| `> 5` and `<= 10` | Moderate Synergy |
| `> 0` and `<= 5` | Mild Synergy |
| `> -5` and `<= 0` | Neutral / Additive |
| `<= -5` | Antagonistic |

## Verification

```bash
python -m py_compile app.py verify_app.py
node --check static/js/app.js
python verify_app.py
```

For deterministic chat verification, set Gemini variables to placeholders before running `verify_app.py`.

## Safety Note

Predictions, confidence notes, SHAP values, and chat answers are research screening aids only. They are not clinical advice, biological proof, or a substitute for experimental validation.
