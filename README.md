# Smart Phishing URL Detection System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy-003B57?logo=sqlite&logoColor=white)](https://www.sqlalchemy.org/)
[![Tests](https://img.shields.io/badge/tests-91%20passing-34d399)](#14-testing)
[![License](https://img.shields.io/badge/license-MIT-64748b)](LICENSE)

A full-stack web application that classifies a URL as **Safe**, **Suspicious**
or **Phishing** using a machine-learning model trained on 140,878 real URLs.
It extracts 28 lexical and host-based features from the URL string, scores them
with a Random Forest, converts the probability into a 0–100 risk score, and
explains the verdict with a per-URL counterfactual analysis.

> **The URL is never visited.** The system is a *classifier*, not a crawler:
> it analyses the URL string only. Nothing is fetched, resolved, rendered or
> executed.

---

## Table of contents

1. [Overview](#1-overview) · 2. [Features](#2-features) · 3. [System architecture](#3-system-architecture)
4. [Machine-learning pipeline](#4-machine-learning-pipeline) · 5. [Feature engineering](#5-feature-engineering)
6. [Model comparison](#6-model-comparison) · 7. [Installation](#7-installation) · 8. [Dataset setup](#8-dataset-setup)
9. [Training the model](#9-training-the-model) · 10. [Running the application](#10-running-the-application)
11. [API documentation](#11-api-documentation) · 12. [Screens](#12-screens) · 13. [Project structure](#13-project-structure)
14. [Testing](#14-testing) · 15. [Future improvements](#15-future-improvements) · 16. [Disclaimer](#16-disclaimer)
17. [Interview questions](#17-interview-questions)

---

## 1. Overview

Phishing works because a URL is easy to disguise and hard to read carefully.
`http://appleid.apple.com-verify-account.serv-login.ml/update/login.php` looks
like Apple at a glance, but `apple.com` there is only a *label inside another
domain*. This project measures the structural properties that give such URLs
away — length, hyphen and dot counts, subdomain depth, character entropy,
suspicious keywords, high-abuse TLDs — and lets a trained classifier weigh
them.

**What the system does end to end**

```
User submits URL
   → validation (scheme, length, hostname syntax)
   → canonicalisation (strip scheme + www., keep them as features)
   → 28-feature extraction from the string
   → Random Forest → P(phishing)
   → thresholds → Safe / Suspicious / Phishing, risk score 0-100
   → counterfactual explanation of the top drivers
   → stored in SQLite → dashboard statistics update
```

Every number shown in the interface — the confidence, the risk score, the
dashboard counters, the model-comparison chart — is computed at runtime from
the model or the database. Nothing is hard-coded or mocked.

---

## 2. Features

**Detection**
- Real scikit-learn classifier trained on 140,878 labelled URLs (no rules-only shortcut, no hardcoded verdicts)
- 28 lexical / host-based features, extracted identically at training and serving time
- Three-tier verdict with a 0–100 risk score and Low / Medium / High / Critical severity
- Per-URL explanation: each top feature is re-scored against its training median to *measure* what it did to this prediction

**Application**
- Dark security-operations UI: glass panels, CSS-only risk gauge, responsive layout, no build step
- Analytics dashboard with four Chart.js views (verdict split, 14-day trend, risk distribution, model comparison)
- Searchable, filterable, sortable, paginated scan history
- REST API: `POST /api/analyze`, `GET /api/history`, `GET /api/statistics`, `GET /api/health`
- Every scan persisted to SQLite through SQLAlchemy

**Engineering**
- Application-factory Flask layout with blueprints; no logic in `app.py`
- Environment-driven configuration (`.env`), rotating file + stdout logging
- Friendly error handling everywhere; stack traces never reach the client
- 91 pytest tests covering features, validation, prediction, API, persistence and the training pipeline

---

## 3. System architecture

```
                          ┌────────────────────────────────┐
  Browser  ──── HTML ────►│  Flask (application factory)   │
  API client ─ JSON ─────►│  src/routes/web.py  +  api.py  │
                          └───────────────┬────────────────┘
                                          │ both call one service
                                  src/routes/service.py
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
   src/utils/validators.py     src/features/url_features.py    src/ml/predictor.py
   scheme / length / host      28 features from the string     bundle → P(phishing)
   whitelisting                (never touches the network)     → verdict, risk, drivers
                                          │
                                          ▼
                              src/database/repository.py
                              SQLAlchemy → SQLite (URLAnalysis)
                                          │
                                          ▼
                          dashboard counters, history, /api/statistics
```

**Design decisions worth defending in an interview**

| Decision | Reason |
|---|---|
| One shared service (`analyse_url`) behind both the UI and the API | A verdict can never differ between the two entry points, and logging happens in exactly one place |
| The model bundle stores the feature *names*, not just the estimator | Prediction rebuilds the vector from that list, so adding a feature later cannot silently shift columns |
| Scheme and `www.` stripped before counting characters | The corpus stores canonicalised URLs; without this, a URL you type would be measured differently from the ones the model learned on |
| Zero-variance features dropped at training time | A feature that never varies in the corpus carries no signal (`has_https` here — still extracted and shown as an observed indicator) |
| Prediction thresholds live in config, not in the model | The safe/suspicious/phishing cut-offs are a product decision and can be retuned without retraining |

---

## 4. Machine-learning pipeline

`python src/ml/train.py` runs the whole thing:

1. **Load** `data/phishing_urls.csv` (flexible column/label detection).
2. **Clean** — drop blank URLs, uninterpretable labels and duplicate URLs (duplicates would leak across the split).
3. **Extract** the 28-feature matrix; replace any inf/NaN with 0.
4. **Drop constant columns** and record which were removed.
5. **Split** 80/20, stratified on the label.
6. **Scale** with `StandardScaler` (fitted on the training split only — fitting on all data would leak test statistics). Applied to Logistic Regression; tree models are scale-invariant and use the raw matrix.
7. **Train** Logistic Regression, Decision Tree, Random Forest and Gradient Boosting.
8. **Evaluate** each on the held-out split: accuracy, precision, recall, F1, ROC-AUC, Brier score.
9. **Select** the best model by ROC-AUC (threshold-independent), with F1 as the tie-break.
10. **Calibrate** the winner with isotonic regression (3-fold) and keep the calibrated version *only if* it lowers the Brier score — the interface reports a probability, so that probability has to mean what it says.
11. **Explain** — compute global feature importances (impurity-based for tree models, permutation importance otherwise).
12. **Persist** the bundle to `models/phishing_model.pkl` (estimator, scaler, feature names, training medians, importances, metrics, timestamp), the scaler to `models/scaler.pkl` and the full report to `models/metrics.json`.

### From probability to verdict

The classifier is binary (the labels in every public URL corpus are), so the
three product verdicts come from thresholds on P(phishing), configurable in
`config.py`:

| P(phishing) | Verdict | Risk score | Risk level |
|---|---|---|---|
| `< 0.35` | **Safe** | `round(p × 100)` | Low `<25`, Medium `<50` |
| `0.35 – 0.70` | **Suspicious** | `round(p × 100)` | High `<75` |
| `≥ 0.70` | **Phishing** | `round(p × 100)` | Critical `≥75` |

"Suspicious" is not a third trained class — it is the band where the model is
genuinely uncertain, and the UI says so rather than forcing a confident answer.
`confidence` is the model's confidence in its *binary* call,
`max(p, 1 − p) × 100`, which is why it is never below 50%.

### Explaining a single prediction

Global feature importance answers "what matters on average", not "what
happened to *this* URL". So for every scan the service builds one matrix
containing the real feature vector plus one copy per feature with that feature
reset to its **training-set median**, and scores all of them in a single
`predict_proba` call. The difference is a measured counterfactual:

> *Suspicious keyword count = 4 → +26.6 points of phishing probability
> (versus a URL with the median keyword count)*

That is a real measurement of this prediction, cheap enough to run per request,
and it never claims a cause the model did not actually use. Purely observational
notes ("uses a URL shortener", "plain HTTP") are shown in a **separate** panel
labelled as observations, so the interface never dresses up a rule as a model
decision.

---

## 5. Feature engineering

All 28 features come from `src/features/url_features.py::extract_features(url)`,
which returns an ordered dictionary and never touches the network.

| # | Feature | Type | What it captures |
|---|---|---|---|
| 1 | `url_length` | int | Total length; phishing URLs are padded to push the real domain out of view |
| 2 | `domain_length` | int | Hostname length |
| 3 | `path_length` | int | Path length |
| 4 | `num_dots` | int | Dot count — proxy for domain nesting |
| 5 | `num_hyphens` | int | Hyphens, heavily used in brand impersonation (`secure-paypal-login`) |
| 6 | `num_underscores` | int | Underscores, rare in legitimate hostnames |
| 7 | `num_slashes` | int | Path depth |
| 8 | `num_special_chars` | int | Density of `?&=%@~#$…` |
| 9 | `num_digits` | int | Digit count |
| 10 | `num_subdomains` | int | Subdomain depth excluding `www` |
| 11 | `has_https` | bool | Whether the scheme is HTTPS |
| 12 | `has_ip_address` | bool | Raw IP (incl. hex-obfuscated) instead of a domain name |
| 13 | `has_at_symbol` | bool | `@` — everything before it is discarded by browsers, a classic disguise |
| 14 | `has_double_slash_redirect` | bool | `//` inside the path, an open-redirect pattern |
| 15 | `has_port` | bool | Explicit non-standard port |
| 16 | `num_suspicious_keywords` | int | Hits from 46 credential-harvesting tokens (`login`, `verify`, `wallet`, `webscr`, …) |
| 17 | `num_query_params` | int | Query-parameter count |
| 18 | `num_fragments` | int | Fragment count |
| 19 | `num_percent_encodings` | int | `%XX` sequences used to obfuscate |
| 20 | `domain_entropy` | float | Shannon entropy of the hostname — algorithmically generated domains score high |
| 21 | `url_entropy` | float | Shannon entropy of the whole URL |
| 22 | `digit_ratio` | float | Digits ÷ characters |
| 23 | `letter_ratio` | float | Letters ÷ characters |
| 24 | `longest_token_length` | float | Longest unbroken alphanumeric run |
| 25 | `tld_length` | int | TLD length |
| 26 | `is_suspicious_tld` | bool | Membership of 44 high-abuse TLDs (`.tk`, `.ml`, `.xyz`, `.top`, …) |
| 27 | `is_shortened` | bool | Known URL shortener — the destination is hidden |
| 28 | `has_hyphen_in_domain` | bool | Hyphen inside the registrable domain specifically |

**Two details that matter**

*Canonicalisation.* The scheme and a leading `www.` are stripped before any
counting and recorded separately (`has_https`). Public corpora store URLs in
that normalised form while users type both, so without this step the same URL
would produce different features in training and in production.

*Stable ordering.* `FEATURE_NAMES` is the single source of truth for column
order, and the trained bundle pins the exact subset it was fitted on.
`features_to_vector()` is the only way a matrix is built, in both phases.

---

## 6. Model comparison

Produced by the last run of `python src/ml/train.py` on the bundled dataset
(140,878 URLs, 20% held out = 28,176 test URLs). These are the actual numbers
written to `models/metrics.json` — re-running training regenerates them, and
the dashboard reads the same file.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 76.24% | 77.94% | 73.18% | 75.49% | 85.73% |
| Decision Tree | 85.34% | 86.22% | 84.13% | 85.16% | 91.12% |
| **Random Forest** ★ | **88.11%** | **88.28%** | **87.88%** | **88.08%** | **95.55%** |
| Gradient Boosting | 85.65% | 85.43% | 85.95% | 85.69% | 93.43% |

★ selected automatically on ROC-AUC.

**Confusion matrix — Random Forest, 28,176 held-out URLs**

|  | Predicted legitimate | Predicted phishing |
|---|---|---|
| **Actually legitimate** | 12,445 | 1,643 (false positives) |
| **Actually phishing** | 1,707 (false negatives) | 12,381 |

Isotonic calibration was fitted and evaluated: it moved the Brier score from
**0.0855 to 0.0862**, i.e. slightly worse, so the pipeline kept the
uncalibrated forest. That decision is recorded in `models/metrics.json`.

**Reading these numbers honestly.** ~88% accuracy is what a *URL-string-only*
model achieves on this corpus. It has no access to domain age, WHOIS records,
TLS certificates, hosting reputation or page content — all of which a
commercial product would use. Roughly one URL in eight is misjudged, and the
"Suspicious" band exists precisely because the model is not always confident.
Legitimate URLs with hyphenated domains, deep subdomains or short cryptic paths
are its weak spot.

---

## 7. Installation

```bash
git clone https://github.com/yuvrajnag/shanthi2.git
cd shanthi2

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

Optional configuration:

```bash
cp .env.example .env      # then edit SECRET_KEY, DATABASE_URL, thresholds, ...
```

Requires **Python 3.11+**. Chart.js is vendored in `static/js/vendor/`, so the
dashboard renders without internet access (only the web font is remote, and it
degrades to a system font).

---

## 8. Dataset setup

The repository already ships a ready-to-train dataset at
`data/phishing_urls.csv` (140,878 URLs, 50/50 balanced), so you can skip
straight to training.

To rebuild it from its public sources, or to change its size:

```bash
python scripts/build_dataset.py                 # full balanced corpus (~141k rows)
python scripts/build_dataset.py --rows 40000    # smaller, trains in ~20s
```

To use **your own** dataset, drop a CSV with `url` and `label` columns at
`data/phishing_urls.csv` — the loader auto-detects common column names and
textual labels. See [`data/README.md`](data/README.md) for the full
specification and for why the bundled corpus is re-sampled rather than used
as-is.

---

## 9. Training the model

```bash
python src/ml/train.py
```

Useful flags: `--sample 20000` (stratified subsample), `--dataset PATH`,
`--test-size 0.25`, `--seed 7`, `--output models/other.pkl`.

Abridged output of a real run:

```
================================================================
SMART PHISHING URL DETECTION - MODEL TRAINING
================================================================

Dataset file : data/phishing_urls.csv
Dataset size : 140,878 URLs
  Phishing   : 70,439 (50.0%)
  Legitimate : 70,439 (50.0%)

Extracting features ...
  27 model features retained (of 28 extracted)
  Dropped as constant in this corpus: has_https

Train / test split: 112,702 / 28,176 (test_size=0.2)

----------------------------------------------------------------
MODEL COMPARISON
----------------------------------------------------------------

Random Forest
  Accuracy  :  88.11%
  Precision :  88.28%
  Recall    :  87.88%
  F1 Score  :  88.08%
  ROC-AUC   :  95.55%
  Brier     : 0.0855  (lower is better)
  Fit time  :   5.11s

...

----------------------------------------------------------------
BEST MODEL: Random Forest
----------------------------------------------------------------
  Selected on ROC-AUC (95.55%), F1 tie-break (88.08%)

Calibrating probabilities ...
  Brier score 0.0855 -> 0.0862 (kept uncalibrated model)

Confusion matrix (rows = actual, columns = predicted)
                  Legitimate    Phishing
    Legitimate        12,445       1,643
      Phishing         1,707      12,381

Classification report
              precision    recall  f1-score   support
  Legitimate     0.8794    0.8834    0.8814     14088
    Phishing     0.8828    0.8788    0.8808     14088
    accuracy                         0.8811     28176

Top 10 features by importance
  num_suspicious_keywords    11.24%  #######
  num_hyphens                 9.31%  ######
  longest_token_length        7.86%  #####
  ...

Model saved to   : models/phishing_model.pkl
Scaler saved to  : models/scaler.pkl
Metrics saved to : models/metrics.json
================================================================
```

Training takes roughly 90 seconds on the full dataset (Gradient Boosting
dominates that time) and about 20 seconds with `--sample 40000`.

Model artefacts are git-ignored: they are build outputs, and pickles are tied
to the scikit-learn version that produced them. Retrain after upgrading
scikit-learn.

---

## 10. Running the application

```bash
python app.py
```

Then open **http://127.0.0.1:5000**.

| Route | Page |
|---|---|
| `/` | URL scanner |
| `/dashboard` | Security analytics dashboard |
| `/history` | Searchable scan history |
| `/about` | How it works, measured metrics, known limitations |

Production:

```bash
pip install gunicorn
FLASK_ENV=production SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')" \
  gunicorn "app:create_app()" --bind 0.0.0.0:8000 --workers 2
```

`FLASK_ENV=production` disables debug mode and guarantees that no stack trace
is ever rendered.

If you start the app before training, every page still loads and shows a banner
explaining that the model has not been trained yet.

---

## 11. API documentation

### `POST /api/analyze`

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal.com.secure-login.verify-account.xyz/signin"}'
```

```json
{
  "url": "http://paypal.com.secure-login.verify-account.xyz/signin",
  "prediction": "Phishing",
  "confidence": 87.71,
  "risk_score": 88,
  "risk_level": "Critical",
  "phishing_probability": 0.877087,
  "model": "Random Forest",
  "timestamp": "2026-08-31T14:16:35+00:00",
  "stored": true
}
```

Add `?details=true` to include `features` (all 28), `contributions` (the
measured drivers) and `indicators` (the observational notes).

| Status | When |
|---|---|
| `200` | Analysis completed |
| `400` | Missing, malformed, oversized or unsupported URL; body is not a JSON object |
| `415` | `Content-Type` is not `application/json` |
| `503` | No trained model available |

### `GET /api/history`

```bash
curl "http://127.0.0.1:5000/api/history?prediction=Phishing&sort=risk&order=desc&page=1&per_page=20"
```

Parameters: `search`, `prediction` (`Safe`/`Suspicious`/`Phishing`), `sort`
(`date`/`risk`/`confidence`/`url`), `order` (`asc`/`desc`), `page`, `per_page`
(capped at 100).

```json
{ "items": [ { "id": 1, "url": "…", "prediction": "Phishing", "confidence": 87.71,
               "risk_score": 88, "risk_level": "Critical", "model": "Random Forest",
               "source": "api", "analysis_timestamp": "2026-08-31T14:16:35" } ],
  "page": 1, "pages": 1, "per_page": 20, "total": 1 }
```

### `GET /api/statistics`

Returns `statistics` (counters computed from the database), `trend` (14-day
per-verdict series) and `model` (name, training metadata, measured metrics,
feature importances, thresholds).

### `GET /api/health`

```json
{ "status": "ok", "model_available": true }
```

All errors share one envelope: `{"error": "...", "message": "..."}`.

---

## 12. Screens

The interface has four screens. Rather than embedding screenshots that could
drift out of date, run the app — every screen is populated with live data from
your own scans:

- **Scanner** (`/`) — hero, URL input with example chips, staged loading animation, recent scans
- **Result** (`POST /analyze`) — verdict badge, CSS risk gauge, confidence, ranked counterfactual drivers, observed indicators, full 28-feature vector
- **Dashboard** (`/dashboard`) — five counters, verdict doughnut, 14-day trend, risk histogram, model-comparison bars, feature-importance ranking, deployed-model facts
- **History** (`/history`) — search, verdict filter, four sort keys, risk meters, pagination

---

## 13. Project structure

```
smart-phishing-url-detector/
├── app.py                     # entry point (thin: creates the app, runs it)
├── config.py                  # environment-driven configuration classes
├── requirements.txt
├── pytest.ini
├── .env.example
├── README.md
│
├── data/
│   ├── phishing_urls.csv      # 140,878 labelled URLs (balanced)
│   └── README.md              # dataset spec + sampling rationale
│
├── models/                    # build outputs (git-ignored)
│   ├── phishing_model.pkl     # estimator + scaler + feature names + medians + metrics
│   ├── scaler.pkl
│   └── metrics.json           # full training report, read by the dashboard
│
├── scripts/
│   └── build_dataset.py       # reproducible dataset builder
│
├── src/
│   ├── __init__.py            # application factory, error handlers, template filters
│   ├── features/
│   │   └── url_features.py    # 28-feature extraction (no network access)
│   ├── ml/
│   │   ├── train.py           # full training pipeline / CLI
│   │   └── predictor.py       # model loading, scoring, risk, explanations
│   ├── database/
│   │   ├── models.py          # URLAnalysis model + aggregate queries
│   │   └── repository.py      # persistence + filtered/paginated history
│   ├── routes/
│   │   ├── service.py         # the one analysis workflow shared by web + API
│   │   ├── web.py             # HTML pages
│   │   └── api.py             # JSON API
│   └── utils/
│       ├── validators.py      # URL validation and input hardening
│       └── logging_config.py  # stdout + rotating file logging
│
├── templates/                 # base, index, result, dashboard, history, about, errors
├── static/
│   ├── css/style.css          # design system (no framework, no build step)
│   ├── js/app.js              # scanner interactions
│   ├── js/dashboard.js        # Chart.js views
│   └── js/vendor/             # Chart.js 4.4.1 (MIT), vendored for offline use
└── tests/                     # 91 pytest tests
```

---

## 14. Testing

```bash
pytest                 # 91 tests
pytest -v              # verbose
pytest tests/test_features.py
```

| File | Covers |
|---|---|
| `test_features.py` | Feature completeness and ordering, scheme/`www` canonicalisation, every binary indicator, entropy behaviour, malformed input never raising, compound public suffixes |
| `test_validators.py` | Accepted URL shapes, rejection of `javascript:`/`data:`/`file:`, empty and oversized input, control-character stripping, normalisation |
| `test_predictor.py` | Missing and corrupt model handling, result contract, risk score derived from probability, threshold consistency, determinism, counterfactual drivers matching real features, indicators not over-claiming (`user@host` is not an IP hostname) |
| `test_api.py` | Response contract, status codes (400/415/404/503), JSON errors under `/api`, page rendering, XSS escaping of a hostile URL |
| `test_database.py` | Field-level persistence, truncation, statistics from real rows, SQL-injection attempts through search and sort, pagination, gap-filled trend, ordering |
| `test_train.py` | Column/label detection, cleaning and de-duplication, single-class rejection, constant-column dropping |

Tests that need a trained model skip automatically when
`models/phishing_model.pkl` is absent, so a fresh clone runs green before
training.

---

## 15. Future improvements

- **Host-based signals** — domain age, WHOIS registration date and TLS certificate details would address the model's biggest blind spot; they need network lookups, so they belong behind a separate opt-in service with caching
- **Character-level deep model** — a small CNN/LSTM over raw URL characters typically beats hand-engineered lexical features, at the cost of explainability
- **Brand-impersonation detection** — edit distance between the registrable domain and a list of frequently targeted brands (`paypa1.com`, `app1eid.com`)
- **Threat-feed cross-check** — PhishTank / OpenPhish lookups as a second opinion alongside the model
- **Feedback loop** — let analysts mark a verdict wrong and use those labels for periodic retraining
- **Bulk scanning** — CSV upload with a background worker and a downloadable report
- **Deployment hardening** — rate limiting, API keys, CSP headers, Postgres instead of SQLite

---

## 16. Disclaimer

This is an educational project. Its verdicts are probabilistic and derived from
URL text alone; it will produce both false positives and false negatives (see
the confusion matrix in [section 6](#6-model-comparison)). Do not rely on it as
your only defence against phishing, and do not use it to make security
decisions for other people without a human in the loop.

The application never visits, downloads from, or executes anything at a
submitted URL.

---

## 17. Interview questions

<details>
<summary><b>Why a Random Forest instead of deep learning?</b></summary>

The features are 27 hand-engineered numeric columns, not raw sequences — a
regime where tree ensembles are strong, train in seconds and expose usable
importances. It was selected *automatically* by ROC-AUC against three other
candidates, not chosen upfront. A character-level CNN would likely score higher
on raw URLs but would cost the per-feature explanations this project shows, and
would need far more data and tuning.
</details>

<details>
<summary><b>You report 88% accuracy. Is that good?</b></summary>

For a URL-string-only model, it is a realistic number. The classes are exactly
balanced, so 50% is chance and 88% is well above it. But it also means roughly
one URL in eight is misjudged. Commercial products reach higher by combining
lexical features with domain age, reputation feeds, TLS metadata and page
content — none of which this system has, by design. ROC-AUC of 95.55% says the
*ranking* is considerably better than the accuracy at a fixed 0.5 threshold
suggests, which is exactly why the product uses three bands rather than a hard
binary cut.
</details>

<details>
<summary><b>How do you get three classes from a binary classifier?</b></summary>

Every public URL corpus is labelled binary, so the model is binary. The verdict
layer thresholds P(phishing): below 0.35 Safe, 0.35–0.70 Suspicious, above 0.70
Phishing. "Suspicious" is the band where the model is genuinely uncertain — the
UI says so instead of forcing a confident answer. The thresholds live in
`config.py`, so they can be retuned for a stricter or looser posture without
retraining.
</details>

<details>
<summary><b>How do you know the model isn't learning an artefact of the dataset?</b></summary>

I checked, and the first version *was*. In the source corpus 77% of path-less
URLs were malicious and the benign half had been normalised to bare hosts, so
the model learned "no path ⇒ bad" and "any subdomain ⇒ bad" — it scored
`example.com` 82/100 and `google.com` 58/100. The fix was in the data, not the
model: `scripts/build_dataset.py` stratifies on URL structure (path present,
hostname depth) and balances the classes *within* each structural bucket, so
bucket membership carries no signal. The feature extractor also strips `www.`
and the scheme, because the corpus stores URLs without them. That is the single
most valuable thing I learned building this.
</details>

<details>
<summary><b>Your explanation panel — is that SHAP?</b></summary>

No, it is a leave-one-feature-out counterfactual. For each feature the URL is
re-scored with that feature reset to its training-set median; the change in
P(phishing) is the reported impact. All 28 perturbed rows are scored in a
single `predict_proba` call, so it costs one extra inference per scan. It is
less rigorous than SHAP (it ignores interactions between features) but it is
honest, fast and needs no extra dependency. Global impurity importance is shown
alongside it, clearly labelled as global.
</details>

<details>
<summary><b>Why calibrate, and why did you keep the uncalibrated model?</b></summary>

The UI displays a confidence percentage and derives the risk score directly from
the probability, so the probability has to be meaningful — a forest that says
0.9 should be right about 90% of the time. Random Forests are usually
over-confident near the extremes, so the pipeline fits an isotonic calibrator
and compares Brier scores. Here calibration moved it from 0.0855 to 0.0862 —
slightly worse, so the pipeline kept the raw model. The point is that the
decision was measured and recorded in `metrics.json`, not assumed.
</details>

<details>
<summary><b>Is analysing a phishing URL dangerous?</b></summary>

No, and that is a deliberate architectural property. The URL is only ever
treated as a string: no HTTP request, no DNS resolution, no rendering, no
JavaScript. `urlsplit` parses it and the feature functions count characters.
The worst a hostile URL can do is be long — and length is bounded at 2,048
characters. It is also escaped by Jinja's autoescaping before display, so it
cannot inject markup into the result page.
</details>

<details>
<summary><b>How do you prevent SQL injection and XSS?</b></summary>

Queries go through SQLAlchemy with bound parameters; the search term is bound,
never string-formatted, and `sort`/`order` are matched against a whitelist
dictionary rather than interpolated as column names. There are tests that fire
`'; DROP TABLE url_analysis; --` through both paths and assert the table still
holds its rows. For XSS, Jinja autoescaping handles output, and there is a test
asserting a URL containing `<script>` is not reflected as live HTML.
</details>

<details>
<summary><b>Why does the same URL always give the same answer?</b></summary>

Feature extraction is deterministic and the model is fitted with a fixed random
seed, so scoring has no randomness at all — there is a test asserting two
analyses of the same URL produce identical probabilities. Only retraining
changes a verdict.
</details>

<details>
<summary><b>What would you do differently at 10× the scale?</b></summary>

Swap SQLite for Postgres and move logging off the request path onto a queue;
serve the model behind a small inference service so web workers don't each hold
a 31 MB forest in memory; cache verdicts by URL hash with a TTL; add rate
limiting and API keys; and put the training run in CI so metrics are regenerated
and diffed on every change to the feature extractor.
</details>

---

**Built with** Python · Flask · scikit-learn · pandas · NumPy · SQLAlchemy · SQLite · Chart.js
