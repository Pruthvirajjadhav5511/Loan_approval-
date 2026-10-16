# 🚀 FinRisk AI — Intelligent Credit Scoring & Loan Risk Underwriting Platform

A complete, production-ready **API + Machine Learning + Database Persistence & Retrieval** application built with Python, FastAPI, Scikit-Learn, SQLite/SQLAlchemy 2.0, and a responsive glassmorphic web dashboard.

---

## 🌟 Key Features

1. **Machine Learning Pipeline (Scikit-Learn)**:
   - Uses an ensemble **Random Forest Classifier** wrapped in a Scikit-Learn `Pipeline` with a `ColumnTransformer`.
   - Handles numerical normalization (`StandardScaler`) and categorical encoding (`OneHotEncoder`).
   - Achieves **>94% Accuracy** and **>0.98 ROC-AUC** for predicting loan approvals.
   - Computes **Gini Feature Importances** and generates human-interpretable **decision explainability drivers** (positive/negative impact factors).
   - Supports **hot-reloading & on-demand retraining** via `POST /api/model/retrain` without server restart.

2. **Persistent Database Storage & Retrieval (SQLite + SQLAlchemy 2.0)**:
   - Every single applicant evaluation is automatically stored in the `predictions` table with a unique UUID `reference_id`, full applicant inputs, ML risk metrics, and timestamps.
   - Full CRUD support:
     - `POST /api/predict`: ML scoring & instant database persistence.
     - `GET /api/predictions`: Paginated retrieval, text search (applicant name, UUID, purpose), and filtering (risk tier, approval decision).
     - `GET /api/predictions/{id}`: Single record detail inspection.
     - `PATCH /api/predictions/{id}`: Manual underwriter override & audit notes.
     - `DELETE /api/predictions/{id}`: Record deletion.
   - Model training audit log table (`model_training_logs`) tracking algorithm versioning, accuracy, ROC-AUC, and feature weights.

3. **Real-time Database Analytics (`GET /api/stats`)**:
   - Calculates aggregate portfolio metrics directly via SQL:
     - Total applications evaluated & loan volume
     - Approval rate percentage
     - Average risk score and credit score
     - Risk tier distribution (Low Risk, Moderate Risk, High Risk)
     - Loan purpose distribution and recent evaluation trends

4. **Modern Web Interface**:
   - Modern glassmorphic dark theme with ambient background glow effects.
   - **Interactive Form** with real-time sliders and **Quick Preset Buttons** (🌟 Prime, ⚖️ Moderate, ⚠️ Subprime).
   - **Dynamic SVG Circular Risk Gauge** with animated stroke-dashoffset transitions.
   - **Searchable Database Table** with status badges, pagination, and CSV export.
   - **Interactive Charts** (Chart.js) for portfolio analytics and ML feature importance.
   - Direct link to interactive **Swagger OpenAPI UI** (`/docs`).

---

## 📁 Project Structure

```
mini_project /
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, CORS, routers & static mounts
│   ├── database.py          # SQLAlchemy 2.0 engine, SessionLocal, and DB path
│   ├── models.py            # SQLite ORM models (PredictionRecord, ModelTrainingLog)
│   ├── schemas.py           # Pydantic v2 schemas for requests & responses
│   ├── crud.py              # Database query, search, filtering & aggregation functions
│   └── ml/
│       ├── __init__.py
│       ├── model_pipeline.py# ML inference, explainability & memory cache
│       └── trainer.py       # Dataset generation, model training & serialization
├── static/
│   ├── index.html           # Modern web dashboard interface
│   ├── css/
│   │   └── style.css        # Glassmorphic CSS design system
│   └── js/
│       └── app.js           # UI state, Chart.js integrations & API handlers
├── artifacts/
│   ├── model.joblib         # Serialized Scikit-Learn pipeline
│   └── metadata.json        # Model version, accuracy, ROC-AUC & feature weights
├── tests/
│   ├── conftest.py          # Pytest setup & sys.path configuration
│   └── test_api.py          # Automated tests for API, ML, and Database CRUD
├── finrisk.db               # SQLite database file (auto-generated)
├── run.py                   # Application launch script
├── requirements.txt         # Project dependencies
└── README.md                # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### 1. Launch the Application

From within the `mini_project ` directory:

```bash
python run.py
```

Console output will display:
```
=================================================================
🚀 FinRisk AI - Intelligent Credit Scoring & ML Prediction API
=================================================================
📊 Web Application Dashboard:  http://127.0.0.1:8000/
📑 Interactive Swagger Docs:    http://127.0.0.1:8000/docs
🩺 API Healthcheck:            http://127.0.0.1:8000/api/health
🗄️ Database File:              .../mini_project /finrisk.db
=================================================================
```

### 2. Open the Web Dashboard
Open your browser and navigate to:
👉 **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Application Dashboard |
| `GET` | `/docs` | Interactive Swagger OpenAPI UI |
| `GET` | `/api/health` | Healthcheck (API, Database, ML model) |
| `POST` | `/api/predict` | Run ML inference & persist record in DB |
| `GET` | `/api/predictions` | Retrieve stored records (page, size, search, filters) |
| `GET` | `/api/predictions/{id}` | Retrieve single record by DB ID |
| `PATCH` | `/api/predictions/{id}` | Update review status / add underwriter notes |
| `DELETE` | `/api/predictions/{id}` | Delete record from database |
| `GET` | `/api/stats` | Aggregated portfolio analytics from DB |
| `GET` | `/api/model/info` | ML model version, accuracy, ROC-AUC & feature weights |
| `POST` | `/api/model/retrain` | Retrain ML model and hot-reload in memory |
| `GET` | `/api/model/sample-applicants` | Pre-configured test profiles |

---

## 💻 cURL Examples

### 1. Evaluate Applicant & Store in Database
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_name": "Sarah Connor",
    "age": 34,
    "annual_income": 85000,
    "credit_score": 740,
    "loan_amount": 25000,
    "loan_term_months": 36,
    "employment_years": 5.5,
    "debt_to_income_ratio": 18.5,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "home_improvement",
    "has_prior_default": false
  }'
```

### 2. Retrieve Stored Records with Filters
```bash
# Get page 1 with 10 records filtered by Low Risk
curl "http://127.0.0.1:8000/api/predictions?page=1&size=10&risk_tier=Low+Risk"

# Search by applicant name
curl "http://127.0.0.1:8000/api/predictions?search=Sarah"
```

### 3. Fetch Real-Time Analytics
```bash
curl "http://127.0.0.1:8000/api/stats"
```

### 4. Retrain Model On-Demand
```bash
curl -X POST "http://127.0.0.1:8000/api/model/retrain?samples=5000"
```

---

## 🧪 Running Automated Tests

Run the complete test suite verifying API endpoints, ML inference, database persistence, search, patch, and delete operations:

```bash
pytest -v
```

All 7 test suites pass seamlessly:
```
tests/test_api.py::test_health_check PASSED               [ 14%]
tests/test_api.py::test_predict_and_persist_low_risk PASSED [ 28%]
tests/test_api.py::test_predict_and_persist_high_risk PASSED [ 42%]
tests/test_api.py::test_list_and_search_predictions PASSED  [ 57%]
tests/test_api.py::test_update_and_delete_record PASSED     [ 71%]
tests/test_api.py::test_analytics_endpoint PASSED           [ 85%]
tests/test_api.py::test_model_info_and_retrain PASSED       [100%]
```
