import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app.models import PredictionRecord
from app.schemas import (
    ApplicantInput,
    PredictionRecordResponse,
    PaginatedPredictions,
    RecordUpdate,
    AnalyticsSummary,
    ModelInfoResponse,
    RetrainResponse,
)
from app.crud import (
    create_prediction,
    get_predictions,
    get_prediction_by_id,
    update_prediction,
    delete_prediction,
    get_analytics,
    record_training_log,
    get_latest_training_log,
)
from app.ml.model_pipeline import (
    get_model,
    get_metadata,
    predict_applicant,
    reload_model_artifact,
)
from app.ml.trainer import train_and_export_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finrisk_api")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


def serialize_record(record: PredictionRecord) -> Dict[str, Any]:
    """Helper to deserialize stored JSON fields before returning to client."""
    factors = []
    top_factors_raw = getattr(record, "top_factors", None)
    if top_factors_raw:
        try:
            factors = json.loads(str(top_factors_raw))
        except Exception:
            factors = []

    return {
        "id": record.id,
        "reference_id": record.reference_id,
        "applicant_name": record.applicant_name,
        "age": record.age,
        "annual_income": record.annual_income,
        "credit_score": record.credit_score,
        "loan_amount": record.loan_amount,
        "loan_term_months": record.loan_term_months,
        "employment_years": record.employment_years,
        "debt_to_income_ratio": record.debt_to_income_ratio,
        "home_ownership": record.home_ownership,
        "loan_purpose": record.loan_purpose,
        "has_prior_default": record.has_prior_default,
        "prediction": record.prediction,
        "approval_probability": record.approval_probability,
        "risk_score": record.risk_score,
        "risk_tier": record.risk_tier,
        "suggested_interest_rate": record.suggested_interest_rate,
        "confidence_score": record.confidence_score,
        "top_factors": factors,
        "review_status": record.review_status,
        "notes": record.notes,
        "model_version": record.model_version,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def seed_initial_demo_records(db: Session):
    """Seed initial demo records if the database is currently empty."""
    count = db.query(PredictionRecord).count()
    if count > 0:
        return

    demo_applicants = [
        {
            "applicant_name": "Eleanor Vance",
            "age": 38,
            "annual_income": 125000.0,
            "credit_score": 785,
            "loan_amount": 28000.0,
            "loan_term_months": 36,
            "employment_years": 8.5,
            "debt_to_income_ratio": 16.2,
            "home_ownership": "OWN",
            "loan_purpose": "home_improvement",
            "has_prior_default": False
        },
        {
            "applicant_name": "Marcus Chen",
            "age": 29,
            "annual_income": 68000.0,
            "credit_score": 670,
            "loan_amount": 18000.0,
            "loan_term_months": 48,
            "employment_years": 3.2,
            "debt_to_income_ratio": 29.5,
            "home_ownership": "RENT",
            "loan_purpose": "debt_consolidation",
            "has_prior_default": False
        },
        {
            "applicant_name": "Jackson Reed",
            "age": 45,
            "annual_income": 42000.0,
            "credit_score": 540,
            "loan_amount": 35000.0,
            "loan_term_months": 60,
            "employment_years": 1.0,
            "debt_to_income_ratio": 49.0,
            "home_ownership": "RENT",
            "loan_purpose": "business",
            "has_prior_default": True
        },
        {
            "applicant_name": "Amara Patel",
            "age": 32,
            "annual_income": 98000.0,
            "credit_score": 735,
            "loan_amount": 20000.0,
            "loan_term_months": 36,
            "employment_years": 6.0,
            "debt_to_income_ratio": 21.0,
            "home_ownership": "MORTGAGE",
            "loan_purpose": "education",
            "has_prior_default": False
        },
        {
            "applicant_name": "Liam Gallagher",
            "age": 52,
            "annual_income": 155000.0,
            "credit_score": 810,
            "loan_amount": 45000.0,
            "loan_term_months": 60,
            "employment_years": 14.0,
            "debt_to_income_ratio": 14.0,
            "home_ownership": "OWN",
            "loan_purpose": "home_improvement",
            "has_prior_default": False
        },
        {
            "applicant_name": "Chloe Dupont",
            "age": 26,
            "annual_income": 48000.0,
            "credit_score": 595,
            "loan_amount": 15000.0,
            "loan_term_months": 36,
            "employment_years": 1.5,
            "debt_to_income_ratio": 38.5,
            "home_ownership": "RENT",
            "loan_purpose": "medical",
            "has_prior_default": False
        }
    ]

    for item in demo_applicants:
        app_input = ApplicantInput.model_validate(item)
        ml_out = predict_applicant(app_input.model_dump())
        create_prediction(db, app_input, ml_out)

    logger.info("Seeded %d demo loan predictions into database.", len(demo_applicants))


# Ensure tables exist immediately upon import
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context: initialize tables, guarantee ML model exists and is logged."""
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)

    # Ensure model exists in memory and artifacts
    get_model()
    metadata = get_metadata()

    # Log model training run into DB
    db = next(get_db())
    try:
        if metadata:
            record_training_log(db, metadata)
        seed_initial_demo_records(db)
    finally:
        db.close()

    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="FinRisk AI - Credit Scoring & Loan Risk Prediction API",
    description="Full-stack Machine Learning Microservice with Database Persistence, Retrieval, and Analytics.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    """Serves the interactive web user interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "FinRisk AI API is running. Visit /docs for the interactive Swagger UI."}


@app.get("/api/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Health check validating API, database connectivity, and ML pipeline."""
    try:
        db_count = db.query(PredictionRecord).count()
        meta = get_metadata()
        return {
            "status": "healthy",
            "database": "connected",
            "stored_records": db_count,
            "ml_model": {
                "status": "loaded",
                "version": meta.get("version", "v1.0"),
                "algorithm": meta.get("algorithm", "RandomForestClassifier"),
                "accuracy": meta.get("accuracy", 0.0),
                "roc_auc": meta.get("roc_auc", 0.0)
            }
        }
    except Exception as exc:
        logger.error("Health check failure: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(exc)}
        )


@app.post("/api/predict", response_model=PredictionRecordResponse, status_code=status.HTTP_201_CREATED, tags=["ML Inference & Storage"])
def predict_and_store(applicant: ApplicantInput, db: Session = Depends(get_db)):
    """
    Execute ML prediction pipeline on applicant data, evaluate risk,
    and persist the complete decision record to the database for auditing and retrieval.
    """
    try:
        # Run ML model inference & explainability
        ml_result = predict_applicant(applicant.model_dump())

        # Persist to database
        db_record = create_prediction(db, applicant, ml_result)

        return serialize_record(db_record)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@app.get("/api/predictions", response_model=PaginatedPredictions, tags=["Database Retrieval"])
def list_stored_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by applicant name, purpose, or reference UUID"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier (Low Risk, Moderate Risk, High Risk)"),
    prediction: Optional[str] = Query(None, description="Filter by decision (APPROVED, REJECTED)"),
    db: Session = Depends(get_db)
):
    """
    Retrieve stored predictions from the database with pagination, search, and filtering.
    """
    skip = (page - 1) * size
    records, total = get_predictions(
        db,
        skip=skip,
        limit=size,
        search=search,
        risk_tier=risk_tier,
        prediction=prediction
    )

    total_pages = (total + size - 1) // size if total > 0 else 1
    serialized_items = [serialize_record(r) for r in records]

    return {
        "items": serialized_items,
        "total": total,
        "page": page,
        "size": size,
        "total_pages": total_pages
    }


@app.get("/api/predictions/{record_id}", response_model=PredictionRecordResponse, tags=["Database Retrieval"])
def get_stored_prediction_by_id(record_id: int, db: Session = Depends(get_db)):
    """Retrieve single stored prediction by database ID."""
    record = get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {record_id} not found."
        )
    return serialize_record(record)


@app.patch("/api/predictions/{record_id}", response_model=PredictionRecordResponse, tags=["Database Management"])
def update_prediction_record(
    record_id: int,
    update_data: RecordUpdate,
    db: Session = Depends(get_db)
):
    """Update review status (e.g. manual underwriting override) or notes on a stored prediction."""
    record = get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {record_id} not found."
        )

    updated = update_prediction(db, record, update_data)
    return serialize_record(updated)


@app.delete("/api/predictions/{record_id}", status_code=status.HTTP_200_OK, tags=["Database Management"])
def delete_prediction_record(record_id: int, db: Session = Depends(get_db)):
    """Delete a prediction record from the database."""
    record = get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {record_id} not found."
        )

    delete_prediction(db, record)
    return {"status": "success", "message": f"Record {record_id} deleted successfully."}


@app.get("/api/stats", response_model=AnalyticsSummary, tags=["Analytics & Reporting"])
def get_portfolio_stats(db: Session = Depends(get_db)):
    """Retrieve aggregate database analytics: approval rate, average risk, loan volume, and distributions."""
    return get_analytics(db)


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["ML Model Operations"])
def get_model_information(db: Session = Depends(get_db)):
    """Retrieve current ML model algorithm, performance metrics, and feature importances."""
    latest_log = get_latest_training_log(db)
    meta = get_metadata()

    if latest_log:
        try:
            feat_imp_raw = getattr(latest_log, "feature_importances", None)
            feat_imp = json.loads(str(feat_imp_raw)) if feat_imp_raw else meta.get("feature_importances", {})
        except Exception:
            feat_imp = meta.get("feature_importances", {})

        return {
            "version": latest_log.version,
            "algorithm": latest_log.algorithm,
            "accuracy": latest_log.accuracy,
            "roc_auc": latest_log.roc_auc,
            "f1_score": latest_log.f1_score,
            "precision": latest_log.precision,
            "recall": latest_log.recall,
            "training_samples": latest_log.training_samples,
            "feature_importances": feat_imp,
            "trained_at": latest_log.trained_at
        }

    return {
        "version": meta.get("version", "v1.0"),
        "algorithm": meta.get("algorithm", "RandomForestClassifier"),
        "accuracy": meta.get("accuracy", 0.0),
        "roc_auc": meta.get("roc_auc", 0.0),
        "f1_score": meta.get("f1_score", 0.0),
        "precision": meta.get("precision", 0.0),
        "recall": meta.get("recall", 0.0),
        "training_samples": meta.get("training_samples", 0),
        "feature_importances": meta.get("feature_importances", {}),
        "trained_at": meta.get("trained_at")
    }


@app.post("/api/model/retrain", response_model=RetrainResponse, tags=["ML Model Operations"])
def retrain_model_endpoint(
    samples: int = Query(5000, ge=500, le=20000, description="Number of synthetic/training samples to fit"),
    db: Session = Depends(get_db)
):
    """
    Retrain the machine learning model pipeline on demand, log metrics in database,
    and dynamically reload the model artifact in memory.
    """
    try:
        pipeline, metadata = train_and_export_model(n_samples=samples)
        reload_model_artifact()
        record_training_log(db, metadata)

        return {
            "status": "success",
            "message": "Model retrained and reloaded successfully into memory.",
            "new_version": metadata["version"],
            "metrics": {
                "accuracy": metadata["accuracy"],
                "roc_auc": metadata["roc_auc"],
                "f1_score": metadata["f1_score"],
                "precision": metadata["precision"],
                "recall": metadata["recall"]
            }
        }
    except Exception as exc:
        logger.exception("Retraining failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining error: {str(exc)}"
        )


@app.get("/api/model/sample-applicants", tags=["Demo Utilities"])
def get_sample_applicants():
    """Provides ready-to-test applicant profiles for quick 1-click form population in the UI."""
    return [
        {
            "label": "Prime Applicant (High Approval)",
            "description": "780+ Credit Score, High Income, Low DTI, Owns Home, Zero Defaults",
            "data": {
                "applicant_name": "Dr. Sophia Sterling",
                "age": 41,
                "annual_income": 145000.0,
                "credit_score": 795,
                "loan_amount": 25000.0,
                "loan_term_months": 36,
                "employment_years": 9.5,
                "debt_to_income_ratio": 14.5,
                "home_ownership": "OWN",
                "loan_purpose": "home_improvement",
                "has_prior_default": False
            }
        },
        {
            "label": "Moderate / Borderline Risk",
            "description": "650 Credit Score, Moderate Income, 32% DTI, Renter, Zero Defaults",
            "data": {
                "applicant_name": "Jordan Hayes",
                "age": 28,
                "annual_income": 62000.0,
                "credit_score": 648,
                "loan_amount": 16000.0,
                "loan_term_months": 48,
                "employment_years": 3.0,
                "debt_to_income_ratio": 31.8,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation",
                "has_prior_default": False
            }
        },
        {
            "label": "High Risk Applicant (Rejection Likely)",
            "description": "540 Credit Score, Prior Credit Default, High DTI (46%), Short Tenure",
            "data": {
                "applicant_name": "Damian Frost",
                "age": 35,
                "annual_income": 38000.0,
                "credit_score": 535,
                "loan_amount": 32000.0,
                "loan_term_months": 60,
                "employment_years": 0.8,
                "debt_to_income_ratio": 46.5,
                "home_ownership": "RENT",
                "loan_purpose": "personal",
                "has_prior_default": True
            }
        }
    ]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"🚀 Starting FinRisk AI on http://{host}:{port}/ ...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)

