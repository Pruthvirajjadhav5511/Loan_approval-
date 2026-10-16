import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
from app.ml.trainer import (
    MODEL_PATH,
    METADATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    train_and_export_model,
)

logger = logging.getLogger(__name__)

_PIPELINE = None
_METADATA = None


def get_model():
    """Load and cache the trained ML pipeline in memory."""
    global _PIPELINE, _METADATA
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            logger.info("Trained model not found at %s. Triggering training...", MODEL_PATH)
            _PIPELINE, _METADATA = train_and_export_model(version_tag="v1.0")
        else:
            _PIPELINE = joblib.load(MODEL_PATH)
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r") as f:
                    _METADATA = json.load(f)
            else:
                _METADATA = {"version": "v1.0", "algorithm": "Random Forest"}
    return _PIPELINE


def get_metadata() -> Dict[str, Any]:
    """Retrieve current model metadata and performance metrics."""
    global _METADATA
    if _METADATA is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r") as f:
                _METADATA = json.load(f)
        else:
            get_model()  # triggers training and loads metadata
    return _METADATA or {}


def reload_model_artifact():
    """Force reload of the pipeline and metadata from disk after retraining."""
    global _PIPELINE, _METADATA
    if MODEL_PATH.exists():
        _PIPELINE = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            _METADATA = json.load(f)
    return _PIPELINE, _METADATA


def explain_prediction(applicant: Dict[str, Any], prob_approved: float) -> List[Dict[str, str]]:
    """Generate human-interpretable feature contribution drivers for the decision."""
    factors: List[Dict[str, str]] = []

    # 1. Credit Score impact
    cs = applicant["credit_score"]
    if cs >= 740:
        factors.append({
            "feature": "Credit Score",
            "impact": "POSITIVE",
            "description": f"Exceptional credit score ({cs}) significantly strengthens approval credibility."
        })
    elif cs < 620:
        factors.append({
            "feature": "Credit Score",
            "impact": "NEGATIVE",
            "description": f"Subprime credit score ({cs}) indicates heightened default probability."
        })
    else:
        factors.append({
            "feature": "Credit Score",
            "impact": "NEUTRAL",
            "description": f"Fair credit score ({cs}) meets basic requirements."
        })

    # 2. Prior Default impact
    if applicant["has_prior_default"]:
        factors.append({
            "feature": "Prior Default",
            "impact": "NEGATIVE",
            "description": "Recorded prior credit default in history triggers high risk penalty."
        })
    else:
        factors.append({
            "feature": "Prior Default",
            "impact": "POSITIVE",
            "description": "Clean historical credit record with zero recorded defaults."
        })

    # 3. Debt-to-Income (DTI)
    dti = applicant["debt_to_income_ratio"]
    if dti <= 22.0:
        factors.append({
            "feature": "Debt-to-Income Ratio",
            "impact": "POSITIVE",
            "description": f"Healthy debt-to-income ratio ({dti}%) demonstrates ample cash flow headroom."
        })
    elif dti >= 38.0:
        factors.append({
            "feature": "Debt-to-Income Ratio",
            "impact": "NEGATIVE",
            "description": f"High debt-to-income ratio ({dti}%) limits debt servicing capacity."
        })

    # 4. Loan-to-Income leverage
    income = applicant["annual_income"]
    loan_amt = applicant["loan_amount"]
    lti = loan_amt / max(income, 1.0)
    if lti > 0.40:
        factors.append({
            "feature": "Loan to Income",
            "impact": "NEGATIVE",
            "description": f"Requested loan equals {round(lti * 100, 1)}% of annual income, reflecting heavy leverage."
        })
    elif lti < 0.20:
        factors.append({
            "feature": "Loan to Income",
            "impact": "POSITIVE",
            "description": f"Requested loan is only {round(lti * 100, 1)}% of annual income (conservative leverage)."
        })

    # 5. Employment tenure
    emp = applicant["employment_years"]
    if emp >= 5.0:
        factors.append({
            "feature": "Employment Stability",
            "impact": "POSITIVE",
            "description": f"Solid job tenure of {emp} years indicates stable income reliability."
        })
    elif emp < 1.0:
        factors.append({
            "feature": "Employment Stability",
            "impact": "NEGATIVE",
            "description": f"Short tenure ({emp} yrs) implies recent career transition risk."
        })

    # 6. Home Ownership
    home = applicant["home_ownership"]
    if home == "OWN":
        factors.append({
            "feature": "Home Ownership",
            "impact": "POSITIVE",
            "description": "Applicant outright owns primary residence, providing substantial asset backing."
        })

    return factors


def predict_applicant(applicant_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute end-to-end ML scoring pipeline for an applicant."""
    pipeline = get_model()
    metadata = get_metadata()

    # Create single-row DataFrame
    row_data = {
        k: [applicant_dict[k]]
        for k in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    }
    df = pd.DataFrame(row_data)

    # Model inference
    proba = pipeline.predict_proba(df)[0]
    # Class 1 is APPROVED (positive class)
    prob_approved = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # Risk score: 0 (safest) to 100 (highest risk)
    risk_score = round((1.0 - prob_approved) * 100.0, 1)

    # Decision threshold
    prediction = "APPROVED" if prob_approved >= 0.50 else "REJECTED"

    # Risk Tier classification
    if risk_score < 30.0:
        risk_tier = "Low Risk"
    elif risk_score < 60.0:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "High Risk"

    # Suggested APR % based on risk tier and score
    # Base 5.2% + (risk_score / 100) * 16.5%
    base_apr = 5.20
    calc_apr = base_apr + (risk_score / 100.0) * 16.50
    suggested_apr = round(min(max(calc_apr, 4.99), 24.99), 2)

    # Model Confidence: distance from decision threshold
    confidence = round(max(prob_approved, 1.0 - prob_approved) * 100.0, 1)

    # Explainability factors
    factors = explain_prediction(applicant_dict, prob_approved)

    return {
        "prediction": prediction,
        "approval_probability": round(prob_approved, 4),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "suggested_interest_rate": suggested_apr,
        "confidence_score": confidence,
        "top_factors": factors,
        "model_version": metadata.get("version", "v1.0"),
    }
