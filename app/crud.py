import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from app.models import PredictionRecord, ModelTrainingLog, utc_now
from app.schemas import ApplicantInput, RecordUpdate


def create_prediction(
    db: Session,
    applicant: ApplicantInput,
    ml_result: Dict[str, Any]
) -> PredictionRecord:
    """Store complete applicant data and ML prediction outputs into the database."""
    record = PredictionRecord(
        applicant_name=applicant.applicant_name,
        age=applicant.age,
        annual_income=applicant.annual_income,
        credit_score=applicant.credit_score,
        loan_amount=applicant.loan_amount,
        loan_term_months=applicant.loan_term_months,
        employment_years=applicant.employment_years,
        debt_to_income_ratio=applicant.debt_to_income_ratio,
        home_ownership=applicant.home_ownership,
        loan_purpose=applicant.loan_purpose,
        has_prior_default=applicant.has_prior_default,
        # ML outputs
        prediction=ml_result["prediction"],
        approval_probability=ml_result["approval_probability"],
        risk_score=ml_result["risk_score"],
        risk_tier=ml_result["risk_tier"],
        suggested_interest_rate=ml_result["suggested_interest_rate"],
        confidence_score=ml_result["confidence_score"],
        top_factors=json.dumps(ml_result["top_factors"]),
        model_version=ml_result["model_version"],
        review_status="AUTO_DECISION"
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction_by_id(db: Session, record_id: int) -> Optional[PredictionRecord]:
    """Retrieve single prediction record by primary key."""
    return db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()


def get_prediction_by_reference_id(db: Session, ref_id: str) -> Optional[PredictionRecord]:
    """Retrieve single prediction record by unique UUID reference."""
    return db.query(PredictionRecord).filter(PredictionRecord.reference_id == ref_id).first()


def get_predictions(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    search: Optional[str] = None,
    risk_tier: Optional[str] = None,
    prediction: Optional[str] = None
) -> Tuple[List[PredictionRecord], int]:
    """Retrieve paginated predictions with optional search and filtering."""
    query = db.query(PredictionRecord)

    if search:
        search_fmt = f"%{search.strip()}%"
        query = query.filter(
            or_(
                PredictionRecord.applicant_name.ilike(search_fmt),
                PredictionRecord.reference_id.ilike(search_fmt),
                PredictionRecord.loan_purpose.ilike(search_fmt)
            )
        )

    if risk_tier:
        query = query.filter(PredictionRecord.risk_tier == risk_tier)

    if prediction:
        query = query.filter(PredictionRecord.prediction == prediction)

    total = query.count()
    items = query.order_by(desc(PredictionRecord.created_at)).offset(skip).limit(limit).all()
    return items, total


def update_prediction(
    db: Session,
    record: PredictionRecord,
    update_data: RecordUpdate
) -> PredictionRecord:
    """Update prediction review status or underwriting notes."""
    if update_data.review_status is not None:
        setattr(record, "review_status", update_data.review_status)
    if update_data.notes is not None:
        setattr(record, "notes", update_data.notes)
    setattr(record, "updated_at", utc_now())
    db.commit()
    db.refresh(record)
    return record


def delete_prediction(db: Session, record: PredictionRecord) -> None:
    """Delete prediction record from the database."""
    db.delete(record)
    db.commit()


def get_analytics(db: Session) -> Dict[str, Any]:
    """Compute aggregate analytics across all stored loan predictions."""
    total = db.query(func.count(PredictionRecord.id)).scalar() or 0

    if total == 0:
        return {
            "total_applications": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "approval_rate": 0.0,
            "avg_risk_score": 0.0,
            "avg_loan_amount": 0.0,
            "avg_credit_score": 0.0,
            "total_loan_volume": 0.0,
            "risk_tier_breakdown": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
            "purpose_breakdown": {},
            "recent_trend": []
        }

    approved = db.query(func.count(PredictionRecord.id)).filter(
        PredictionRecord.prediction == "APPROVED"
    ).scalar() or 0
    rejected = total - approved

    avg_risk = db.query(func.avg(PredictionRecord.risk_score)).scalar() or 0.0
    avg_loan = db.query(func.avg(PredictionRecord.loan_amount)).scalar() or 0.0
    avg_credit = db.query(func.avg(PredictionRecord.credit_score)).scalar() or 0.0
    total_volume = db.query(func.sum(PredictionRecord.loan_amount)).scalar() or 0.0

    # Risk tier distribution
    tier_rows = db.query(
        PredictionRecord.risk_tier,
        func.count(PredictionRecord.id)
    ).group_by(PredictionRecord.risk_tier).all()
    tier_map = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}
    for t_name, count in tier_rows:
        tier_map[t_name] = count

    # Purpose breakdown
    purpose_rows = db.query(
        PredictionRecord.loan_purpose,
        func.count(PredictionRecord.id)
    ).group_by(PredictionRecord.loan_purpose).all()
    purpose_map = {p_name: count for p_name, count in purpose_rows}

    # Trend by recent records (last 10 entries grouped or listed)
    recent_records = db.query(
        PredictionRecord.created_at,
        PredictionRecord.prediction,
        PredictionRecord.risk_score,
        PredictionRecord.loan_amount
    ).order_by(desc(PredictionRecord.created_at)).limit(15).all()

    recent_trend = [
        {
            "timestamp": r[0].strftime("%H:%M:%S") if r[0] else "",
            "prediction": r[1],
            "risk_score": round(r[2], 1),
            "loan_amount": round(r[3], 2)
        }
        for r in reversed(recent_records)
    ]

    return {
        "total_applications": total,
        "approved_count": approved,
        "rejected_count": rejected,
        "approval_rate": round((approved / total) * 100, 1),
        "avg_risk_score": round(float(avg_risk), 1),
        "avg_loan_amount": round(float(avg_loan), 2),
        "avg_credit_score": round(float(avg_credit), 1),
        "total_loan_volume": round(float(total_volume), 2),
        "risk_tier_breakdown": tier_map,
        "purpose_breakdown": purpose_map,
        "recent_trend": recent_trend
    }


def record_training_log(db: Session, metadata: Dict[str, Any]) -> ModelTrainingLog:
    """Save model training metrics and features into the database log table."""
    # Check if this version already logged
    existing = db.query(ModelTrainingLog).filter(ModelTrainingLog.version == metadata["version"]).first()
    if existing:
        return existing

    log = ModelTrainingLog(
        version=metadata["version"],
        algorithm=metadata["algorithm"],
        accuracy=metadata["accuracy"],
        roc_auc=metadata["roc_auc"],
        f1_score=metadata["f1_score"],
        precision=metadata["precision"],
        recall=metadata["recall"],
        training_samples=metadata["training_samples"],
        feature_importances=json.dumps(metadata.get("feature_importances", {})),
        trained_at=datetime.fromisoformat(metadata["trained_at"]).replace(tzinfo=None)
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_latest_training_log(db: Session) -> Optional[ModelTrainingLog]:
    """Fetch the most recent model training audit entry."""
    return db.query(ModelTrainingLog).order_by(desc(ModelTrainingLog.trained_at)).first()
