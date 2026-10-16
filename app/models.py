import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text
)
from app.database import Base, engine


def utc_now():
    """Return timezone-naive UTC datetime for consistent SQLite storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PredictionRecord(Base):
    """Stores every ML prediction request and decision for auditing and retrieval."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    reference_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))

    # Applicant profile inputs
    applicant_name = Column(String(100), nullable=False, index=True)
    age = Column(Integer, nullable=False)
    annual_income = Column(Float, nullable=False)
    credit_score = Column(Integer, nullable=False, index=True)
    loan_amount = Column(Float, nullable=False)
    loan_term_months = Column(Integer, nullable=False)
    employment_years = Column(Float, nullable=False)
    debt_to_income_ratio = Column(Float, nullable=False)
    home_ownership = Column(String(30), nullable=False)
    loan_purpose = Column(String(50), nullable=False)
    has_prior_default = Column(Boolean, nullable=False, default=False)

    # ML Model outputs & Risk assessment
    prediction = Column(String(20), nullable=False, index=True)  # "APPROVED" | "REJECTED"
    approval_probability = Column(Float, nullable=False)        # 0.0 to 1.0
    risk_score = Column(Float, nullable=False, index=True)      # 0.0 to 100.0 (higher = riskier)
    risk_tier = Column(String(30), nullable=False, index=True)  # Low Risk, Moderate Risk, High Risk
    suggested_interest_rate = Column(Float, nullable=False)     # Recommended APR %
    confidence_score = Column(Float, nullable=False)            # Model confidence %
    top_factors = Column(Text, nullable=False)                  # JSON string with positive/negative drivers

    # Workflow & Manual Override metadata
    review_status = Column(String(30), nullable=False, default="AUTO_DECISION", index=True)
    notes = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=False, default="v1.0")

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=utc_now, index=True)
    updated_at = Column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)


class ModelTrainingLog(Base):
    """Tracks training runs, metrics, and parameters of the underlying ML models."""
    __tablename__ = "model_training_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True, index=True)
    algorithm = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    training_samples = Column(Integer, nullable=False)
    feature_importances = Column(Text, nullable=False)  # JSON string
    trained_at = Column(DateTime, nullable=False, default=utc_now)


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ FinRisk AI: SQLAlchemy models and database tables verified successfully!")
