import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ApplicantInput(BaseModel):
    """Input payload for making a credit risk & loan approval prediction."""
    applicant_name: str = Field(..., min_length=2, max_length=100, description="Full legal name of the applicant", examples=["Sarah Connor"])
    age: int = Field(..., ge=18, le=100, description="Age in years", examples=[34])
    annual_income: float = Field(..., gt=1000, description="Gross annual income in USD", examples=[85000.0])
    credit_score: int = Field(..., ge=300, le=850, description="FICO Credit Score (300 to 850)", examples=[740])
    loan_amount: float = Field(..., gt=500, description="Requested loan amount in USD", examples=[25000.0])
    loan_term_months: int = Field(..., ge=6, le=120, description="Loan term duration in months", examples=[36])
    employment_years: float = Field(..., ge=0.0, le=60.0, description="Years in current employment", examples=[5.5])
    debt_to_income_ratio: float = Field(..., ge=0.0, le=100.0, description="Debt-to-income ratio percentage", examples=[18.5])
    home_ownership: Literal["OWN", "MORTGAGE", "RENT", "OTHER"] = Field(..., description="Home ownership status", examples=["MORTGAGE"])
    loan_purpose: Literal[
        "debt_consolidation",
        "home_improvement",
        "business",
        "education",
        "medical",
        "personal",
        "other"
    ] = Field(..., description="Stated purpose of the loan", examples=["debt_consolidation"])
    has_prior_default: bool = Field(False, description="Has the applicant defaulted on any credit line in past 5 years?", examples=[False])


class FactorImpact(BaseModel):
    feature: str
    impact: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    description: str


class PredictionResult(BaseModel):
    prediction: Literal["APPROVED", "REJECTED"]
    approval_probability: float
    risk_score: float
    risk_tier: Literal["Low Risk", "Moderate Risk", "High Risk"]
    suggested_interest_rate: float
    confidence_score: float
    top_factors: List[FactorImpact]


class PredictionRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reference_id: str
    applicant_name: str
    age: int
    annual_income: float
    credit_score: int
    loan_amount: float
    loan_term_months: int
    employment_years: float
    debt_to_income_ratio: float
    home_ownership: str
    loan_purpose: str
    has_prior_default: bool
    prediction: str
    approval_probability: float
    risk_score: float
    risk_tier: str
    suggested_interest_rate: float
    confidence_score: float
    top_factors: List[Dict[str, Any]]
    review_status: str
    notes: Optional[str] = None
    model_version: str
    created_at: datetime
    updated_at: datetime


class RecordUpdate(BaseModel):
    review_status: Optional[Literal["AUTO_DECISION", "APPROVED_OVERRIDE", "REJECTED_OVERRIDE", "FLAGGED", "ARCHIVED"]] = None
    notes: Optional[str] = None


class PaginatedPredictions(BaseModel):
    items: List[PredictionRecordResponse]
    total: int
    page: int
    size: int
    total_pages: int


class AnalyticsSummary(BaseModel):
    total_applications: int
    approved_count: int
    rejected_count: int
    approval_rate: float
    avg_risk_score: float
    avg_loan_amount: float
    avg_credit_score: float
    total_loan_volume: float
    risk_tier_breakdown: Dict[str, int]
    purpose_breakdown: Dict[str, int]
    recent_trend: List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: str
    algorithm: str
    accuracy: float
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    training_samples: int
    feature_importances: Dict[str, float]
    trained_at: datetime


class RetrainResponse(BaseModel):
    status: str
    message: str
    new_version: str
    metrics: Dict[str, float]
