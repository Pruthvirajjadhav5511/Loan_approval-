import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = [
    "age",
    "annual_income",
    "credit_score",
    "loan_amount",
    "loan_term_months",
    "employment_years",
    "debt_to_income_ratio",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "has_prior_default",
]

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def generate_synthetic_credit_data(n_samples: int = 4000, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic consumer loan underwriting dataset with known risk patterns."""
    np.random.seed(random_seed)

    # 1. Age: 21 to 72
    age = np.random.randint(21, 72, size=n_samples)

    # 2. Annual Income: $20,000 to $220,000 (log-normal distribution)
    income_raw = np.random.lognormal(mean=11.1, sigma=0.55, size=n_samples)
    annual_income = np.clip(np.round(income_raw, -2), 20000, 250000)

    # 3. Credit Score: 350 to 850 (FICO)
    credit_raw = np.random.normal(loc=680, scale=80, size=n_samples)
    credit_score = np.clip(np.round(credit_raw), 350, 850).astype(int)

    # 4. Loan Amount: $1,500 to $65,000
    loan_raw = np.random.lognormal(mean=9.8, sigma=0.6, size=n_samples)
    loan_amount = np.clip(np.round(loan_raw, -2), 1500, 65000)

    # 5. Loan Term in months
    loan_term_months = np.random.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # 6. Employment Years
    employment_years = np.clip(np.round(np.random.exponential(scale=5.0, size=n_samples), 1), 0.0, 35.0)

    # 7. Debt to Income ratio (%): 5% to 55%
    dti_raw = np.random.normal(loc=26, scale=10, size=n_samples)
    debt_to_income_ratio = np.clip(np.round(dti_raw, 1), 5.0, 60.0)

    # 8. Home Ownership
    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
        size=n_samples,
        p=[0.42, 0.44, 0.12, 0.02]
    )

    # 9. Loan Purpose
    loan_purpose = np.random.choice(
        ["debt_consolidation", "home_improvement", "business", "education", "medical", "personal", "other"],
        size=n_samples,
        p=[0.40, 0.18, 0.12, 0.10, 0.08, 0.08, 0.04]
    )

    # 10. Prior Default history (higher chance if credit score < 600)
    default_prob = np.where(credit_score < 580, 0.45, np.where(credit_score < 660, 0.18, 0.03))
    has_prior_default = np.random.binomial(1, default_prob).astype(bool)

    # Calculate realistic Credit Approval Score log-odds
    # Strong positive weights: High credit score, High income, Stable employment, Own home
    # Strong negative weights: Prior default, High DTI, High Loan-to-Income ratio
    loan_to_income = loan_amount / np.maximum(annual_income, 1.0)

    score = (
        (credit_score - 670) * 0.032
        + (annual_income - 60000) * 0.000035
        - (debt_to_income_ratio - 25) * 0.09
        - (loan_to_income - 0.25) * 4.5
        + (employment_years - 4) * 0.08
        - (has_prior_default.astype(int) * 2.8)
        + np.where(home_ownership == "OWN", 0.5, np.where(home_ownership == "MORTGAGE", 0.2, -0.1))
        + np.random.normal(0, 0.4, size=n_samples)  # realistic real-world variance
    )

    # Sigmoid to approval probability
    prob = 1.0 / (1.0 + np.exp(-score))
    approved = (prob >= 0.50).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "employment_years": employment_years,
        "debt_to_income_ratio": debt_to_income_ratio,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "has_prior_default": has_prior_default,
        "approved": approved,
    })

    return df


def train_and_export_model(
    n_samples: int = 5000,
    version_tag: Optional[str] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train RandomForest underwriting model pipeline and serialize artifact with metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not version_tag:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_tag = f"v{timestamp_str}"

    logger.info("Generating synthetic credit dataset with %d samples...", n_samples)
    data = generate_synthetic_credit_data(n_samples=n_samples)

    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = data["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    # Classifier
    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    logger.info("Training credit risk Random Forest classifier...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    # Feature Importance aggregation
    rf_model: RandomForestClassifier = pipeline.named_steps["classifier"]
    fitted_preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    feature_names = []
    # Numerical names
    feature_names.extend(NUMERICAL_FEATURES)
    # Categorical encoded names
    cat_encoder: OneHotEncoder = fitted_preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(encoded_cat_names)

    importances = rf_model.feature_importances_

    # Aggregate importance by base feature name
    aggregated_importance: Dict[str, float] = {f: 0.0 for f in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES)}
    for name, imp in zip(feature_names, importances):
        for base in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES):
            if name == base or name.startswith(f"{base}_"):
                aggregated_importance[base] += float(imp)
                break

    # Normalize aggregated importances to sum to 100%
    total_imp = sum(aggregated_importance.values()) or 1.0
    importance_pct = {k: round((v / total_imp) * 100, 2) for k, v in aggregated_importance.items()}
    # Sort descending
    importance_pct = dict(sorted(importance_pct.items(), key=lambda item: item[1], reverse=True))

    metadata = {
        "version": version_tag,
        "algorithm": "RandomForestClassifier with StandardScaler & OneHotEncoder",
        "training_samples": len(data),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "feature_importances": importance_pct,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save to disk
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model %s trained successfully (ROC-AUC: %.4f, Accuracy: %.4f)", version_tag, roc_auc, acc)
    return pipeline, metadata


if __name__ == "__main__":
    train_and_export_model(version_tag="v1.0")
