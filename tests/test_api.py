import sys
from pathlib import Path

# Add project root directory to sys.path so 'app' can always be imported when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient fixture with proper lifespan context."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Verify health endpoint reports connected database and loaded model."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["ml_model"]["status"] == "loaded"


def test_predict_and_persist_low_risk(client):
    """Verify inference on prime profile results in APPROVAL and database persistence."""
    payload = {
        "applicant_name": "Test Prime Applicant",
        "age": 38,
        "annual_income": 120000.0,
        "credit_score": 790,
        "loan_amount": 20000.0,
        "loan_term_months": 36,
        "employment_years": 8.0,
        "debt_to_income_ratio": 15.0,
        "home_ownership": "OWN",
        "loan_purpose": "home_improvement",
        "has_prior_default": False
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["prediction"] == "APPROVED"
    assert data["risk_tier"] == "Low Risk"
    assert data["risk_score"] < 40.0
    assert data["id"] is not None
    assert data["reference_id"] is not None
    assert len(data["top_factors"]) > 0

    record_id = data["id"]

    # Verify retrieval by ID
    get_res = client.get(f"/api/predictions/{record_id}")
    assert get_res.status_code == 200
    retrieved = get_res.json()
    assert retrieved["applicant_name"] == "Test Prime Applicant"
    assert retrieved["prediction"] == "APPROVED"


def test_predict_and_persist_high_risk(client):
    """Verify inference on subprime profile results in REJECTION and database persistence."""
    payload = {
        "applicant_name": "Test Subprime Applicant",
        "age": 25,
        "annual_income": 28000.0,
        "credit_score": 510,
        "loan_amount": 30000.0,
        "loan_term_months": 60,
        "employment_years": 0.5,
        "debt_to_income_ratio": 50.0,
        "home_ownership": "RENT",
        "loan_purpose": "personal",
        "has_prior_default": True
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["prediction"] == "REJECTED"
    assert data["risk_tier"] == "High Risk"
    assert data["risk_score"] >= 60.0


def test_list_and_search_predictions(client):
    """Verify paginated listing and text search functionality."""
    # List all
    res = client.get("/api/predictions?page=1&size=5")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) <= 5

    # Search by name
    res_search = client.get("/api/predictions?search=Prime")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert search_data["total"] >= 1
    assert any("Prime" in item["applicant_name"] for item in search_data["items"])


def test_update_and_delete_record(client):
    """Verify record manual override and deletion."""
    # Create a dummy record
    payload = {
        "applicant_name": "To Be Modified",
        "age": 30,
        "annual_income": 60000.0,
        "credit_score": 670,
        "loan_amount": 10000.0,
        "loan_term_months": 24,
        "employment_years": 3.0,
        "debt_to_income_ratio": 25.0,
        "home_ownership": "RENT",
        "loan_purpose": "education",
        "has_prior_default": False
    }
    create_res = client.post("/api/predict", json=payload)
    rec_id = create_res.json()["id"]

    # Patch record
    patch_res = client.patch(f"/api/predictions/{rec_id}", json={
        "review_status": "APPROVED_OVERRIDE",
        "notes": "Verified employment via employer call."
    })
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["review_status"] == "APPROVED_OVERRIDE"
    assert updated["notes"] == "Verified employment via employer call."

    # Delete record
    del_res = client.delete(f"/api/predictions/{rec_id}")
    assert del_res.status_code == 200

    # Ensure 404 after deletion
    check_res = client.get(f"/api/predictions/{rec_id}")
    assert check_res.status_code == 404


def test_analytics_endpoint(client):
    """Verify statistical aggregation endpoint returns calculated metrics."""
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_applications"] > 0
    assert "approval_rate" in data
    assert "risk_tier_breakdown" in data
    assert "total_loan_volume" in data


def test_model_info_and_retrain(client):
    """Verify model metadata endpoint and on-demand retraining."""
    # Info
    info_res = client.get("/api/model/info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert "accuracy" in info
    assert "roc_auc" in info
    assert "feature_importances" in info
    assert "credit_score" in info["feature_importances"]

    # Retrain with small batch for speed
    retrain_res = client.post("/api/model/retrain?samples=800")
    assert retrain_res.status_code == 200
    retrain_data = retrain_res.json()
    assert retrain_data["status"] == "success"
    assert "new_version" in retrain_data
    assert "metrics" in retrain_data


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", "-W", "ignore::pytest.PytestAssertRewriteWarning", str(Path(__file__).resolve())]))

