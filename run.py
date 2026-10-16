import os
import sys
from pathlib import Path
import uvicorn

# Ensure the project directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")

    print("=" * 65)
    print("🚀 FinRisk AI - Intelligent Credit Scoring & ML Prediction API")
    print("=" * 65)
    print(f"📊 Web Application Dashboard:  http://{host}:{port}/")
    print(f"📑 Interactive Swagger Docs:    http://{host}:{port}/docs")
    print(f"🩺 API Healthcheck:            http://{host}:{port}/api/health")
    print(f"🗄️ Database File:              {PROJECT_ROOT / 'finrisk.db'}")
    print("=" * 65)

    uvicorn.run("app.main:app", host=host, port=port, reload=True)
