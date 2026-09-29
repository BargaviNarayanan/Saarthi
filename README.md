# Saarthi
A chatbot that helps citizens access government services (like applying for certificates or reporting issues) in local languages. Makes governance more transparent and accessible.

## Backend

The FastAPI backend is organized as a Python package under `backend/app` and
uses local mock AI service clients under `src/ai_services`; no cloud account,
service account key, or cloud SDK is required. Install the runtime dependencies
in your Python environment, then start the development server from the repository
root:

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

For production, omit `--reload`. The API documentation is available at
`http://localhost:8000/api/docs`.

Backend modules are grouped by responsibility:

- `app/main.py` — FastAPI application and API endpoints
- `app/integrations/` — backend adapters for the local AI service mocks
- `app/nlp/processor.py` — multilingual NLP processing
- `app/database/repository.py` — SQLite-backed service data repository
