# Saarthi
A chatbot that helps citizens access government services (like applying for certificates or reporting issues) in local languages. Makes governance more transparent and accessible.

## Backend

The FastAPI backend is organized as a Python package under `backend/app`. Install
the backend's runtime dependencies in your Python environment, then start the
development server from the `backend` directory:

```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

For production, omit `--reload`. The API documentation is available at
`http://localhost:8000/api/docs`.

Backend modules are grouped by responsibility:

- `app/main.py` — FastAPI application and API endpoints
- `app/integrations/` — Dialogflow, translation, Vertex AI, and voice clients
- `app/nlp/processor.py` — multilingual NLP processing
- `app/database/repository.py` — SQLite-backed service data repository
