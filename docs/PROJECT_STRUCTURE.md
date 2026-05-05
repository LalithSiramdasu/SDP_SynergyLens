# Project Structure

## Active application

These files and folders are the active app we should build on:

- `app.py`: Flask backend and API routes
- `templates/index.html`: main frontend UI
- `models/`: trained models and lookup data
- `static/css/app.css`: theme tokens and UI component styles
- `static/js/app.js`: frontend behavior adapted to the current API contracts
- `requirements.txt`: Python dependencies
- `README.md`: project overview
- `sample_batch.csv`: batch prediction sample input
- `verify_app.py`: Flask test-client checks for current routes, helper endpoints, and CSV batch output
- `.gitignore`: excludes local secrets, Gemini key files, logs, caches, and generated screenshots

## Archived files

These files were moved out of the active root so the project is easier to navigate:

- `archive/root_backups/`: older backup Python files
- `archive/duplicate_project/sdp_app/`: duplicate full copy of the project

## Recommended development direction

Use the root-level app as the single source of truth.

Recommended focus areas for next work:

1. Keep model artifacts, feature construction, and route payloads as the backend source of truth
2. Keep helper endpoints additive: `/api/health`, `/api/cell-lines`, `/api/system-summary`, and `/api/model-performance-summary`
3. Keep the static Project Assistant browser-only; prediction-specific chat should continue to use `/api/chat`
4. Break backend helpers into smaller modules when the feature set grows
