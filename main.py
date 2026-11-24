"""Vercel entrypoint: Vercel's Python framework detection only recognizes app.py/index.py/
main.py/server.py/wsgi.py/asgi.py at the project root (or in src/ or app/), not api/index.py.
The real app lives in api/index.py (also used for local dev via `uvicorn api.index:app`)."""
from api.index import app
