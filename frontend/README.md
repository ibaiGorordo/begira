# begira frontend

This is the web viewer for **begira**.

## Dev

- Start Python backend:
  - `uvicorn begira.runtime.app:app --reload --host 127.0.0.1 --port 8000`
- Start frontend dev server:
  - `npm install`
  - `npm run dev`

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

## Build

`npm run build`

Outputs to `dist/`. The Python server serves the built files when present.
