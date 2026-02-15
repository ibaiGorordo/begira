# Refactor Map

## Backend Moves
- `src/begira/elements.py` -> `src/begira/core/elements.py`
- `src/begira/timeline.py` -> `src/begira/core/timeline.py`
- `src/begira/animation.py` -> `src/begira/core/animation.py`
- `src/begira/conventions.py` -> `src/begira/core/conventions.py`
- `src/begira/viewer_settings.py` -> `src/begira/core/viewer_settings.py`
- `src/begira/registry.py` -> `src/begira/core/registry/service.py`
- `src/begira/elements_api.py` -> `src/begira/api/routes/elements.py`
- `src/begira/api_time.py` -> `src/begira/api/parsing/time.py`
- `src/begira/element_projection.py` -> `src/begira/api/serializers/elements.py`
- `src/begira/runner.py` -> `src/begira/runtime/server.py`
- `src/begira/server.py` -> `src/begira/runtime/app.py`
- `src/begira/web.py` -> `src/begira/runtime/web.py`
- `src/begira/client.py` -> `src/begira/sdk/client.py`
- `src/begira/handles.py` -> `src/begira/sdk/handles.py`
- `src/begira/ply.py` -> `src/begira/io/ply.py`
- `src/begira/image_logging.py` -> `src/begira/io/image.py`

Compatibility wrappers remain at all legacy module paths.

## Frontend Additions/Reorg
- `frontend/src/App.tsx` -> `frontend/src/app/AppShell.tsx` (with compatibility entrypoint in `App.tsx`)
- Added feature folders:
  - `frontend/src/features/hierarchy/`
  - `frontend/src/features/inspector/`
  - `frontend/src/features/timeline/`
  - `frontend/src/features/workspace/`
  - `frontend/src/features/view3d/`
- Added shared API split:
  - `frontend/src/shared/api/index.ts`
  - `frontend/src/shared/api/types.ts`
  - `frontend/src/shared/api/elements.ts`
  - `frontend/src/shared/api/timeline.ts`
  - `frontend/src/shared/api/animation.ts`
  - `frontend/src/shared/api/viewer.ts`
- `frontend/src/viewer/api.ts` now re-exports from `frontend/src/shared/api/`.

## Hygiene
- Removed accidental root-level Node package files:
  - `package.json`
  - `package-lock.json`
  - `node_modules/`
