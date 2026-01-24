.PHONY: help py-install web-install web-build dev-backend dev-frontend run

help:
	@echo "Targets:"
	@echo "  py-install     Install python package in editable mode"
	@echo "  web-install    Install frontend deps"
	@echo "  web-build      Build frontend into frontend/dist (served by Python)"
	@echo "  run            One-command run: build web + start begira"
	@echo "  dev-backend    Run backend only (reload)"
	@echo "  dev-frontend   Run frontend dev server"

py-install:
	python -m pip install -e .

web-install:
	cd frontend && npm install

web-build:
	cd frontend && npm run build

run: web-build
	python -c "import begira; begira.run()"

dev-backend:
	uvicorn begira.server:app --reload --host 127.0.0.1 --port 8000

dev-frontend:
	cd frontend && npm run dev

