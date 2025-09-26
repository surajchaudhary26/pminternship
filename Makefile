.PHONY: run-api stop-api

# API ko run karo (agar purana process hai toh kill karega)
run-api:
	@echo "🚀 Starting API on http://127.0.0.1:8001 ..."
	@-kill -9 $$(lsof -t -i:8001) 2>/dev/null || true
	uvicorn ml_engine.api.app:app --reload --port 8001

# API ko stop karo manually
stop-api:
	@echo "🛑 Stopping API on port 8001..."
	@-kill -9 $$(lsof -t -i:8001) 2>/dev/null || true
