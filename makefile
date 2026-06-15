# ── Makefile — DPE Étiquette Prédiction (MLOps) ──────────────────────────────
# Projet de classification binaire : prédiction des passoires thermiques DPE
#
# Prérequis : uv installé (pip install uv)
# Usage     : make help
# -----------------------------------------------------------------------------

.PHONY: help install train train-optuna train-models api frontend \
        docker-build docker-up docker-down test test-cov lint \
        export-data clean

CYAN  := \033[36m
RESET := \033[0m
BOLD  := \033[1m

PYTHON     := uv run python
PYTEST     := uv run pytest
PYTHONPATH := .

help: ## Affiche cette aide
	@echo ""
	@echo "$(BOLD)DPE Étiquette Prédiction — MLOps$(RESET)"
	@echo "─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Environnement ─────────────────────────────────────────────────────────────

install: ## Installe les dépendances via uv
	uv sync --all-extras
	@echo "Environnement prêt."

install-dev: ## Installe les dépendances + dev (pytest, httpx...)
	uv sync --all-extras
	@echo "Environnement dev prêt."

# ── Données ───────────────────────────────────────────────────────────────────

export-data: ## Exporte un échantillon Silver → data/dpe_silver_sample.csv
	$(PYTHON) scripts/export_silver.py
	@echo "Données exportées dans data/dpe_silver_sample.csv"

# ── Entraînement ──────────────────────────────────────────────────────────────

train: ## Entraînement baseline avec MLflow (TP S5)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train

train-optuna: ## Optimisation Optuna + Model Registry (TP S6)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train_optuna

train-models: ## Comparaison modèles GridSearchCV + SHAP (TP S7)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlproject.train_models

# ── API ───────────────────────────────────────────────────────────────────────

api: ## Démarre l'API FastAPI en local (TP S12)
	PYTHONPATH=$(PYTHONPATH) uv run uvicorn mlproject.api:app --reload --port 8000

# ── Frontend ──────────────────────────────────────────────────────────────────

frontend: ## Démarre le frontend Streamlit (TP S14bis)
	PYTHONPATH=$(PYTHONPATH) uv run streamlit run frontend/app.py

# ── MLflow UI ─────────────────────────────────────────────────────────────────

mlflow-ui: ## Démarre l'interface MLflow
	uv run mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build toutes les images Docker
	docker compose build

docker-up: ## Démarre la stack complète (MLflow + API + Frontend + Airflow)
	docker compose up -d
	@echo ""
	@echo "$(BOLD)Stack démarrée :$(RESET)"
	@echo "  MLflow    → http://localhost:5000"
	@echo "  API       → http://localhost:8000/docs"
	@echo "  Frontend  → http://localhost:8501"
	@echo "  Airflow   → http://localhost:8080  (admin/admin)"

docker-down: ## Arrête la stack Docker
	docker compose down

docker-logs: ## Affiche les logs en temps réel
	docker compose logs -f

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Lance tous les tests unitaires
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v --tb=short

test-cov: ## Lance les tests avec rapport de couverture
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		--cov=mlproject \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean: ## Supprime les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Nettoyage terminé."

clean-models: ## Supprime les modèles locaux (MLflow artifacts conservés)
	rm -rf models/*.pkl models/*.json 2>/dev/null || true
	@echo "Modèles locaux supprimés."