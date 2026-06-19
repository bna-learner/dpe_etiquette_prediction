# ── Makefile — DPE Étiquette Prédiction (MLOps) ──────────────────────────────
# Projet de classification binaire : prédiction des passoires thermiques DPE
#
# Prérequis : uv installé (pip install uv)
# Usage     : make help
# -----------------------------------------------------------------------------

.PHONY: help install install-dev \
        train train-optuna train-models evaluate \
        api frontend mlflow-ui \
        docker-build docker-up docker-down docker-logs \
        docker-train docker-airflow-init docker-airflow \
        deploy deploy-build deploy-push deploy-pull deploy-up \
        test test-cov lint types \
        clean clean-models all

CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m
BOLD  := \033[1m

PYTHON     := uv run python
PYTEST     := uv run pytest
PYTHONPATH := .

# ── Variables de déploiement (à adapter) ──────────────────────────────────────
VPS_USER  ?= ubuntu
VPS_HOST  ?= 88.96.51.180
VPS_DIR   ?= /home/ubuntu/dpe-etiquette-prediction
SSH_KEY   ?= $(HOME)/Téléchargements/ssh-key-2026-06-18.key
GHCR_IMAGE     ?= ghcr.io/$(shell git config user.name | tr '[:upper:]' '[:lower:]')/dpe-etiquette-prediction/dpe-api
GIT_SHA        := $(shell git rev-parse --short HEAD 2>/dev/null || echo "latest")

# ── Aide ──────────────────────────────────────────────────────────────────────

help: ## Affiche cette aide
	@echo ""
	@echo "$(BOLD)DPE Étiquette Prédiction — MLOps$(RESET)"
	@echo "─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-28s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Environnement ─────────────────────────────────────────────────────────────

install: ## Installe les dépendances de production
	uv sync --no-dev
	@echo "$(GREEN)Environnement de production prêt.$(RESET)"

install-dev: ## Installe toutes les dépendances (y compris dev)
	uv sync
	@echo "$(GREEN)Environnement de développement prêt.$(RESET)"

# ── Qualité ───────────────────────────────────────────────────────────────────

lint: ## Vérifie le style avec ruff
	uv run ruff check src/

lint-fix: ## Corrige automatiquement les erreurs ruff
	uv run ruff check src/ --fix

types: ## Vérifie les types avec mypy
	uv run mypy src/

test: ## Lance tous les tests unitaires
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v --tb=short

test-cov: ## Lance les tests avec rapport de couverture
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov

# ── Entraînement local ────────────────────────────────────────────────────────

train: ## Entraînement baseline avec MLflow (TP S5)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.train

train-optuna: ## Optimisation Optuna + Model Registry (TP S6)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.train_optuna

train-models: ## Comparaison modèles GridSearchCV + SHAP (TP S7)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.train_models

evaluate: ## Évaluation automatique + porte qualité (TP S11)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.evaluate

predict-client: ## Test de l'API via le client de prévisions (TP S15)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/predict_client.py --n 5

# ── Serveurs locaux ───────────────────────────────────────────────────────────

api: ## Démarre l'API FastAPI en local (TP S12)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn src.api:app --reload --port 8000

frontend: ## Démarre le frontend Streamlit (TP S14bis)
	PYTHONPATH=$(PYTHONPATH) uv run streamlit run frontend/app.py

mlflow-ui: ## Démarre le serveur MLflow local
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--host 127.0.0.1 --port 5000

# ── Docker local ──────────────────────────────────────────────────────────────

docker-build: ## Build toutes les images Docker
	docker compose build
	docker compose --profile train build

docker-train: ## Entraîne le modèle dans Docker (one-shot)
	docker compose --profile train build train
	docker compose --profile train run --rm train

docker-airflow-init: ## Initialise la base Airflow (à lancer une seule fois)
	docker compose run --rm airflow-init

docker-airflow: ## Démarre Airflow (webserver + scheduler)
	docker compose up -d airflow-webserver airflow-scheduler
	@echo "$(GREEN)Airflow démarré → http://localhost:8080 (admin/admin)$(RESET)"

docker-up: ## Démarre la stack complète (MLflow + API + Frontend)
	docker compose up -d mlflow
	@echo "Attente MLflow..."
	@sleep 5
	docker compose --profile train run --rm train
	docker compose up -d api frontend
	@echo ""
	@echo "$(BOLD)Stack démarrée :$(RESET)"
	@echo "  $(GREEN)MLflow$(RESET)   → http://localhost:5000"
	@echo "  $(GREEN)API$(RESET)      → http://localhost:8000/docs"
	@echo "  $(GREEN)Frontend$(RESET) → http://localhost:8501"

docker-up-full: ## Démarre la stack complète avec Airflow
	$(MAKE) docker-up
	$(MAKE) docker-airflow-init
	$(MAKE) docker-airflow
	@echo "  $(GREEN)Airflow$(RESET)  → http://localhost:8080 (admin/admin)"

docker-down: ## Arrête toute la stack Docker
	docker compose down

docker-down-volumes: ## Arrête la stack et supprime les volumes
	docker compose down -v

docker-logs: ## Affiche les logs en temps réel
	docker compose logs -f

docker-logs-api: ## Logs de l'API uniquement
	docker compose logs -f api

docker-logs-airflow: ## Logs Airflow uniquement
	docker compose logs -f airflow-webserver airflow-scheduler

# ── Déploiement VPS Oracle ────────────────────────────────────────────────────

deploy-build: ## Build et tag l'image API pour GHCR
	docker build -f docker/Dockerfile.api \
		-t $(GHCR_IMAGE):$(GIT_SHA) \
		-t $(GHCR_IMAGE):latest \
		.
	@echo "$(GREEN)Image buildée : $(GHCR_IMAGE):$(GIT_SHA)$(RESET)"

deploy-push: deploy-build ## Push l'image API sur GHCR
	docker push $(GHCR_IMAGE):$(GIT_SHA)
	docker push $(GHCR_IMAGE):latest
	@echo "$(GREEN)Image poussée sur GHCR$(RESET)"

deploy-pull: ## Pull la dernière image sur le VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) \
		"docker pull $(GHCR_IMAGE):latest"

deploy-sync: ## Synchronise les fichiers de config sur le VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) "mkdir -p $(VPS_DIR)"
	rsync -avz --exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='mlflow.db' \
		--exclude='models/*.joblib' \
		--exclude='data/' \
		--exclude='htmlcov/' \
		docker-compose.yml \
		docker/ \
		dags/ \
		src/ \
		frontend/ \
		$(VPS_USER)@$(VPS_HOST):$(VPS_DIR)/
	@echo "$(GREEN)Fichiers synchronisés sur $(VPS_HOST)$(RESET)"

deploy-up: ## Démarre la stack sur le VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) \
		"cd $(VPS_DIR) && \
		 docker compose pull && \
		 docker compose up -d mlflow && \
		 sleep 5 && \
		 docker compose --profile train run --rm train && \
		 docker compose up -d api frontend && \
		 docker compose run --rm airflow-init 2>/dev/null || true && \
		 docker compose up -d airflow-webserver airflow-scheduler"
	@echo ""
	@echo "$(BOLD)Stack déployée sur $(VPS_HOST) :$(RESET)"
	@echo "  $(GREEN)MLflow$(RESET)   → http://$(VPS_HOST):5000"
	@echo "  $(GREEN)API$(RESET)      → http://$(VPS_HOST):8000/docs"
	@echo "  $(GREEN)Frontend$(RESET) → http://$(VPS_HOST):8501"
	@echo "  $(GREEN)Airflow$(RESET)  → http://$(VPS_HOST):8080"

deploy-down: ## Arrête la stack sur le VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) \
		"cd $(VPS_DIR) && docker compose down"

deploy-logs: ## Affiche les logs de la stack sur le VPS
	ssh -i $(SSH_KEY) $(VPS_USER)@$(VPS_HOST) \
		"cd $(VPS_DIR) && docker compose logs -f"

# ── Workflow complet local → VPS ──────────────────────────────────────────────

all: ## 🚀 Lance tout : qualité → build → push → déploiement VPS
	@echo "$(BOLD)── Étape 1/5 : Qualité du code ──$(RESET)"
	$(MAKE) lint
	$(MAKE) types
	$(MAKE) test
	@echo "$(BOLD)── Étape 2/5 : Entraînement local ──$(RESET)"
	$(MAKE) train
	$(MAKE) train-models
	$(MAKE) evaluate
	@echo "$(BOLD)── Étape 3/5 : Build & push Docker ──$(RESET)"
	$(MAKE) deploy-push
	@echo "$(BOLD)── Étape 4/5 : Synchronisation VPS ──$(RESET)"
	$(MAKE) deploy-sync
	@echo "$(BOLD)── Étape 5/5 : Déploiement VPS ──$(RESET)"
	$(MAKE) deploy-up
	@echo ""
	@echo "$(BOLD)$(GREEN)✓ Déploiement complet terminé !$(RESET)"

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean: ## Supprime les fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Nettoyage terminé.$(RESET)"

clean-models: ## Supprime les modèles locaux (MLflow artifacts conservés)
	rm -f models/*.joblib models/*.pkl 2>/dev/null || true
	@echo "$(GREEN)Modèles locaux supprimés.$(RESET)"

clean-all: clean clean-models ## Nettoyage complet (fichiers temp + modèles)
	rm -f mlflow.db 2>/dev/null || true
	@echo "$(GREEN)Nettoyage complet terminé.$(RESET)"