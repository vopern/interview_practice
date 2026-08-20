.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@/ {printf "\n%s\n", substr($$0, 5); next} \
		/^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Running locally

install: ## Install dependencies (incl. dev) into the local environment
	pip install -r requirements.txt pytest

run: ## Run the app locally
	streamlit run frontend/app.py

##@ Code quality

test: ## Run the test suite
	pytest tests/

format: ## Format code with black
	black src/ frontend/ tests/

lint: ## Lint with flake8
	flake8 src/ frontend/ tests/ --max-line-length=100

##@ Docker

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the app in Docker (http://localhost:8501)
	docker compose up -d

docker-down: ## Stop the Docker container
	docker compose down

docker-logs: ## Tail container logs
	docker logs --follow --tail 100 interview-trainer

.PHONY: help install run test format lint docker-build docker-up docker-down docker-logs