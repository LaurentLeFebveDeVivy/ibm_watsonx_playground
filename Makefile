# ──────────────────────────────────────────────
# Supports Linux (Ubuntu/Debian) and macOS
# ──────────────────────────────────────────────

SHELL := /bin/bash

# ── OS detection ──────────────────────────────
UNAME_S := $(shell uname -s)

# ── Configuration ─────────────────────────────
PYTHON_VERSION := 3.12.3
VENV_DIR       := watsonx-agent-skeletton/venv
REQ_FILE       := watsonx-agent-skeletton/requirements.txt
ENV_EXAMPLE    := watsonx-agent-skeletton/config/.env.example
ENV_FILE       := watsonx-agent-skeletton/config/.env

PYENV_ROOT := $(HOME)/.pyenv
PYENV_BIN  := $(PYENV_ROOT)/bin/pyenv
PYENV_PATH := $(PYENV_ROOT)/bin:$(PYENV_ROOT)/shims:$(PATH)

# Inline pyenv init — needed in every recipe that calls pyenv or python
PYENV_INIT := export PYENV_ROOT="$(PYENV_ROOT)" && \
	export PATH="$(PYENV_PATH)" && \
	eval "$$($(PYENV_BIN) init --path)" && \
	eval "$$($(PYENV_BIN) init -)"

# ── Default target ────────────────────────────
.DEFAULT_GOAL := help

.PHONY: help setup install-deps install-pyenv install-python venv install env run clean

help: ## Show available targets
	@echo ""
	@echo "Usage:  make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo ""

setup: install-deps install-pyenv install-python venv install env ## One-shot full project setup
	@echo ""
	@echo "============================================"
	@echo " Setup complete!"
	@echo ""
	@echo " 1. Fill in your credentials:"
	@echo "      $(ENV_FILE)"
	@echo ""
	@echo " 2. Run the agent:"
	@echo "      make run"
	@echo "============================================"

install-deps: ## Install OS-level build dependencies
ifeq ($(UNAME_S),Linux)
	sudo apt update && sudo apt install -y \
		build-essential curl git \
		libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
		wget llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
		libffi-dev liblzma-dev python3-openssl
else ifeq ($(UNAME_S),Darwin)
	brew install openssl readline sqlite3 xz zlib tcl-tk
else
	$(error Unsupported OS: $(UNAME_S). Use setup.ps1 on Windows.)
endif

install-pyenv: ## Install pyenv (skipped if already present)
	@if command -v pyenv >/dev/null 2>&1 || [ -x "$(PYENV_BIN)" ]; then \
		echo "pyenv is already installed."; \
	else \
		echo "Installing pyenv..."; \
		curl -fsSL https://pyenv.run | bash; \
		echo ""; \
		echo "Adding pyenv to shell profile..."; \
		if [ -n "$$ZSH_VERSION" ] || [ -f "$$HOME/.zshrc" ] && [ "$$SHELL" = *zsh* ]; then \
			PROFILE="$$HOME/.zshrc"; \
		else \
			PROFILE="$$HOME/.bashrc"; \
		fi; \
		echo '' >> "$$PROFILE"; \
		echo '# pyenv' >> "$$PROFILE"; \
		echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> "$$PROFILE"; \
		echo 'export PATH="$$PYENV_ROOT/bin:$$PATH"' >> "$$PROFILE"; \
		echo 'eval "$$(pyenv init --path)"' >> "$$PROFILE"; \
		echo 'eval "$$(pyenv init -)"' >> "$$PROFILE"; \
		echo "pyenv added to $$PROFILE — restart your shell or run: source $$PROFILE"; \
	fi

install-python: ## Install Python 3.12.3 via pyenv
	$(PYENV_INIT) && pyenv install -s $(PYTHON_VERSION)
	$(PYENV_INIT) && pyenv local $(PYTHON_VERSION)

venv: ## Create virtual environment
	$(PYENV_INIT) && python -m venv $(VENV_DIR)

install: ## Install Python dependencies into venv
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)

env: ## Create .env from template (if it doesn't exist)
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "$(ENV_FILE) already exists — skipping."; \
	else \
		cp $(ENV_EXAMPLE) $(ENV_FILE); \
		echo "Created $(ENV_FILE) from template."; \
		echo "Please edit it and fill in your IBM watsonx credentials."; \
	fi

run: ## Run the watsonx agent
	$(VENV_DIR)/bin/python watsonx-agent-skeletton/main.py

clean: ## Remove virtual environment
	rm -rf $(VENV_DIR)
	@echo "Removed $(VENV_DIR)"
