.DEFAULT_GOAL := help
.PHONY: test dev-deps deps style lint mypy install help setup clean docs train validate demo

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Project structure paths
SRC_DIRS := src/trainer examples scripts tests
TARGET_DIRS := src examples tests scripts

# =============================================================================
# Development Setup
# =============================================================================

setup:  ## Set up development environment
	@echo "🚀 Setting up development environment..."
	python setup_dev.py

setup-quick:  ## Quick setup (skip tests and pre-commit)
	@echo "⚡ Quick development setup..."
	python setup_dev.py --quick

venv:  ## Create virtual environment
	@echo "🐍 Creating virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created. Activate with: source venv/bin/activate"

install:  ## Install package in development mode
	@echo "📦 Installing package in development mode..."
	pip install -e .

install-dev:  ## Install with development dependencies
	@echo "🛠️  Installing with development dependencies..."
	pip install -e .[dev,test]

# =============================================================================
# Code Quality
# =============================================================================

test:  ## Run tests
	@echo "🧪 Running tests..."
	uv run coverage run -m pytest -x ${SRC_DIRS}

test-all:  ## Run all tests without stopping on errors
	@echo "🧪 Running all tests..."
	uv run coverage run -m pytest ${SRC_DIRS}

test-unit:  ## Run unit tests only
	@echo "🔬 Running unit tests..."
	uv run pytest tests/unit/ -v

test-integration:  ## Run integration tests only
	@echo "🔗 Running integration tests..."
	uv run pytest tests/integration/ -v

test-failed:  ## Re-run only failed tests
	@echo "🔄 Re-running failed tests..."
	uv run coverage run -m pytest --ff ${SRC_DIRS}

coverage:  ## Generate test coverage report
	@echo "📊 Generating coverage report..."
	uv run coverage report
	uv run coverage html

style:  ## Format code style
	@echo "✨ Formatting code..."
	uv run --only-dev ruff format ${TARGET_DIRS}

lint:  ## Run linter
	@echo "🔍 Running linter..."
	uv run --only-dev ruff check ${TARGET_DIRS}

lint-fix:  ## Run linter with auto-fix
	@echo "🔧 Running linter with auto-fix..."
	uv run --only-dev ruff check --fix ${TARGET_DIRS}

mypy:  ## Run type checking
	@echo "🏷️  Running type checking..."
	uv run --group mypy mypy ${TARGET_DIRS}

pre-commit:  ## Run pre-commit on all files
	@echo "🪝 Running pre-commit hooks..."
	uv run pre-commit run --all-files

check:  ## Run all code quality checks
	@echo "✅ Running all quality checks..."
	$(MAKE) lint
	$(MAKE) mypy
	$(MAKE) test

# =============================================================================
# XTTS Training and Validation
# =============================================================================

train-basic:  ## Train basic XTTS model
	@echo "🎤 Starting basic XTTS training..."
	python examples/xtts/train_xtts_enhanced.py \
		--config configs/xtts/enhanced_xtts_config.json \
		--output_path ./output/basic_model

train-enhanced:  ## Train enhanced XTTS model
	@echo "🚀 Starting enhanced XTTS training..."
	python scripts/training/launch_full_enhanced_training.py \
		--config configs/xtts/full_enhanced_config.json

train-phase2:  ## Train with Phase 2 enhancements
	@echo "🔬 Starting Phase 2 enhanced training..."
	python examples/xtts/train_xtts_enhanced.py \
		--config configs/xtts/phase2_xtts_config.json \
		--use_phase2_enhancements

validate:  ## Validate trained model
	@echo "✅ Validating model..."
	python scripts/validation/validate_enhanced_xtts.py

validate-quick:  ## Quick model validation
	@echo "⚡ Quick validation..."
	python scripts/validation/quick_validate.py

demo:  ## Run interactive demo
	@echo "🎭 Starting demo..."
	python examples/demos/demo_enhanced_xtts.py

# =============================================================================
# Documentation
# =============================================================================

docs:  ## Open documentation
	@echo "📚 Opening documentation..."
	@echo "Main README: docs/README.md"
	@echo "Project Structure: PROJECT_STRUCTURE.md"
	@echo "Training Guide: docs/guides/FULL_ENHANCED_TRAINING_GUIDE.md"

docs-serve:  ## Serve documentation (if using mkdocs or similar)
	@echo "🌐 Documentation server not configured yet"
	@echo "📖 View documentation files in docs/ directory"

# =============================================================================
# Utilities
# =============================================================================

clean:  ## Clean up temporary files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.log" -delete

clean-models:  ## Clean up model outputs
	@echo "🗑️  Cleaning model outputs..."
	rm -rf output/*
	rm -rf logs/*.log

structure:  ## Show project structure
	@echo "📁 Project Structure:"
	@tree -I '__pycache__|*.pyc|.git|venv|*.egg-info' -L 3

list-configs:  ## List available configurations
	@echo "⚙️  Available configurations:"
	@ls -la configs/xtts/

list-examples:  ## List available examples
	@echo "📖 Available examples:"
	@find examples/ -name "*.py" -type f

env-info:  ## Show environment information
	@echo "🔍 Environment Information:"
	python bin/collect_env_info.py

# =============================================================================
# Quick Start Targets
# =============================================================================

quickstart:  ## Complete quickstart setup
	@echo "🚀 Running complete quickstart..."
	$(MAKE) setup-quick
	$(MAKE) install-dev
	@echo "✅ Quickstart complete! Run 'make demo' to test."

first-train:  ## First time training setup
	@echo "🎓 Setting up first training..."
	$(MAKE) setup
	$(MAKE) train-basic
	$(MAKE) validate-quick

# =============================================================================
# Advanced Features
# =============================================================================

benchmark:  ## Run performance benchmarks
	@echo "⏱️  Running benchmarks..."
	python scripts/utilities/benchmark_xtts.py

compare:  ## Compare model architectures
	@echo "🔬 Comparing architectures..."
	python scripts/utilities/compare_architectures.py

migrate:  ## Migrate to Phase 2
	@echo "🚚 Migrating to Phase 2..."
	python scripts/utilities/migrate_to_phase2.py

debug:  ## Run debugging tools
	@echo "🐛 Running debug tools..."
	python scripts/utilities/debug_tensor_mismatch.py
