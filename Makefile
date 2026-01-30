.PHONY: test benchmark build publish release clean

# Run all tests
test:
	poetry run pytest tests/

# Run performance benchmarks
benchmark:
	poetry run python benchmarks/benchmark_optimizations.py

# Run ODE solver benchmark
benchmark-ode:
	poetry run python benchmarks/benchmark_ode_solver.py

# Build package
build:
	poetry build

# Publish to PyPI (requires configured token)
publish:
	poetry publish

# Build and publish in one step
release: build publish

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies
install:
	poetry install

# Update dependencies
update:
	poetry update
