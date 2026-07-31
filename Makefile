.PHONY: install run test clean benchmarks

# Install dependencies using uv
install:
	uv venv
	uv pip install -r requirements.txt

# Run the FastAPI server
run:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	uv run pytest -v

# Clean up pycache and other temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Run OCR benchmarks
benchmarks:
	uv run python -m benchmarks.run
