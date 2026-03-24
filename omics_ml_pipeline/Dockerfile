FROM python:3.11-slim

WORKDIR /omics_ml_pipeline

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ ./app/

# Default entry point
CMD ["python", "-m", "app.main"]
