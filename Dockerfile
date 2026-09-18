# Use official slim Python 3.11 image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and dataset
COPY . .

# Expose API and Dashboard port
EXPOSE 8000
EXPOSE 8501

ENV PORT=8000
ENV HOST=0.0.0.0

# Start server
CMD ["python", "run.py"]
