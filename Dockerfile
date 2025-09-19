# Use the official slim Python image (Debian-based)
FROM python:3.11-slim

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose Flask port
EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=80

# Run Flask app in production with Gunicorn
CMD ["python", "src/app.py"]