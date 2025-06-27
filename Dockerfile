# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install only CPU versions of torch and torchvision first (saves space)
RUN pip install --no-cache-dir torch torchvision

# Install other dependencies
RUN pip install --no-cache-dir \
    flask \
    flask_sqlalchemy \
    numpy \
    pillow \
    opencv-python \
    gunicorn

# Expose the port (Railway uses PORT env var)
ENV PORT=8080

# Start the app
CMD ["gunicorn", "webapp.backend.app:app", "--bind", "0.0.0.0:$PORT"]
