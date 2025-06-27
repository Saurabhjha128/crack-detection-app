FROM python:3.10-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project code
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only torch and torchvision
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
RUN pip install --no-cache-dir \
    flask \
    flask_sqlalchemy \
    numpy \
    pillow \
    gunicorn \
    opencv-python

# Expose port
ENV PORT=8080

# Start app (use shell form to expand $PORT)
CMD gunicorn webapp.backend.app:app --bind 0.0.0.0:$PORT
