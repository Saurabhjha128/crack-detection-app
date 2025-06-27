FROM python:3.10-slim

WORKDIR /app

# Install only the absolutely necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project code
COPY . /app

# Install pip first
RUN pip install --upgrade pip

# Install CPU-only torch first to avoid pulling CUDA wheels
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install the rest
RUN pip install --no-cache-dir \
    flask \
    flask_sqlalchemy \
    numpy \
    pillow \
    gunicorn

# **IMPORTANT: skip opencv-python here** (see note below)

# Expose Railway port
ENV PORT=8080

# Start app
CMD ["gunicorn", "webapp.backend.app:app", "--bind", "0.0.0.0:$PORT"]


RUN pip install --no-cache-dir opencv-python