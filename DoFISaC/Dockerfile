FROM python:3.11-slim

# Install system dependencies required for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpython3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt ./
COPY index-and-cluster.py ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create stripped package list
RUN pip list --format=freeze | cut -d "=" -f 1 > /app/packages.txt

# Add entrypoint script
COPY run-multi.sh /app/run-multi.sh
RUN chmod +x /app/run-multi.sh

CMD ["/app/run-multi.sh"]