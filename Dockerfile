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
COPY face-indexer.py ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run face indexer script with a default path inside the container
CMD ["python", "face-indexer.py", "/data"]