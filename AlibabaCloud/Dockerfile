# Stage 1: Build stage
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS builder

# Install Python 3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python runtime dependencies
RUN apt-get update && apt-get install -y python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python installation from the build stage
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8

# Set working directory
WORKDIR /app

# Copy application files
COPY serverless_function.py vgg19.pth payload.json .

# Expose the port used by the application
EXPOSE 9000

# Start the application
CMD ["python3", "/app/serverless_function.py"]
