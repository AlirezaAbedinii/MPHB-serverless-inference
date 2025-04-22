# Stage 1: Build stage
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS builder

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Stage 2: Runtime environment
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python runtime dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Gunicorn explicitly in the runtime stage
RUN pip3 install gunicorn

# Set working directory
WORKDIR /app

# Copy installed dependencies from the build stage
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8

# Copy application files
COPY master.py utils.py payload.json structure.json /app/

COPY partitions /app/partitions/

# Expose the port used by Flask
EXPOSE 9000

# Start the Flask app using Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:9000", "--timeout", "120", "master:app"]
