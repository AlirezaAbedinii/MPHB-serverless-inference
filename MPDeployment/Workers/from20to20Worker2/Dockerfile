# Use NVIDIA CUDA with cuDNN
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set cuDNN library path
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Install Python dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY worker.py utils.py 20_20_2.json /app/

# Expose port
EXPOSE 9000

# Run Flask app
CMD ["python3", "/app/worker.py"]
