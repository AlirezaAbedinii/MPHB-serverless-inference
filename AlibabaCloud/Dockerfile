# Stage 1: Build environment
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS builder
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8
WORKDIR /app
COPY serverless_function.py vgg19.pth .
EXPOSE 9000
CMD ["python3", "/app/serverless_function.py"]
