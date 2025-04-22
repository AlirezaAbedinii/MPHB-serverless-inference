import os

# Template for worker.py
WORKER_TEMPLATE = """from flask import Flask, request, jsonify
import os
from utils import *

app = Flask(__name__)

# Get model name and shape from environment variables
WORKER_ID = os.getenv("WORKER_ID", "{worker_id}")  # Unique ID for each worker
INPUT_SHAPE = {input_shape}  # Shape of the model input
INPUT_NAME = "{input_name}"  # Model input name

# Load model
model = get_model(WORKER_ID, INPUT_SHAPE, INPUT_NAME)

@app.route('/invoke', methods=['POST'])
def infer():
    \"\"\"Receive and process inference requests\"\"\"
    try:
        data = request.get_json()

        # Get compressed input data
        compressed_input = data.get("compressed_data", None)
        if compressed_input is None:
            return jsonify({{"error": "Missing compressed data"}}), 400

        # Decompress input data
        decompressed_event = decompress_data(compressed_input)
        
        # Execute model inference
        response = execute_model(decompressed_event, model)

        # Compress response before sending it back
        compressed_response = compress_data(response)

        return jsonify({{"compressed_data": compressed_response}})

    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
"""

# Template for Dockerfile
DOCKERFILE_TEMPLATE = """# Use NVIDIA CUDA with cuDNN
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && \\
    ln -s /usr/bin/python3 /usr/bin/python && \\
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set cuDNN library path
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Install Python dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY worker.py utils.py {partition_file} /app/

# Expose port
EXPOSE 9000

# Run Flask app
CMD ["python3", "/app/worker.py"]
"""

# List of workers with their configurations
WORKERS = [
    # {"worker_id": "from1to2Worker1", "input_shape": "[1, 64, 224, 113]", "input_name": "vgg0_relu0_fwd"},
    
    # {"worker_id": "from11to11Worker1", "input_shape": "[1, 256, 15, 15]", "input_name": "vgg0_pool2_fwd"},
    {"worker_id": "from11to11Worker2", "input_shape": "[1, 256, 15, 15]", "input_name": "vgg0_pool2_fwd"},
    {"worker_id": "from11to11Worker3", "input_shape": "[1, 256, 15, 15]", "input_name": "vgg0_pool2_fwd"},
    
    {"worker_id": "from15to15Worker1", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu11_fwd"},
    {"worker_id": "from15to15Worker2", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu11_fwd"},    
    {"worker_id": "from15to15Worker3", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu11_fwd"},
   
    {"worker_id": "from16to16Worker1", "input_shape": "[1, 512, 8, 8]", "input_name": "vgg0_pool3_fwd"},
    {"worker_id": "from16to16Worker2", "input_shape": "[1, 512, 8, 8]", "input_name": "vgg0_pool3_fwd"},
    {"worker_id": "from16to16Worker3", "input_shape": "[1, 512, 8, 8]", "input_name": "vgg0_pool3_fwd"},
    
    {"worker_id": "from20to20Worker1", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu15_fwd"},
    {"worker_id": "from20to20Worker2", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu15_fwd"},
    {"worker_id": "from20to20Worker3", "input_shape": "[1, 512, 14, 14]", "input_name": "vgg0_relu15_fwd"},
    
    {"worker_id": "from21to21Worker1", "input_shape": "[1, 25088]", "input_name": "flatten_72"}

]

# Create worker directories and files
for worker in WORKERS:
    worker_dir = worker["worker_id"]
    os.makedirs(worker_dir, exist_ok=True)
    print(worker["worker_id"])
    # Create worker.py
    worker_py = WORKER_TEMPLATE.format(
        worker_id=worker["worker_id"],
        input_shape=worker["input_shape"],
        input_name=worker["input_name"],
    )
    with open(os.path.join(worker_dir, "worker.py"), "w") as f:
        f.write(worker_py)

    # Copy utils.py (must be in the same directory as this script)
    if os.path.exists("utils.py"):
        os.system(f"cp utils.py {worker_dir}/utils.py")

    # Copy requirements.txt (must be in the same directory as this script)
    if os.path.exists("requirements.txt"):
        os.system(f"cp requirements.txt {worker_dir}/requirements.txt")

    # Create Dockerfile
    partition_file = f"{worker['worker_id']}.json"
    dockerfile = DOCKERFILE_TEMPLATE.format(partition_file=partition_file)
    with open(os.path.join(worker_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)

print("Worker folders and files generated! Now, manually copy partition.json into each worker folder.")
