from flask import Flask, request, jsonify
import requests
import json
import time
import os
import base64
import gzip
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load model partition structure
with open("structure.json", "r") as f:
    stage_dict = json.load(f)

# Load sample input from payload.json
with open("payload.json", "r") as f:
    payload_template = json.load(f)

# Function Compute Base URL (replace with actual FC endpoint)
FC_BASE_URL = os.getenv("FC_BASE_URL", "https://<your-region>.fc.aliyuncs.com")
worker_urls = {}

# Get DEVICE type from environment variable (CPU or GPU)
DEVICE = os.getenv("DEVICE", "cpu").lower()  # Default to CPU

# Enable logging (for debugging)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def log(message):
    """Print log messages only if debugging is enabled."""
    if DEBUG:
        print(message)


### Compression Helper Functions ###
def compress_data(data):
    """Compress and encode data using Gzip and Base64"""
    json_data = json.dumps(data).encode("utf-8")
    compressed_data = gzip.compress(json_data)
    return base64.b64encode(compressed_data).decode("utf-8")


def decompress_data(compressed_string):
    """Decode and decompress Base64-encoded Gzip data"""
    compressed_bytes = base64.b64decode(compressed_string)
    json_data = gzip.decompress(compressed_bytes).decode("utf-8")
    return json.loads(json_data)


### Worker Invocation Function ###
def invoke_worker(func_dict):
    """Send inference request to worker function"""

    split_name = func_dict["model_name"].split("_")
    worker_url = f"{FC_BASE_URL}/invoke"

    compressed_payload = compress_data(func_dict)

    # Send request to the worker function
    try:
        response = requests.post(
            worker_url,
            json={"compressed_data": compressed_payload, "device": DEVICE},
        )
        json_result = response.json()

        # Decompress worker response
        decompressed_result = decompress_data(json_result["compressed_data"])
        return decompressed_result
    except Exception as e:
        log(f"Worker Invocation Failed: {str(e)}")
        return {"error": str(e)}


### Parallel Execution Function ###
def do_parallel(model_dict):
    """Executes inference in parallel across worker functions"""
    outputs = []
    futures = []
    latencies = []

    thread_num = len(model_dict)

    if thread_num > 0:
        pool = ThreadPoolExecutor(max_workers=thread_num)
        for _, func_dict in model_dict.items():
            fu = pool.submit(invoke_worker, func_dict)
            futures.append(fu)

    # Collect results
    for f in futures:
        fu_result = f.result()
        outputs.append(fu_result["output"])
        latencies.append(fu_result["latency"])

    return outputs, latencies


@app.route("/invoke", methods=["GET"])
def infer():
    """Main function that handles inference requests"""
    try:
        start_time = time.time()

        # Get batch size from query parameter (default to 1)
        batch_size = int(request.args.get("batch", 1))

        # Load sample input from `payload.json` and repeat based on batch size
        input_data = payload_template["inputs"] * batch_size

        # Load input into partitions
        stage_info = []
        for stage in stage_dict["stage_list"]:
            partition_shape = stage["partition_shape"]
            model_dict = {}
            for model in stage["models"]:
                fid = model["function_id"]
                model_dict[fid] = {
                    "input_shape": model["input_shape"],
                    "model_name": stage["coordinate"] + "_" + str(fid),
                    "input_data": input_data,  # Replace with partitioned input
                    "device": DEVICE,  # Send device type to workers
                }
            stage_info.append((partition_shape, model_dict))

        # Execute in parallel
        all_outputs = []
        all_latencies = []
        for _, model_dict in stage_info:
            outputs, latencies = do_parallel(model_dict)
            all_outputs.extend(outputs)
            all_latencies.extend(latencies)

        total_time = int((time.time() - start_time) * 1000)
        return jsonify({"outputs": all_outputs, "latencies": all_latencies, "total_time": total_time, "batch_size": batch_size, "device": DEVICE})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
