
from flask import Flask, request, jsonify
import os
from utils import *
import time

app = Flask(__name__)

# Get model name and shape from environment variables
WORKER_ID = os.getenv("WORKER_ID", "11_11_1")  # Example: "1_2_1"
INPUT_SHAPE = [1, 256, 15, 15]  # Modify based on the partitioned model
INPUT_NAME = "vgg0_pool2_fwd"

# Load model
# model = get_model(WORKER_ID, INPUT_SHAPE, INPUT_NAME)
models = {}

@app.route('/invoke', methods=['POST'])
def infer():
    """Receive and process inference requests"""
    try:
        if request.content_type != 'application/json':
            return jsonify({"error": "Unsupported Media Type. Use application/json."}), 415
        data = request.get_json()

        start = time.time()
        # Get compressed input data
        compressed_input = data.get("compressed_data", None)
        if compressed_input is None:
            return jsonify({"error": "Missing compressed data"}), 400

        # Decompress input data
        decompressed_event = decompress_data(compressed_input)
        decompressed_event['device'] = data.get("device", 'cpu')

        end = time.time()
        print('compression time: ', end - start)

        
        device = data.get("device", 'cpu').lower()
        ctx = mx.gpu() if device == 'gpu' else mx.cpu()

        # Load model if not cached
        if device not in models:
            print(f"[LOADING MODEL] for device: {device}")
            models[device] = get_model(WORKER_ID, INPUT_SHAPE, INPUT_NAME, ctx)
        else:
            print(f"[LOADING MODEL] using cached model")
            
        model = models[device]
        
        
        # Execute model inference
        response = execute_model(decompressed_event, model)

        if "error" in response:
            return jsonify(response), 500
        
        # Compress response before sending it back
        compressed_response = compress_data(response)

        return jsonify({"compressed_data": compressed_response})

    except Exception as e:
        print(f"[ERROR] Worker {WORKER_ID} failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
