from flask import Flask, request, jsonify
import os
from utils import *

app = Flask(__name__)

# Get model name and shape from environment variables
WORKER_ID = os.getenv("WORKER_ID", "21_21_1")  # Unique ID for each worker
INPUT_SHAPE = [1, 25088]  # Shape of the model input
INPUT_NAME = "flatten_72"  # Model input name

# Load model
model = get_model(WORKER_ID, INPUT_SHAPE, INPUT_NAME)

@app.route('/invoke', methods=['POST'])
def infer():
    """Receive and process inference requests"""
    try:
        data = request.get_json()

        # Get compressed input data
        compressed_input = data.get("compressed_data", None)
        if compressed_input is None:
            return jsonify({"error": "Missing compressed data"}), 400

        # Decompress input data
        decompressed_event = decompress_data(compressed_input)
        
        # Execute model inference
        response = execute_model(decompressed_event, model)

        # Compress response before sending it back
        compressed_response = compress_data(response)

        return jsonify({"compressed_data": compressed_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
