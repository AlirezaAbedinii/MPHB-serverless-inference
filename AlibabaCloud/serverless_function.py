from flask import Flask, request, jsonify, json
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64 
import time

app = Flask(__name__)

# Determine the device dynamically based on the environment variable
device_type = os.getenv("DEVICE", "cpu")  # Set DEVICE to 'cpu' or 'cuda'
device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

# Load the model on the specified device
MODEL_PATH = "vgg19.pth"  # Path inside the container
model = models.vgg19(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/', methods=['POST'])
@app.route('/invoke', methods=['POST'])
def infer():
    try:
        start_time = time.time()
        # Parse incoming request
        data = request.get_json()
        inputs = data.get("inputs", [])
        if not inputs:
            return jsonify({"error": "No inputs provided"}), 400

        # Preprocess and batch inputs
        batched_tensors = []
        for image_data in inputs:
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            tensor = transform(image).unsqueeze(0)  # Add batch dimension
            batched_tensors.append(tensor)

        # Concatenate tensors into a batch and move to the correct device
        batched_tensor = torch.cat(batched_tensors).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(batched_tensor)
            predictions = outputs.argmax(dim=1).tolist()

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return jsonify({"predictions": predictions, "infMs": latency})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/', methods=['GET'])
@app.route('/invoke', methods=['GET'])
def get_infer():
    try:
        start_time = time.time()
        # Retrieve the batch size from the query parameters
        batch_size = request.args.get("BATCH", default=1, type=int)
        print(batch_size)

        # Read JSON input file
        with open("payload.json", 'r') as f:
            data = json.load(f)
        
        # Parse inputs from the JSON file
        inputs = data.get("inputs", [])
        if not inputs:
            return jsonify({"error": "No inputs provided"}), 400

        # Preprocess and batch inputs
        batched_tensors = []
        for _ in range(batch_size):  # Duplicate the same input for the specified batch size
            for image_data in inputs:
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                tensor = transform(image).unsqueeze(0)  # Add batch dimension
                batched_tensors.append(tensor)

        # Concatenate tensors into a batch and move to the correct device
        batched_tensor = torch.cat(batched_tensors).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(batched_tensor)
            predictions = outputs.argmax(dim=1).tolist()

        end_time = time.time()
        latency = (end_time - start_time)*1000
        return jsonify({"predictions": predictions, "infMs": latency})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
