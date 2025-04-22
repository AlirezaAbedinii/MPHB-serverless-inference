from flask import Flask, request, jsonify
import requests
import json
import time
import os
import base64
import gzip
import numpy as np
import mxnet as mx
from collections import namedtuple, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from utils import get_model, get_splits, encode_array, decode_array, compress_data, decompress_data
import urllib
import threading

app = Flask(__name__)
Batch = namedtuple("Batch", ["data"])

# URLs for each deployed worker function
url_dict = {
    'from1to2Worker1': 'https://fromtoworker-ggcazmvmzo.us-east-1-vpc.fcapp.run/invoke',
    'from11to11Worker1': 'https://fromtoworker-qmgeqcthxf.us-east-1-vpc.fcapp.run/invoke',
    'from11to11Worker2': 'https://fromtoworker-qlgeqcthxf.us-east-1-vpc.fcapp.run/invoke',
    'from11to11Worker3': 'https://fromtoworker-qkgeqcthxf.us-east-1-vpc.fcapp.run/invoke',
    'from15to15Worker1': 'https://fromtoworker-qmgaqcthxj.us-east-1-vpc.fcapp.run/invoke',
    'from15to15Worker2': 'https://fromtoworker-qlgaqcthxj.us-east-1-vpc.fcapp.run/invoke',
    'from15to15Worker3': 'https://fromtoworker-qkgaqcthxj.us-east-1-vpc.fcapp.run/invoke',
    'from16to16Worker1': 'https://fromtoworker-qmgdqcthxg.us-east-1-vpc.fcapp.run/invoke',
    'from16to16Worker2': 'https://fromtoworker-qlgdqcthxg.us-east-1-vpc.fcapp.run/invoke',
    'from16to16Worker3': 'https://fromtoworker-qkgdqcthxg.us-east-1-vpc.fcapp.run/invoke',
    'from20to20Worker1': 'https://fromtoworker-qmffqcthue.us-east-1-vpc.fcapp.run/invoke',
    'from20to20Worker2': 'https://fromtoworker-qlffqcthue.us-east-1-vpc.fcapp.run/invoke',
    'from20to20Worker3': 'https://fromtoworker-qkffqcthue.us-east-1-vpc.fcapp.run/invoke',
    'from21to21Worker1': 'https://fromtoworker-qmfeqcthuf.us-east-1-vpc.fcapp.run/invoke'
}

# Load structure and grids
with open("structure.json", "r") as f:
    stage_dict = json.load(f)

with open("payload.json", "r") as f:
    payload_template = json.load(f)

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/multimedia-berkeley/tutorials/master/grids.txt", "/tmp/grids.txt"
)

grids = []
with open("/tmp/grids.txt", "r") as f:
    for line in f:
        lat, lng = float(line.split('\t')[1]), float(line.split('\t')[2])
        grids.append((lat, lng))

# ENV
DEVICE = os.getenv("DEVICE", "cpu").lower()
DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# Model setup
mydict = lambda: defaultdict(mydict)
stage_info = []
master_models = {}

for stage in stage_dict["stage_list"]:
    coord = stage["coordinate"]
    partition_shape = stage["partition_shape"]
    model_dict = mydict()
    for model in stage["models"]:
        fid = model["function_id"]
        model_name = f"{coord}_{fid}"
        model_dict[fid]["input_shape"] = model["input_shape"]
        model_dict[fid]["model_name"] = model_name
        if fid == 0:
            master_models[model_name] = get_model(model_name, model["input_shape"], model["input_name"])
    stage_info.append((partition_shape, model_dict))

### ✅ NEW: Split → Then Batch
def get_splits_with_batch(single_input, partition_shape, model_dict, batch_size):
    splits = get_splits(single_input, partition_shape, model_dict)
    return [mx.nd.concat(*[split] * batch_size, dim=0) for split in splits]

### Worker Call
def invoke_worker(func_dict, max_retries = 4, delay = 1):
    split_name = func_dict["model_name"].split("_")
    worker_name = f"from{split_name[0]}to{split_name[1]}Worker{split_name[2]}"
    url = url_dict.get(worker_name)
    if not url:
        log(f"Missing URL for {worker_name}")
        return None, -1
    try:
        payload = compress_data(func_dict)
        # log(f"Request sent to: {worker_name} with input_shape={func_dict['input_shape']}")
        log(f"[DEBUG] Payload actual shape: {func_dict['input_data'].shape if hasattr(func_dict['input_data'], 'shape') else 'encoded'}")

        for attempt in range(max_retries):
            try:
                log(f"[Request] Request sent to: {worker_name} with input_shape={func_dict['input_shape']}")
                res = requests.post(url, json={"compressed_data": payload, "device": DEVICE}, 
                                    headers={"Content-Type": "application/json"}, timeout=120)
                res.raise_for_status()
                result = res.json()
                decompressed = decompress_data(result["compressed_data"])
                log(f"[Response] Response received from: {worker_name}")
                np_output = decode_array(decompressed["shape"], decompressed["output"])
                return mx.nd.array(np_output), decompressed["latency"]
            except requests.exceptions.HTTPError as e:
                if res.status_code in [429, 503] and attempt < max_retries - 1:
                    # log(f"[HEADERS] {res.headers}")
                    log(f"[Retry] 429 Too Many Requests from {worker_name}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    continue
                log(f"Worker {worker_name} failed: {e}")
                return None, -1
    except Exception as e:
        log(f"Worker {worker_name} failed: {e}")
        return None, -1

### Parallel Execution
def do_parallel(model_dict):
    outputs, futures, lats = [], [], []
    master_input = None
    master_model = None

    for fid, fd in model_dict.items():
        if fid == 0:
            master_input = fd["input_data"]
            master_model = master_models[fd["model_name"]]
        else:
            shape, encoded = encode_array(fd["input_data"].asnumpy())
            fd["input_data"] = encoded
            fd["input_shape"] = shape
    max_workers = max(1, len(model_dict)-1)
    log(f"max workers = {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fid, fd in model_dict.items():
            if fid != 0:
                futures.append(pool.submit(invoke_worker, fd))
                # time.sleep(1)
                # log(f"one request sent with fid of {fid} and fd of {fd}")

    if master_input is not None and master_model is not None:
        master_model.forward(Batch([master_input]), is_train=False)
        outputs.append(master_model.get_outputs()[0])

    for f in futures:
        result, latency = f.result()
        if result is not None:
            outputs.append(result)
            lats.append(latency)

    return outputs, lats

### Load test image
def load_data():
    img_path = "/tmp/img.jpg"
    urllib.request.urlretrieve(
        "https://farm5.staticflickr.com/4275/34103081894_f7c9bfa86c_k_d.jpg", img_path
    )
    img = mx.image.imread(img_path)
    img = mx.image.imresize(img, 224, 224).transpose((2, 0, 1)).expand_dims(axis=0)
    return img

def warmup_workers(model_dict):
    for fid, fd in model_dict.items():
        if fid == 0:
            continue
        model_name = fd["model_name"]
        worker_name = f"from{model_name.split('_')[0]}to{model_name.split('_')[1]}Worker{model_name.split('_')[2]}"
        url = url_dict.get(worker_name)
        if not url:
            continue
        try:
            ping_url = url.replace("/invoke", "/health")
            log(f"[WARMUP] Warming {worker_name} up")
            res = requests.get(ping_url, timeout=30)
            if res.status_code == 200:
                log(f"[WARMUP] {worker_name} warmed up")
            else:
                log(f"[WARMUP] Failed for {worker_name}: {res.status_code} error")
        except Exception as e:
            log(f"[WARMUP] Failed for {worker_name}: {e}")


@app.route("/invoke", methods=["GET"])
def infer():
    try:
        start = time.time()
        batch_size = int(request.args.get("batch", 1))

        ## 1. Loading Image
        t0 = time.time()
        img = load_data()
        log(f"[TIME] load_data: {(time.time() - t0) * 1000:.2f} ms")

        input_data = img
        all_latencies = []

        ## 2. Each Stage
        for stage_idx, (partition_shape, model_dict) in enumerate(stage_info):
            stage_start = time.time()

            log(f"input data shape: {input_data.shape}, partition shape: {partition_shape}, batch_size: {batch_size}")

            ## 2a. Splitting
            t0 = time.time()
            single_splits = get_splits(input_data, partition_shape, model_dict)
            log(f"[TIME] get_splits: {(time.time() - t0) * 1000:.2f} ms")

            ## 2b. Batching
            t0 = time.time()
            if input_data.shape[0] < batch_size:
                splits = [mx.nd.concat(*[split] * batch_size, dim=0) for split in single_splits]
            else:
                splits = single_splits
            log(f"[TIME] batching splits: {(time.time() - t0) * 1000:.2f} ms")

            ## 2c. Assign splits
            for i, fid in enumerate(sorted(model_dict.keys())):
                model_dict[fid]["input_data"] = splits[i]
                log(f"[DEBUG] Assigned split {i} to worker {fid}, shape: {splits[i].shape}")

            ## 2d. Warm up next stage
            if stage_idx + 1 < len(stage_info):
                t0 = time.time()
                next_model_dict = stage_info[stage_idx + 1][1]
                # warmup_workers(next_model_dict)
                threading.Thread(target=warmup_workers, args=(next_model_dict,), daemon=True).start()
                log(f"[TIME] warmup next stage: {(time.time() - t0) * 1000:.2f} ms")

            ## 2e. Do Parallel Inference
            t0 = time.time()
            outputs, lats = do_parallel(model_dict)
            log(f"[TIME] do_parallel: {(time.time() - t0) * 1000:.2f} ms")

            ## 2f. Combine outputs
            t0 = time.time()
            if isinstance(partition_shape, int):
                input_data = reduce(lambda x, y: mx.nd.concat(x, y, dim=1), outputs)
            else:
                dim2, dim3 = partition_shape
                rows = []
                for i in range(dim2):
                    rows.append(reduce(lambda x, y: mx.nd.concat(x, y, dim=3),
                                       outputs[i * dim3: (i + 1) * dim3]))
                input_data = reduce(lambda x, y: mx.nd.concat(x, y, dim=2), rows)
            log(f"[TIME] recombine outputs: {(time.time() - t0) * 1000:.2f} ms")

            all_latencies.extend(lats)
            log(f"[Stage {stage_idx} Complete] outputs: {len(outputs)}, lats: {lats}")
            log(f"[TIME] Total Stage {stage_idx} Time: {(time.time() - stage_start) * 1000:.2f} ms")

        ## 3. Final Result
        result = input_data.asnumpy()
        pred = np.argsort(result[0])[::-1]
        pred_loc = grids[int(pred[0])]
        total_time = int((time.time() - start) * 1000)

        log(f"[TIME] Total inference time: {total_time} ms")

        return jsonify({
            "prediction": pred_loc,
            "latencies": all_latencies,
            "total_time": total_time,
            "device": DEVICE
        })

    except Exception as e:
        log(f"Failed inference: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9091)
