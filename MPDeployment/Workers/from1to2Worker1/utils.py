try:
    import unzip_requirements
except ImportError:
    pass
import time
import os
import urllib
import json
import numpy as np
import mxnet as mx
import struct
import base64
from collections import namedtuple
from functools import reduce
import gzip
import base64

Batch = namedtuple('Batch', ['data'])

def compress_data(data):
    """Compress and encode data using Gzip and Base64"""
    json_data = json.dumps(data).encode('utf-8')
    compressed_data = gzip.compress(json_data)
    return base64.b64encode(compressed_data).decode('utf-8')

def decompress_data(compressed_string):
    """Decode and decompress Base64-encoded Gzip data"""
    compressed_bytes = base64.b64decode(compressed_string)
    json_data = gzip.decompress(compressed_bytes).decode('utf-8')
    return json.loads(json_data)

def encode_array(tensor_data):
    shape = tensor_data.shape
    byte_data = struct.pack("<%df" % tensor_data.size, *tensor_data.flatten())
    base64_data = base64.b64encode(byte_data)
    string_data = str(base64_data)[2:-1]  # throw "b'" and "'"
    # result_json = json.dumps(string_data)
    return shape, string_data


def decode_array(shape, string_data):
    all_len = reduce(lambda a, b: a * b, shape)
    # string_data = json.loads(result_json)
    base64_data = bytes(string_data, 'utf-8')
    byte_data = base64.b64decode(base64_data)
    new_list = list(struct.unpack("<%df" % all_len, byte_data))
    return np.array(new_list).reshape(shape)


def get_splits(input, partition_shape, model_dict):
    
    sorted_keys = sorted(model_dict.keys())
    
    # partition Dense
    if isinstance(partition_shape, int):
        input_shape = model_dict[sorted_keys[0]]['input_shape'][1]
        return [input[:, :input_shape]] * partition_shape

    # filter non-4D tensor
    if len(input.shape) < 4:
        input_shape = model_dict[sorted_keys[0]]['input_shape'][1]
        return [input[:, :input_shape]]

    return_list = []
    dim2_size = input.shape[2]
    dim3_size = input.shape[3]
    dim2_num = partition_shape[0]
    dim3_num = partition_shape[1]
    for i in range(dim2_num):
        for j in range(dim3_num):
            idx = i * dim3_num + j
            dim2_delta = model_dict[sorted_keys[idx]]['input_shape'][2]
            dim3_delta = model_dict[sorted_keys[idx]]['input_shape'][3]

            if dim2_num < 2:
                dim2_s = 0
            else:
                dim2_s = i * ((dim2_size - dim2_delta) // (dim2_num - 1))
            if dim3_num < 2:
                dim3_s = 0
            else:
                dim3_s = j * ((dim3_size -  dim3_delta) // (dim3_num - 1))

            dim2_e = dim2_s + dim2_delta
            dim3_e = dim3_s + dim3_delta
            return_list.append(input[:, :, dim2_s:dim2_e, dim3_s:dim3_e])
    
    return return_list

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_model(model_name, shape, name='data', ctx = mx.cpu()):
    mod = mx.mod.Module(mx.symbol.load("./{}.json".format(model_name)), data_names=[name], label_names=None, context=ctx)
    mod.bind(for_training=False, data_shapes=[(name, shape)], label_shapes=None)
    mod.init_params()
    return mod

# def execute_model(event, mod):
    start = time.time()
    # input = np.asarray(event['data'])

    in_shape = event['input_shape']
    json_data = event['input_data']

    input = decode_array(in_shape, json_data)
    input = mx.nd.array(input)
    decode_clock = int((time.time() - start) * 1000)

    mod.forward(Batch([input]), is_train=False)
    output = mod.get_outputs()[0]
    np_output = output.asnumpy()
    exe_clock = int((time.time() - start) * 1000)

    # json_ret = json.dumps(np_output, cls=NumpyEncoder)    
    encode_clock = int((time.time() - start) * 1000)
    out_shape, json_ret = encode_array(np_output)

    response = {
        "statusCode": 200,
        "output": json_ret,
        "shape": out_shape,
        "latency": encode_clock,
        # "latency": [decode_clock, exe_clock, encode_clock],
    }

    return response

def execute_model(event, mod):
    try:
        print("Executing model...")  # Debugging
        start = time.time()

        # Decide device per request
        device = event.get("device", "cpu").lower()
        ctx = mx.gpu() if device == "gpu" else mx.cpu()
        print(f"Running on: {device.upper()}")
        
        if "input_shape" not in event or "input_data" not in event:
            return {"error": "Missing input data or shape"}

        in_shape = event["input_shape"]
        json_data = event["input_data"]

        print("Input Shape:", in_shape)
        print("Input Data (First 100 chars):", json_data[:100])

        input_array = decode_array(in_shape, json_data)
        input_ndarray = mx.nd.array(input_array, ctx=ctx)

        # Dynamically bind if needed
        if not getattr(mod, "_is_bound", False) or mod.data_shapes[0][1] != tuple(in_shape):
            print(f"Rebinding model for shape {in_shape}")
            mod.bind(
                for_training=False,
                data_shapes=[(mod.data_names[0], tuple(in_shape))],
                force_rebind=True
            )
            mod.init_params()
            mod._is_bound = True

        mod.forward(Batch([input_ndarray]), is_train=False)
        output = mod.get_outputs()[0].asnumpy()

        out_shape, json_ret = encode_array(output)

        latency = int((time.time() - start) * 1000)

        print("Model executed successfully.")  # 🔍 Debugging
        return {
            "statusCode": 200,
            "output": json_ret,
            "shape": out_shape,
            "latency": latency,
        }

    except Exception as e:
        print("Error in execute_model:", str(e))  # 🔍 Debugging
        return {"error": str(e)}
