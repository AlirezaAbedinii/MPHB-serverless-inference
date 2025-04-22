import time 
import os
import json
import numpy as np
import mxnet as mx
import struct
import base64
from collections import namedtuple
from functools import reduce
import gzip

Batch = namedtuple("Batch", ["data"])

###  Compression Helper Functions ###
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

###  Encoding/Decoding Helper Functions ###
def encode_array(tensor_data):
    """Encodes a NumPy array into a compressed Base64 string"""
    shape = tensor_data.shape
    byte_data = struct.pack("<%df" % tensor_data.size, *tensor_data.flatten())
    base64_data = base64.b64encode(byte_data)
    return shape, base64_data.decode("utf-8")  # Convert to UTF-8 string

def decode_array(shape, string_data):
    """Decodes a Base64 string back into a NumPy array"""
    all_len = reduce(lambda a, b: a * b, shape)
    byte_data = base64.b64decode(string_data.encode("utf-8"))
    new_list = list(struct.unpack("<%df" % all_len, byte_data))
    return np.array(new_list).reshape(shape)

###  Model Loading Function ###
def get_model(model_name, shape, name="data"):
    """Loads a partitioned MXNet model from a JSON file"""
    model_path = f"partitions/{model_name}.json"  # Model JSON file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    mod = mx.mod.Module(mx.symbol.load(model_path), data_names=[name], label_names=None)
    mod.bind(for_training=False, data_shapes=[(name, shape)], label_shapes=None)
    mod.init_params()
    return mod

###  Model Execution Function ###
def execute_model(event, mod):
    """Executes model inference and returns compressed output"""
    start_time = time.time()

    in_shape = event["input_shape"]
    json_data = event["input_data"]

    input_tensor = decode_array(in_shape, json_data)
    input_tensor = mx.nd.array(input_tensor)
    
    decode_time = int((time.time() - start_time) * 1000)

    mod.forward(Batch([input_tensor]), is_train=False)
    output = mod.get_outputs()[0].asnumpy()

    encode_time = int((time.time() - start_time) * 1000)
    out_shape, json_ret = encode_array(output)

    return {
        "statusCode": 200,
        "output": json_ret,
        "shape": out_shape,
        "latency": encode_time
    }

###  Partitioning Function ###
# def get_splits(input_data, partition_shape, model_dict):
#     """Splits input data into partitioned sections for distributed inference"""
#     sorted_keys = sorted(model_dict.keys())

#     # Handle Dense Partitioning
#     if isinstance(partition_shape, int):
#         input_shape = model_dict[sorted_keys[0]]["input_shape"][1]
#         return [input_data[:, :input_shape]] * partition_shape

#     # Handle Multi-Dimensional Partitioning
#     if len(input_data.shape) < 4:
#         input_shape = model_dict[sorted_keys[0]]["input_shape"][1]
#         return [input_data[:, :input_shape]]

#     return_list = []
#     dim2_size, dim3_size = input_data.shape[2], input_data.shape[3]
#     dim2_num, dim3_num = partition_shape

#     for i in range(dim2_num):
#         for j in range(dim3_num):
#             idx = i * dim3_num + j
#             dim2_delta = model_dict[sorted_keys[idx]]["input_shape"][2]
#             dim3_delta = model_dict[sorted_keys[idx]]["input_shape"][3]

#             dim2_s = i * ((dim2_size - dim2_delta) // (dim2_num - 1)) if dim2_num > 1 else 0
#             dim3_s = j * ((dim3_size - dim3_delta) // (dim3_num - 1)) if dim3_num > 1 else 0

#             dim2_e = dim2_s + dim2_delta
#             dim3_e = dim3_s + dim3_delta

#             return_list.append(input_data[:, :, dim2_s:dim2_e, dim3_s:dim3_e])

#     return return_list

# def get_splits(input_data, partition_shape, model_dict):
#     """
#     Splits input_data for parallel inference based on partition_shape.
#     Handles batching and avoids out-of-bounds slicing errors.
#     """
#     sorted_keys = sorted(model_dict.keys())

#     batch_size = input_data.shape[0]

#     # Handle Dense partition (e.g. fully connected layer)
#     if isinstance(partition_shape, int):
#         input_channels = model_dict[sorted_keys[0]]["input_shape"][1]
#         return [input_data[:, :input_channels]] * partition_shape

#     # Handle non-4D tensor
#     if len(input_data.shape) < 4:
#         input_channels = model_dict[sorted_keys[0]]["input_shape"][1]
#         return [input_data[:, :input_channels]]

#     return_list = []

#     _, channels, dim2_size, dim3_size = input_data.shape
#     dim2_num, dim3_num = partition_shape

#     for i in range(dim2_num):
#         for j in range(dim3_num):
#             idx = i * dim3_num + j
#             model_shape = model_dict[sorted_keys[idx]]["input_shape"]

#             _, _, delta2, delta3 = model_shape

#             dim2_s = i * ((dim2_size - delta2) // (dim2_num - 1)) if dim2_num > 1 else 0
#             dim3_s = j * ((dim3_size - delta3) // (dim3_num - 1)) if dim3_num > 1 else 0

#             dim2_e = min(dim2_s + delta2, dim2_size)
#             dim3_e = min(dim3_s + delta3, dim3_size)

#             sliced = input_data[:, :, dim2_s:dim2_e, dim3_s:dim3_e]

#             # Sanity check: pad if needed
#             if sliced.shape[2] != delta2 or sliced.shape[3] != delta3:
#                 padded = mx.nd.zeros((batch_size, channels, delta2, delta3))
#                 padded[:, :, :sliced.shape[2], :sliced.shape[3]] = sliced
#                 sliced = padded

#             return_list.append(sliced)

#     return return_list

def get_splits_with_batch(input_data, partition_shape, model_dict, batch_size):
    """
    Splits a single input image, then stacks the split for the full batch.
    """
    # Step 1: Remove batch dimension and split one image (1, C, H, W)
    input_single = input_data[0:1]  # shape (1, C, H, W)
    single_splits = get_splits(input_single, partition_shape, model_dict)  # List[(1, C, h, w)]

    # Step 2: Replicate each split to match batch size
    # if input_data.shape[0] < batch_size:
    batched_splits = [mx.nd.concat(*[s] * batch_size, dim=0) for s in single_splits]

    return batched_splits


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
