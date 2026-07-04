import torch
import os
os.environ["HF_HOME"] = "/mnt/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/mnt/data/huggingface"

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load GPT-2 XL
model_name = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  # set to inference mode

# 2. Prepare dummy input (batch=1, sequence length=16)
text = "Hello, this is a test"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]  # shape: [1, seq_len]

# 3. Define ONNX export path
onnx_model_path = "gpt2-xl.onnx"

# 4. Export to ONNX
class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # return only logits (ignore caches, hidden states, etc.)
        return self.model(input_ids, return_dict=False)[0]

wrapper = GPT2Wrapper(model)

torch.onnx.export(
    wrapper,
    (input_ids,),
    "gpt2-xl.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=14,
    do_constant_folding=True
)


print(f"Exported GPT-2 XL to {onnx_model_path}")
