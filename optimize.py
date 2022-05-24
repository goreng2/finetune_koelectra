import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import AutoModelForSequenceClassification, AutoTokenizer


path = "result/nsmc_small"
torchscript_name = "torchscript_electra-small.pt"
optimized_name = "optimized_torchscript_electra-small.pth"

# Initialize
model = AutoModelForSequenceClassification.from_pretrained(path, torchscript=True)
tokenizer = AutoTokenizer.from_pretrained(path)
dummy_tensors = tokenizer("와 이거 정말 재밌다.", return_tensors="pt")["input_ids"]

# TorchScript
traced_model = torch.jit.trace(model, dummy_tensors)
torch.jit.save(traced_model, torchscript_name)
print("Save TorchScript model!")

# Post Training Dynamic Quantization
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
print("Adjust Quantization!")

# Optimize
traced_model = torch.jit.trace(model_dynamic_quantized, dummy_tensors)
optimized_model = optimize_for_mobile(traced_model)
optimized_model.save(optimized_name)
print("Save Optimized model!")