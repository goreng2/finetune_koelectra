import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Initialize
model = AutoModelForSequenceClassification.from_pretrained("result/koelectra-nsmc", torchscript=True)
tokenizer = AutoTokenizer.from_pretrained("result/koelectra-nsmc")
dummy_tensors = tokenizer("와 이거 정말 재밌다.", return_tensors="pt")["input_ids"]

# TorchScript
traced_model = torch.jit.trace(model, dummy_tensors)
torch.jit.save(traced_model, "torchscript_electra.pt")

# Post Training Dynamic Quantization
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

# Optimize
"""
By default, optimize_for_mobile will perform the following types of optimizations:
  - Conv2D and BatchNorm fusion which folds Conv2d-BatchNorm2d into Conv2d
  - Insert and fold prepacked ops which rewrites the model graph to replace 2D convolutions and linear ops with their prepacked counterparts.
  - ReLU and hardtanh fusion which rewrites graph by finding ReLU/hardtanh ops and fuses them together.
  - Dropout removal which removes dropout nodes from this module when training is false.
"""
traced_model = torch.jit.trace(model_dynamic_quantized, dummy_tensors)
optimized_model = optimize_for_mobile(traced_model)
optimized_model.save("optimized_torchscript_electra.pth")
