from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

"""
Exporting a model requires two things:
- a forward pass with dummy inputs
- model instantiation with the torchscript flag
"""
# Tokenizing input text
tokenizer = AutoTokenizer.from_pretrained("result/koelectra-nsmc")
# tensors = tokenizer("와 이거 정말 재밌다.", return_tensors="pt")["input_ids"]

# Save TorchScript
# model = AutoModelForSequenceClassification.from_pretrained("result/koelectra-nsmc", torchscript=True)  # untie weights between Embedding layer and Decoding layer
# traced_model = torch.jit.trace(model, tensors)
# print(traced_model.code)
# torch.jit.save(traced_model, "traced_electra.pt")

# Load TorchScript
# C++에서 모델 로딩 가능
loaded_model = torch.jit.load("traced_electra.pt")
loaded_model.eval()

tensors = tokenizer("와 이거 정말 재밌다. 미친거같아! 어떻게 이럴수가 있지", return_tensors="pt")["input_ids"]

start = time.time()
result = loaded_model(tensors)[0]
end = time.time() - start
print(result, end)

print(int(result.argmax()))
softmax = torch.nn.Softmax(dim=1)
print(float(torch.max(softmax(result))))