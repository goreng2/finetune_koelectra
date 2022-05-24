from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
import torch


path = "result/nsmc"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path, id2label={0: "부정", 1: "긍정"})
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

# torchscript_model = torch.jit.load("torchscript_electra.pt").eval()
# optimized_model = torch.jit.load("optimized_torchscript_electra.pth").eval()

while True:
    text = input("input: ")

    start = time.time()
    result = classifier(text)
    latency = time.time() - start
    print(result, "{0:.2f}".format(latency))

    # tensors = tokenizer(text, return_tensors="pt")["input_ids"]
    # start = time.time()
    # result = torchscript_model(tensors)
    # end = time.time() - start
    # print("[torchscript]", result, "{0:.2f}".format(end))
    #
    # start = time.time()
    # result = optimized_model(tensors)
    # end = time.time() - start
    # print("[Quantize]", result, "{0:.2f}".format(end))
