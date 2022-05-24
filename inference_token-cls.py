from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import time
import os
import pickle
import torch


path = "result/klue_ner"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForTokenClassification.from_pretrained(path)
classifier = TokenClassificationPipeline(task="token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
# with open(os.path.join(path, "id2label.pkl"), "rb") as f:
#     id2label = pickle.load(f)

# torchscript_model = torch.jit.load("torchscript_electra.pt").eval()
# optimized_model = torch.jit.load("optimized_torchscript_electra.pth").eval()

while True:
    text = input("input: ")
    # tensors = tokenizer(text, return_tensors="pt")["input_ids"]

    start = time.time()
    result = classifier(text)
    end = time.time() - start
    print(result, end)
    # print("[original]", result, "{0:.2f}".format(end))
    # print(id2label[int(result[0]["label"].split("_")[-1])])

    # start = time.time()
    # result = torchscript_model(tensors)
    # end = time.time() - start
    # print("[torchscript]", result, "{0:.2f}".format(end))
    #
    # start = time.time()
    # result = optimized_model(tensors)
    # end = time.time() - start
    # print("[Quantize]", result, "{0:.2f}".format(end))
