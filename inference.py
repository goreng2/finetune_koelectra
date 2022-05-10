from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time


path = "result/koelectra-nsmc"
model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

while True:
    text = input("input: ")
    start = time.time()
    result = classifier(text)
    end = time.time() - start
    print(result, end)
