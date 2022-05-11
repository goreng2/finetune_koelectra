from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import time


class Item(BaseModel):
    text: str


app = FastAPI()


model = AutoModelForSequenceClassification.from_pretrained("rsc")
tokenizer = AutoTokenizer.from_pretrained("rsc")
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@app.post("/")
def main(item: Item):
    text = item.text
    start = time.time()
    result = classifier(text)[0]
    end = time.time() - start

    return {
        "label": result["label"],
        "score": float("{0:.2f}".format(result["score"])),
        "latency": float("{0:.2f}".format(end))
    }
