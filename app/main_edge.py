from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import time


class Item(BaseModel):
    text: str


app = FastAPI()

model = torch.jit.load("rsc/optimized_traced_electra.pth")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("rsc")
softmax = torch.nn.Softmax(dim=1)


@app.post("/")
def main(item: Item):
    text = item.text
    tensors = tokenizer(text, return_tensors="pt")["input_ids"]
    start = time.time()
    result = model(tensors)[0]
    end = time.time() - start

    return {
        "label": int(result.argmax()),
        "score": float("{0:.2f}".format(torch.max(softmax(result)))),
        "latency": float("{0:.2f}".format(end))
    }
