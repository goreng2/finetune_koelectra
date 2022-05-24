from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import argparse


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric("accuracy")


def preprocess_fn(examples):
    return tokenizer(examples["document"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(data, num_labels, batch_size):
    # Preprocess Data
    nsmc = load_dataset(data)
    tokenized_nsmc = nsmc.map(preprocess_fn, batched=True)
    tokenized_nsmc = tokenized_nsmc.remove_columns(["id", "document"])  # Unuse column
    tokenized_nsmc = tokenized_nsmc.rename_column("label", "labels")  # model expects 'labels'
    tokenized_nsmc.set_format("torch")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=num_labels)
    model.to(device)

    # Train
    training_args = TrainingArguments(
        output_dir="./result",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_nsmc["train"],
        eval_dataset=tokenized_nsmc["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KoELECTRA Text-classification")
    parser.add_argument("--data", type=str, default="nsmc", help="Data path")
    parser.add_argument("--num-labels", type=int, default=2, help="# of class")
    parser.add_argument("--batch-size", type=int, default=16, help="train/eval batch size")
    args = parser.parse_args()

    main(args.data, args.num_labels, args.batch_size)
