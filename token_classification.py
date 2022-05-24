from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
import numpy as np
import argparse


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = load_metric("seqeval")


def preprocess_fn(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 토큰을 어절의 인덱스로 매핑
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:  # 스페셜 토큰 [CLS]를 '-100' 태깅
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # 어절의 첫 토큰만 실제 레이블로 태깅
                label_ids.append(label[word_idx])
            else:  # 어절의 나머지 토큰, 스페셜 토큰 [SEP]을 '-100' 태깅
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # '-100' 태그 제거
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def main(data, batch_size):
    # Data
    global label_list
    dataset = load_dataset(data, "ner")
    # print(dataset)
    # print(dataset["train"][0])
    label_list = dataset["train"].features["ner_tags"].feature.names
    # print("[label_list]", label_list)
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    # print("[tokenized_dataset]", tokenized_dataset)
    # print(tokenized_dataset["train"][0])

    # Model
    id2label = {str(i): label for i, label in enumerate(label_list)}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                                            id2label=id2label,
                                                            label2id=label2id)
    model.to(device)

    # Train
    training_args = TrainingArguments(
        output_dir="./klue_ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=20,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KoELECTRA Token-classification")
    parser.add_argument("--data", type=str, default="klue", help="Data path")
    parser.add_argument("--batch-size", type=int, default=32, help="train/eval batch size")
    args = parser.parse_args()

    main(args.data, args.batch_size)
