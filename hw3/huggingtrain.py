import json

import torch
import argparse
import datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import evaluate
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer
from datasets import load_dataset

# data_files = {
#     "train": "lang_to_sem_data.json"
# }
#
# dataset = load_dataset("json", data_files=data_files)
# breakpoint()
# print(dataset)

# def main(args):
#
#     dataset = load_dataset("SetFit/yelp_review_full")
#
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
#     print(tokenized_datasets)
#
#     # model creation
#     model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#
#     # training
#     training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
#     small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#     small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#     # evaluation
#     metric = evaluate.load("accuracy")
#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         return metric.compute(predictions=predictions, references=labels)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=small_train_dataset,
#         eval_dataset=small_eval_dataset,
#         compute_metrics=compute_metrics,
#     )
#     trainer.train()
import pandas as pd
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, episodes, tokenizer: BertTokenizer):
        self.features, self.labels = encode_data(episodes, tokenizer)
        breakpoint()

        # df = pd.DataFrame([subgoal for episode in episodes for subgoal in episode])
        # self.features = df[0].values
        # self.target = df[1].values
        # self.features = tokenizer(list(, padding=True, return_tensors="pt").input_ids
        # breakpoint()
        # self.features =\
        #     np.array(list(map(
        #         lambda x: tokenizer(x, return_tensors="pt", padding=True).input_ids,
        #         self.features.values
        #     )))
        # breakpoint()
        #
        # self.target = \
        #     self.target.applymap(
        #         lambda x: (tokenizer(x[0], return_tensors="pt").input_ids, tokenizer(x[1], return_tensors="pt").input_ids)
        #     )
        # breakpoint()

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def encode_data(data, tokenizer:BertTokenizer):
    x = []
    y = []
    for episode in tqdm(data, desc='episodes'):
        for txt, [act, loc] in episode:
            txt_emb = tokenizer(txt, return_tensors="pt", padding=True).input_ids
            x.append(txt_emb)
            act_emb = tokenizer(act, return_tensors="pt").input_ids
            loc_emb = tokenizer(loc, return_tensors="pt").input_ids
            y.append(([act_emb, loc_emb]))
    x = np.array(x)
    y = np.array(y)
    return x, y

from transformers import BertTokenizer, EncoderDecoderModel
def main(args):
    with open(args.in_data_fn, "r") as data:
        # create train/val split
        dataset = json.loads(data.read())
        train_set = dataset["train"]
        val_set = dataset["valid_seen"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(LabeledDataset(train_set, tokenizer), batch_size=32)
    breakpoint()

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    input_ids = tokenizer(train_set[0][0][0], return_tensors="pt").input_ids
    labels = tokenizer(train_set[0][0][1], return_tensors="pt").input_ids

    loss = model(input_ids=input_ids, labels=labels).loss

    # def tokenize_function(examples):
    #     return tokenizer(examples, padding="max_length", truncation=True)
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_false", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--voc_k", type=int, default=1000, help="vocab size")
    parser.add_argument("--emb_dim", type=int, default=2, help="embedding dimensions")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")

    args = parser.parse_args()

    main(args)