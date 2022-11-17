import json
import tqdm
import torch
import argparse
import datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import evaluate

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizer
from datasets import load_dataset

dataset = load_dataset("json", data_files="lang_to_sem_data.json", field="data")
print(dataset)

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


# def main(args):
#     with open(args.in_data_fn, "r") as data:
#         # create train/val split
#         read = json.loads(data.read())
#         train_set = read["train"]
#         val_set = read["valid_seen"]
#         # dataset = load_dataset("json", data_files="lang_to_sem_data.json", field="data")
#         # dataset = datasets.DatasetDict({"train": train_set, "test": val_set})
#         # dataset = Dataset({})
#         # dataset = {'train': train_set({})}
#
#         # tokenize
#         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#         def tokenize_function(examples):
#             return tokenizer(examples, padding="max_length", truncation=True)
#
#         tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
#         # model setup
#         model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--in_data_fn", type=str, help="data file")
#     parser.add_argument(
#         "--model_output_dir", type=str, help="where to save model outputs"
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=32, help="size of each batch in loader"
#     )
#     parser.add_argument("--force_cpu", action="store_false", help="debug mode")
#     parser.add_argument("--eval", action="store_true", help="run eval")
#     parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
#     parser.add_argument(
#         "--val_every", default=5, help="number of epochs between every eval loop"
#     )
#
#     # ================== TODO: CODE HERE ================== #
#     # Task (optional): Add any additional command line
#     # parameters you may need here
#     # ===================================================== #
#     parser.add_argument("--voc_k", type=int, default=1000, help="vocab size")
#     parser.add_argument("--emb_dim", type=int, default=2, help="embedding dimensions")
#     parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
#
#     args = parser.parse_args()
#
#     main(args)




# model_name = ""
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
#
# res = classifier("I love you.")
#
# print(res)