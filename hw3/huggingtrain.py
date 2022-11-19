import json
import torch
import argparse
import pandas as pd

from utils import (get_device)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from transformers import TrainingArguments, Trainer


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, actions, targets):
        self.encodings = encodings
        self.actions = actions
        self.targets = targets

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # breakpoint()
        item['labels'] = torch.tensor(self.actions[idx])
        # TODO include targets as well 
        return item

def tokenize_episodes(episodes, tokenizer, label_to_index=None):
    df = pd.DataFrame(
        [(subgoal[0], subgoal[1][0], subgoal[1][1])
        for episode in episodes for subgoal in episode],
        columns=['command', 'action', 'target'] 
    )
    encodings = tokenizer(df['command'].to_list(), truncation=True, padding=True)

    if label_to_index is None:
        act_to_idx = {l: i for i, l in enumerate(df['action'].unique())}
        trgt_to_idx = {l: i for i, l in enumerate(df['target'].unique())}
        label_to_index = [act_to_idx, trgt_to_idx]
    
    def _label_to_index(label_to_index, x):
        idx = label_to_index.get(x)
        if idx is None:
            return label_to_index.get(0)
        return idx

    actions = df['action'].apply(lambda x: _label_to_index(label_to_index[0], x))
    targets = df['target'].apply(lambda x: _label_to_index(label_to_index[1], x))
    return encodings, actions, targets, label_to_index


def main(args):
    device = get_device(args.force_cpu)
    with open(args.in_data_fn, "r") as data:
        # create train/val split
        dataset = json.loads(data.read())
        train_episodes = dataset["train"]
        val_episodes = dataset["valid_seen"]

    print('Loading Tokenizer')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # TODO remove next line
    train_episodes = train_episodes[:100]

    print('Loading and tokenizing training dataset')
    encodings, actions, targets, label_to_index =\
        tokenize_episodes(train_episodes, tokenizer)
    train_dataset = LabeledDataset(encodings, actions, targets)
    
    print('Loading and tokenizing validation dataset')
    encodings, actions, targets, label_to_index =\
        tokenize_episodes(val_episodes, tokenizer, label_to_index=label_to_index)
    val_dataset = LabeledDataset(encodings, actions, targets)

    print('Loading classification model')
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", problem_type="multi_label_classification", num_labels=len(label_to_index)
    )
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    print('Setting up arguments')
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    print(training_args)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )
    # trainer.to(device)
    print(trainer)
    
    print('Starting training')
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", default="lang_to_sem_data.json", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
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