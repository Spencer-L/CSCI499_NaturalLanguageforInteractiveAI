import argparse
import tqdm
import torch
import json
from torch.utils.data import TensorDataset, DataLoader


from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    create_train_val_splits,
    build_output_tables,
    encode_data,
)

def main(args):
    # Some hyperparameters
    validate_every_n_epochs = 10
    max_epochs = args.num_epochs
    minibatch_size = 256
    learning_rate = 0.0001

    print(f"CUDA version: {torch.version.cuda}")
    device = get_device(False)
    with open(args.in_data_fn, "r") as data:
        trainingData = json.loads(data.read())
        # print(trainingData["train"][0][0])

        # # example extraction of an instruction string
        # print(trainingData["train"][0][0][0])
        # # example extraction of an action string
        # print(trainingData["train"][0][0][1][0])
        # # example extraction of a location string
        # print(trainingData["train"][0][0][1][1])

        # read in and pre-process data
        # processedLines = process_instruction_set(args.in_data_fn)

        # create train/val splits
        train_lines, val_lines = create_train_val_splits(trainingData["train"])
        print(train_lines[0])

        # Tokenize the training set
        vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_lines, vocab_size=args.voc_k)
        actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_lines)
        print(vocab_to_index)
        print(actions_to_index)

        # Encode the training and validation set inputs/outputs.
        train_np_x, train_np_y = encode_data(train_lines, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
        print(train_np_x)
        train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
        val_np_x, val_np_y = encode_data(val_lines, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
        val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=minibatch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=minibatch_size)
        return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--voc_k", type=int, help="vocabulary size", required=True)
    parser.add_argument("--force_cpu", type=bool, help="debug mode")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")

    args = parser.parse_args()

    main(args)