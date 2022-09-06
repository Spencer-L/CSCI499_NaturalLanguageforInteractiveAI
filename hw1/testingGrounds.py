import argparse
import tqdm
import torch
import json

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    process_instruction_set,
    create_train_val_splits,
    build_output_tables,
)

def main(args):
    device = get_device(False)
    # with open(args.in_data_fn, "r") as data:
    #     trainingData = json.loads(data.read())
        # print(trainingData["train"][0][0])
        
        # # example extraction of an instruction string
        # print(trainingData["train"][0][0][0])
        # # example extraction of an action string
        # print(trainingData["train"][0][0][1][0])
        # # example extraction of a location string
        # print(trainingData["train"][0][0][1][1])
    
    
    # read in and pre-process data
    print(f"CUDA version: {torch.version.cuda}")
    processedLines = process_instruction_set(args.in_data_fn)
    
    # create train/val splits
    train_lines, val_lines = create_train_val_splits(processedLines)
    print(train_lines[0])
    
    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_lines, vocab_size=args.voc_k)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_lines)
    print(vocab_to_index)
    print(actions_to_index)

    # Encode the training and validation set inputs/outputs.



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--voc_k", type=int, help="vocabulary size", required=True)
    parser.add_argument("--force_cpu", type=bool, help="debug mode")

    args = parser.parse_args()

    main(args)

