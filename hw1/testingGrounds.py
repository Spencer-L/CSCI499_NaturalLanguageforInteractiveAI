import argparse
import tqdm
import torch
import json

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    process_instruction_set,
    create_train_val_splits,
    build_output_table,
)

def main(args):
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
    processedLines = process_instruction_set(args.in_data_fn)
    
    # create train/val splits
    train_lines, val_lines = create_train_val_splits(processedLines)
    print(train_lines[0])
    
    # Tokenize the training set
    inst_to_index, index_to_inst, len_cutoff = build_tokenizer_table([train_lines], vocab_size=args.voc_k)
    pairs_to_index, index_to_pairs = build_output_table(train_lines)
    print(pairs_to_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--voc_k", type=int, help="vocabulary size", required=True)

    args = parser.parse_args()

    main(args)

