import re
import torch
import numpy as np
from collections import Counter
import json


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i for i, a in enumerate(actions)}
    targets_to_index = {t: i for i, t in enumerate(targets)}
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def process_instruction_set(d):
    processed_set = []
    with open(d, "r") as data:
        trainingData = json.loads(data.read())
        index = 0
        new_instructions = [[preprocess_string(ep[0]), [preprocess_string(ep[1][0]), preprocess_string(ep[1][1])]] for ep in trainingData["train"] if  len(preprocess_string(ep)) > 0]
        for episodes in trainingData["train"]:
            ep_set = []
            for ep in episodes:
                # preprocess instructions
                # if len(preprocess_string(ep[0])) > 0 and len(preprocess_string(ep[1][0])) > 0 and len(preprocess_string(ep[1][0])) > 0:
                new_instructions = [preprocess_string(ep[0]), [preprocess_string(ep[1][0]), preprocess_string(ep[1][1])]]
                ep_set.append(new_instructions)
            processed_set.extend([ep_set])
            index += 1
            print(index)
        return processed_set

def create_train_val_splits(all_lines, prop_train=0.8):
    train_lines = []
    val_lines = []
    print(range(len(all_lines)))
    lines = [all_lines[idx] for idx in range(len(all_lines))]
    val_idxs = np.random.choice(list(range(len(lines))), size=int(len(lines) * prop_train + 0.5), replace=False)
    train_lines.extend([lines[idx] for idx in range(len(lines)) if idx not in val_idxs])
    val_lines.extend([lines[idx] for idx in range(len(lines)) if idx in val_idxs])

    return train_lines, val_lines

def encode_data(data, v2i, seq_len, a2i, l2i):
    n_lines = len(data)
    n_acts = len(a2i)
    n_locs = len(l2i)
    x = []
    y = []

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for episode in data:
        for txt, [act, loc] in episode:
            txt = preprocess_string(txt)
            inst_embed = np.zeros(seq_len)
            inst_embed[0] = v2i["<start>"]
            jdx = 1
            for word in txt.split():
                if len(word) > 0:
                    inst_embed[jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                    n_unks += 1 if inst_embed[jdx] == v2i["<unk>"] else 0
                    n_tks += 1
                    jdx += 1
                    if jdx == seq_len - 1:
                        n_early_cutoff += 1
                        break
            inst_embed[jdx] = v2i["<end>"]
            label_embed = np.zeros(2)
            # label_embed[a2i[act]] = 1.0
            # label_embed[l2i[loc]] = 1.0
            label_embed[0] = a2i[act]
            label_embed[1] = l2i[loc]
            idx += 1
            x.append(inst_embed)
            y.append(label_embed)
    x = np.array(x, dtype=np.int32)
    y = np.array(y)
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d instances at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d instances without regard to order" % idx)
    return x, y