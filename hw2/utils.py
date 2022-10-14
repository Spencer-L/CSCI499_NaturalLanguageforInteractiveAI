import json
import gensim
import tqdm
import numpy as np
import torch

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


def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.smart_open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )

def create_train_val_splits(all_sentences, prop_train=0.8):
    train_words = []
    val_words = []
    words = []
    # sentences = [all_sentences[idx] for idx in range(len(all_sentences))]
    for sentence in all_sentences:
        for word in sentence:
            if word != 0 and word not in words:
                words.append(word)
    val_idxs = np.random.choice(list(range(len(words))), size=int(len(words) * prop_train + 0.5), replace=False)

    for idx in range(len(words)):
        if idx in val_idxs:
            train_words.extend([words[idx]])
        else:
            val_words.extend([words[idx]])
        # train_words.extend([words[idx] for idx in range(len(words)) if idx not in val_idxs])
        # val_words.extend([words[idx] for idx in range(len(words)) if idx in val_idxs])

    return train_words, val_words