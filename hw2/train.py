import argparse
import os
import tqdm
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
import numpy as np

from eval_utils import downstream_validation
import utils
import data_utils
from utils import(
    create_train_val_splits
)

from model import (CBOW_Model)

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    # create train/val splits
    # train_words, val_words = create_train_val_splits(encoded_sentences)

    # make contexts
    # def transform_sentence(sentence, pbar):
    #     pbar.update(1)
    #     context_target = []
    #     for i in range(2, len(sentence) - 2):
    #         context = [sentence[i + 2], sentence[i + 1],
    #                    sentence[i - 1], sentence[i - 2]]
    #         target = sentence[i]
    #         context_target.append((context, target))
    #         if sentence[i + 2] == vocab_to_index['<end>']:
    #             break
    #     return np.array(context_target)
    #
    # with tqdm.tqdm(total=len(encoded_sentences)) as pbar:
    #     context_target = np.vectorize(transform_sentence)(encoded_sentences, pbar)
    #     print(context_target.shape)

    X = []
    Y = []
    encoded_sentences = encoded_sentences[:int(len(encoded_sentences) * 0.1)]
    for sentence in tqdm.tqdm(encoded_sentences, desc="Sentences"):
        for i in range(2, len(sentence) - 2):
            X.append(
                [sentence[i + 2], sentence[i + 1],
                 sentence[i - 1], sentence[i - 2]]
                     )
            label_encoding = np.zeros(args.vocab_size)
            label_encoding[sentence[i]] = 1
            Y.append(label_encoding)
            if sentence[i + 2] == vocab_to_index['<end>']:
                break

    print("finished context target transformation")

    X = np.array(X)
    Y = np.array(Y)

    

    cut_off = int(len(X)*0.9)
    # np.random.shuffle(context_target)
    # print("shuffling completed")
    x_train, y_train = X[:cut_off], Y[:cut_off]
    x_val, y_val = X[cut_off + 1:], Y[cut_off+1:]
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


    # make data loaders
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, index_to_vocab

def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    model = CBOW_Model(args.vocab_size)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).float()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze(), labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        labels = labels.argmax(-1)
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(False)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.outputs_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    parser.add_argument("--learning_rate", default=0.05, help="learning rate")

    args = parser.parse_args()
    main(args)
