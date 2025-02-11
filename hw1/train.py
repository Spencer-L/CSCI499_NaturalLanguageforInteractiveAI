import json

import matplotlib.pyplot as plt
import numpy
import numpy as np
import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from model import (Model)
from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    create_train_val_splits,
    encode_data,
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    minibatch_size = 256
    print(f"CUDA version: {torch.version.cuda}")
    with open(args.in_data_fn, "r") as data:
        trainingData = json.loads(data.read())

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
        return train_loader, val_loader, vocab_to_index, len_cutoff, actions_to_index, targets_to_index


def setup_model(device, vocab_to_index, len_cutoff, actions_to_index, targets_to_index, embedding_dim):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = Model(device, len(vocab_to_index), len_cutoff, len(actions_to_index), len(targets_to_index), embedding_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)
        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        # breakpoint()
        action_loss = action_criterion(actions_out.squeeze(), labels[:,0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:,1].long())
        # breakpoint()
        
        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    idx = 1
    val_idx = 1
    idx_arr = []
    val_idx_arr = []
    train_act_loss_arr = []
    train_target_loss_arr = []
    train_act_acc_arr = []
    train_target_acc_arr = []
    val_target_loss_arr=[]
    val_act_loss_arr=[]
    val_target_acc_arr=[]
    val_act_acc_arr=[]
    fig, axs = plt.subplots(8)
    fig.set_figheight(10)
    fig.set_figwidth(8)
    fig.tight_layout()
    fig.suptitle('Training Plots')
    axs[0].title.set_text('Training Target Loss')
    # axs[0].set(xlabel='epoch', ylabel='target_target_loss')
    axs[1].title.set_text('Training Action Loss')
    # axs[1].set(xlabel='epoch', ylabel='target_action_loss')
    axs[2].title.set_text('Training Action Accuracy')
    # axs[2].set(xlabel='epoch', ylabel='target_action_accuracy')
    axs[3].title.set_text('Training Target Accuracy')
    # axs[3].set(xlabel='epoch', ylabel='target_action_accuracy')
    axs[4].title.set_text('Val Target Loss')
    # axs[4].set(xlabel='epoch', ylabel='target_action_accuracy')
    axs[5].title.set_text('Val Action Loss')
    # axs[4].set(xlabel='epoch', ylabel='target_action_accuracy')
    axs[6].title.set_text('Val Target Accuracy')
    # axs[4].set(xlabel='epoch', ylabel='target_action_accuracy')
    axs[7].title.set_text('Val Action Accuracy')
    # axs[4].set(xlabel='epoch', ylabel='target_action_accuracy')
    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
        idx_arr.append(idx)
        train_target_loss_arr.append(train_target_loss)
        train_act_loss_arr.append(train_action_loss)
        train_act_acc_arr.append(train_action_acc)
        train_target_acc_arr.append(train_target_acc)
        axs[0].plot(idx_arr, train_target_loss_arr)
        axs[1].plot(idx_arr, train_act_loss_arr)
        axs[2].plot(idx_arr, train_act_acc_arr)
        axs[3].plot(idx_arr, train_target_acc_arr)
        if epoch % args.val_every == 0:
            val_idx_arr.append(val_idx)
            val_target_loss_arr.append(val_target_loss)
            val_act_loss_arr.append(val_action_loss)
            val_target_acc_arr.append(val_target_acc)
            val_act_acc_arr.append(val_action_acc)
            axs[4].plot(val_idx_arr, val_target_loss_arr)
            axs[5].plot(val_idx_arr, val_act_loss_arr)
            axs[6].plot(val_idx_arr, val_target_acc_arr)
            axs[7].plot(val_idx_arr, val_act_acc_arr)
            val_idx += 1

        fig.show()
        idx += 1

def main(args):
    # Some hyperparameters
    validate_every_n_epochs = 10
    max_epochs = args.num_epochs
    learning_rate = 0.0001
    embedding_dim = args.emb_dim

    device = get_device(True)

    # get dataloaders
    train_loader, val_loader, vocab_to_index, len_cutoff, actions_to_index, targets_to_index = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(device, vocab_to_index, len_cutoff, actions_to_index, targets_to_index, args.emb_dim)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
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
    parser.add_argument("--learning_rate", default=0.0001, help="learning rate")
    parser.add_argument("--voc_k", type=int, help="vocabulary size", required=True)
    parser.add_argument("--emb_dim", type=int, help="embedding dimension", required=True)

    args = parser.parse_args()

    main(args)
