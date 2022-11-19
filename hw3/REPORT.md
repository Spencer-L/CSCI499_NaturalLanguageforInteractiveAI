Hyperparameters

Batch Size: I chose a slightly larger batch size of 128 just to help speed training along.

Learning Rate: I determined that 0.01 seemed to be a learning rate that provided good performance without being too slow or start regressing prematurely.

Epochs: I set the epochs to 50 because I noticed that the val accuracy tops out around 20 epochs.  The extra 30 epochs are there just for me to observe the behaviour of the model after that moment.

Embedding Dimensions: I found that 50 dimensions was good so as to provide good performance without unnecessary strain on my machine.

