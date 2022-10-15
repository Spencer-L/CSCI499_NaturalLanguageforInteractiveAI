The model I chose to implement is CBOW.
I used an embedding layer, then took the mean,
and then used a linear regression layer. This is
similar to the original word2vec implementation.

Some constraints I faced when implementing
this model is that I would frequently run
out of memory whilst training on a 32gb
DDR4 system. Some measures that I took to
compensate for this shortcoming is to 
use just 10% of the dataset 
as well as having a low vocab size.

The hyperparameters chosen were:

--vocab_size 1200 This was the maximum
vocab size that I could fit on my RAM.
I believe a larger vocab size would do
my model a lot better as 1200 words
is certainly very low considering that
there is 10 percent of 30 
(pretty large) books in the dataset.

--num_epochs 30 (same as demoed in class)

--batch_size 256 I chose a higher batch
size to make training a little faster.
Each epoch averaged around 1-2 minutes
to complete.

--learning_rate 0.05 I had initially 
used a learning rate of 0.0001 but 
noticed that the model was not learning.
I incrementally increased my learning 
rate to 0.05, allowing my model to
learn much faster in the given
amount of epochs without overfitting.

emb_dim 10 I chose a smaller embedding
dimension because our vocab size is small
and to help speed up training time.

emb_max_norm 1 This would prevent 
exploding gradients.

In Vitro Analysis:

Using the above hyperparameters, my model
was able to achieve a train accuracy of
vs a
validation accuracy of

In Vivo Analysis:

Released Code Analysis:
The released code is first encoding our 
dataset which we are then creating a 
train/val split with one hot encoding.
After that, we make our model and begin
training. The default configuration trains
the model for 30 epochs and prints out in vivo 
and in vitro evaluation every 5 epochs. For the
sake of faster debugging, I set the val_every 
argument to 2 epochs. During in vitro evals,
the code prints and compares the train accuracy and 
loss versus the val accuracy and loss. During in vivo
evaluations, the code runs the downstream_validation
function which compares 