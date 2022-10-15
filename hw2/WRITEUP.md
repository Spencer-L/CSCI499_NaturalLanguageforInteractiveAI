The model I chose to implement is CBOW.
I used an embedding layer, then took the mean,
and then used a linear regression layer. This is
similar to the original word2vec implementation.
A printout of the results using the values down
below is attached in this repo.

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
84.05% and loss of 0.9522 vs a validation accuracy 
of 82.13% and loss of 1.0699. This is pretty good
considering that word2vec achieved around 92% 
accuracy using a much larger corpus (I'm only using
10% of the books corpus we have due to hardware
limitations) as well as being able to use a bigger
vocab size.

In Vivo Analysis:

Due to the lack of vocab size I could use,
my in vivo results were less than stellar compared to
the professor's.  However, I did notice that I received
less inf values as I slowly optimized my hyperparameters.
Slowly increasing my vocab size seemed to have the most
impact in that regard.  I also noticed that I received
a ton of warnings (I commented them out) since my limited
vocab size was unable to account for many keywords.

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
loss versus the val accuracy and loss. The metrics 
used are whether or not the model is producing the 
correct prediction given a context of four adjacent
words. During in vivo evaluations, the code runs the 
downstream_validation function. This function 
evaluates our model against the analogies json file 
which is an analogical reasoning task. The metrics
used for this evaluation are listed in the printed 
output. They include our model's accuracy in a 
couple of different scenarios such as synonyms,
antonyms, similar, derivedfrom, instanceof, etc.
This would let us know the robustness of our
model in handling tasks outside of a lab environment.
We are making the assumption that the books corpus
is representative of language usage in the real world.
However, the books in the corpus are in fact perhaps 
hundreds of years old or use quite a bit of
non-colloquial English. This might train the model
to be more fit towards old English rather than 
modern English, creating a model less suitable
for downstream tasks.