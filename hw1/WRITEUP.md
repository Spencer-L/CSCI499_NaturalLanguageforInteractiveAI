Run command: python train.py --in_data_fn lang_to_sem_data.json --voc_k 1000 --emb_dim 128

My model uses an embedding layer, a maxpool layer, an LSTM layer, then a linear layer.
Aside from the default values, I used a vocabulary count of 1000 and 
an embedding dimension of 128. This yielded me a train action accuracy
of around 48% and a train target accuracy of around 9.5%. Train action loss was around 521 and train target loss was 1499.
I believe that the large discrepancy between target and action 
might be explained by the fact that there are only 8 actions whereas
there are 80 total targets.  There is a fundamental flaw in my setup
in that I encoded my data to use an array of 0's and 1's to represent
the target, action pair. The first 8 numbers represent actions and the
last 80 represent targets and there is a 1 in each group to represent
a target action pair.  I believe this set up fails to build a
context-based correlation between actions and targets which means
the model, although learning initially, tapers off at the 10% accuracy
mark.  I attached a printout of my training results.