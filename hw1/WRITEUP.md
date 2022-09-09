Run command: python train.py --in_data_fn lang_to_sem_data.json --voc_k 1000 --emb_dim 128

My model uses an embedding layer, a maxpool layer, an LSTM layer, then a linear layer.
Aside from the default values, I used a vocabulary count of 1000 and 
an embedding dimension of 128. This yielded me a train action accuracy
of around 48% and a train target accuracy of around 9.5%