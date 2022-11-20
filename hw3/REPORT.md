Commands

Seq2Seq Encoder Decoder (BASE): python train.py --batch_size 128 --learning_rate 0.01 --emb_dim 50 --model_output_dir model_output --in_data_fn lang_to_sem_data.json

HuggingFace DistilBert (TRANSFORMER): python huggingtrain.py



Hyperparameters

Batch Size: I chose a slightly larger batch size of 128 just to help speed training along.

Learning Rate: I determined that 0.01 seemed to be a learning rate that provided good performance without being too slow or start regressing prematurely.

Epochs: I set the epochs to 200 because I noticed that the val accuracy tops out around 140 epochs.  The extra epochs are there just for me to observe the behaviour of the model after that moment.

Val_Every: I run an accuracy check every 5 epochs by default to give the model ample time to train before evaluation.

Embedding Dimensions: It seemed that other papers used embedding dimensions around 50, so I went with that.

Vocab Size: I left the default as 1000 since it seemed to work pretty well from last time.



Seq2Seq Encoder Decoder (BASE) Performance

The base model's performance without attention mechanism showed a maximum accuracy of around 67%.  The model usually peaks around 140 epochs before growth either tapers off or the accuracy gradually diminishes.  I observed that train loss is around 0.66 at the 140 epoch mark whereas val loss is around 1.52.  After that point, train loss gradually decreases and val loss gradually increases, which makes sense seeing as the model begins overfitting the data and thus explains the gradual diminishing of the accuracy rating as well.  



Comparison to DistilBert (HUGGINGFACE)

I implemented a transformer using a pretrained DistilBert model from HuggingFace.  This model was then finetuned on the provided data from this assignment.  After spending a substantial amount of time learning HuggingFace library and debugging, I was able to get the model to run on these hyperparameters:

num_train_epochs=5 (Honestly, I don't have the means to even get 2 epochs since one epoch takes 9 hours),              
per_device_train_batch_size=64 (largest I can fit so as to decrease training time as much as possible),  
per_device_eval_batch_size=64 (see above),   
warmup_steps=500 (default from HuggingFace),                
weight_decay=0.01 (default from HuggingFace),         
logging_steps=10 (prints progress every 10 steps),
evaluation_strategy="epoch" (prints an accuracy evaluation at every epoch)

As of the time of writing, the model is currently training.  Due to computational constraints (I'm running this large model on a GTX 1050 Ti Max-Q mobile gpu), the model takes 9 hours to train 5 epochs.  I lack the necessary VRAM (I only have 4 Gb) to load this model with a substantial batch size above 64.  Until then, I cannot ascertain the performance of my HuggingFace implementation.  However, I can probably extrapolate that this implementation will have much better results than my seq2seq without attention since 67% isn't a particularly high accuracy for this specific task.  A pretrained LLM with finetuning should easily outgun my seq2seq.

*edit: I ran the model through a few epochs (pdf in repo) and observed that the accuracy is as high as around 79% with near-zero loss.  This confirms my speculations that the Transformer implementation will have far better performance than the base implementation.


Seq2Seq Encoder Decoder + Attention Mechanism

I was unable to get to this portion of the assignment.  I spent a good week straight just working on this assignment, but I found that it was challenging to implement the base implementation and, on top of that, to learn HuggingFace in the same span of time.  Nonetheless, my intuition would be that the HuggingFace implementation will be much better than the other two implementations due to the reasons above.  The performance of the attention mechanism implementation will also be better than the base implementation because the attention mechanism would focus on the most important sections of the input text whilst reducing the noise.