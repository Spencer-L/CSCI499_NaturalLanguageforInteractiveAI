
# def encode_data(data, tokenizer:BertTokenizer):
#     x = []
#     y = []
#     breakpoint()
#     for episode in tqdm(data, desc='episodes'):
#         for txt, [act, loc] in episode:
#             txt_emb = tokenizer(txt, return_tensors="pt", padding=True).input_ids
#             x.append(txt_emb)
#             act_emb = tokenizer(act, return_tensors="pt").input_ids
#             loc_emb = tokenizer(loc, return_tensors="pt").input_ids
#             y.append(([act_emb, loc_emb]))
#     x = np.array(x)
#     y = np.array(y)
#     return x, y



# data_files = {
#     "train": "lang_to_sem_data.json"
# }
#
# dataset = load_dataset("json", data_files=data_files)
# breakpoint()
# print(dataset)

# def main(args):
#
#     dataset = load_dataset("SetFit/yelp_review_full")
#
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
#     print(tokenized_datasets)
#
#     # model creation
#     model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#
#     # training
#     training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
#     small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#     small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#     # evaluation
#     metric = evaluate.load("accuracy")
#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         return metric.compute(predictions=predictions, references=labels)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=small_train_dataset,
#         eval_dataset=small_eval_dataset,
#         compute_metrics=compute_metrics,
#     )
#     trainer.train()