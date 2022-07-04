import transformers
import csv
import json
import os


from datasets import load_dataset, load_metric
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from __future__ import absolute_import, division, print_function

model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"




_URLs = {    # 本地文件的路径
    'train': "/home/zhangcw/zjchen/translation2019zh/translation2019zh_train.json",
    'dev': "/home/zhangcw/zjchen/translation2019zh/translation2019zh_valid.json"  
}


dataset = load_dataset('json', data_files=_URLs['train'])
validation = load_dataset('json', data_files=_URLs['dev'])
metric = load_metric("sacrebleu")


    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = ""
max_input_length = 128
max_target_length = 128
# source_lang = "english"
# target_lang = "chinese"
def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples["english"]]
    targets = [ex for ex in examples["chinese"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(dataset['train'][:2])
tokenized_datasets = dataset.map(preprocess_function, batched=True)
preprocess_function(validation['train'][:2])
tokenized_datasets_eval = dataset.map(preprocess_function, batched=True)


tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{'english'}-to-{'chinese'}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True    
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets_eval["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

for dirname, _, filenames in os.walk('opus-mt-en-zh-finetuned-en-to-zh'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# from transformers import MarianMTModel, MarianTokenizer
src_text = ['My name is Sarah and I live in London']

model_name = 'opus-mt-en-zh-finetuned-en-to-zh/checkpoint-38000'
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
[tokenizer.decode(t, skip_special_tokens=True) for t in translated]