#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import sacrebleu
from datasets import load_dataset
from transformers import M2M100Tokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from evaluate import load

def postprocess_text(preds, labels):
    """
    Post-processes the text output from the model and the labels. Strips the predictions and labels from leading/trailing white spaces.

    Args:
        preds (list): List of predicted sequences.
        labels (list): List of true label sequences.

    Returns:
        tuple: Tuple containing lists of processed predictions and labels.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    """
    Calculates BLEU score and generation length for the model predictions.

    Args:
        eval_preds (tuple): Tuple containing predictions and labels.

    Returns:
        dict: Dictionary containing BLEU score and generation length.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def preprocess_data(examples):
    """
    Pre-processes input examples for the model. Tokenizes the source and target text from the examples.

    Args:
        examples (dict): Dictionary containing input examples.

    Returns:
        dict: Dictionary containing tokenized model inputs.
    """
    src_text = [example[source_lang] for example in examples["translation_xml"]]
    tgt_text = [example[target_lang] for example in examples["translation_xml"]]
    model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt", padding="max_length", truncation=True)
    return model_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train M2M100 model.')
    parser.add_argument('--config', type=str, default="config.json", help='Path to configuration JSON file.')
    args = parser.parse_args()

    # Read configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    source_lang = config['src_lang']
    target_lang = config['tgt_lang']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_train_epochs = config['num_train_epochs']
    model_name = f"{config['model_space']}/{config['model_name']}"
    dataset_name = config['dataset']

    sacrebleu = load("sacrebleu")

    dataset = load_dataset(dataset_name, f"{source_lang}-{target_lang}")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, src_lang=source_lang, tgt_lang=target_lang)
    tokenized_dataset = dataset.map(preprocess_data, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{dataset_name}-{model_name}-{source_lang}-{target_lang}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=weight_decay,
        save_total_limit=5,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        save_strategy="epoch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()