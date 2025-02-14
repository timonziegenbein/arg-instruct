from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from functools import partial
import warnings
import numpy as np
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os
import torch
import wandb
import wandb_osh
import datasets
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, setup_chat_format, SFTConfig, DataCollatorForCompletionOnlyLM
from trl.trainer.utils import peft_module_casting_to_bf16
from accelerate import PartialState
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers.data.data_collator import *

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining
from ray.air.integrations.wandb import WandbLoggerCallback

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers.utils.import_utils import is_datasets_available
import sys
sys.path.insert(0, '../arg_eval')
from rouge import rouge_scorer


device_string = PartialState().process_index

if device_string == "0":
    wandb_osh.set_log_level("ERROR")
    wandb.login()
    run = wandb.init(
        project='Fine-tune Gemma-2-9b-it on CA Dataset',
        job_type="training",
        anonymous="allow"
    )

base_model = "/mnt/home/mstahl/argpaca/models/gemma-2-9b"
tokenizer_model = "/mnt/home/mstahl/argpaca/models/gemma-2-9b"
new_model = "Gemma-2-9b-CA52k"

# LoRA config
peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        #target_modules=modules
)

def model_init():
    gc.collect()
    torch.cuda.empty_cache()

    
    # Set torch dtype and attention implementation
    torch_dtype = torch.bfloat16
    attn_implementation = "eager"
    
    # QLoRA config
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch_dtype,
    #    bnb_4bit_use_double_quant=True,
    #)
    
    # Load model
    model = AutoLigerKernelForCausalLM.from_pretrained(
        base_model,
        #quantization_config=bnb_config,
        #device_map="auto",
        attn_implementation=attn_implementation,
        device_map={'':device_string}
    ).to(torch_dtype)
    
    
    
    #def find_all_linear_names(model):
    #    cls = bnb.nn.Linear4bit
    #    lora_module_names = set()
    #    for name, module in model.named_modules():
    #        if isinstance(module, cls):
    #            names = name.split('.')
    #            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    #    if 'lm_head' in lora_module_names:  # needed for 16 bit
    #        lora_module_names.remove('lm_head')
    #    return list(lora_module_names)
    #
    #
    #modules = find_all_linear_names(model)
    
    #model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)
    peft_module_casting_to_bf16(model)
    return model


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

# Importing the dataset
dataset_name_train = "timonziegenbein/ca52k"
dataset_train = load_dataset(dataset_name_train, split="train")
#jdataset_train = dataset_train.select(range(200))
dataset_val_1 = load_dataset("timonziegenbein/ca52k_train", split="validation")
dataset_val_1 = dataset_val_1.add_column("source", ["ca52k_train"]*len(dataset_val_1))
dataset_val_2 = load_dataset("timonziegenbein/ca52k_eval", split="validation")
dataset_val_2 = dataset_val_2.add_column("source", ["ca52k_eval"]*len(dataset_val_2))
dataset_val = concatenate_datasets([dataset_val_1, dataset_val_2])
#dataset_val = dataset_val.select(range(200))

def format_chat_template(row):
    #row_json = [{"role": "user", "content": row["instruction"] + "\n" + row["input"]},
    #            {"role": "assistant", "content": row["output"]}]
    #row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    instruct_pre = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''
    instruct_pattern = '''### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}'''
    row["text"] = instruct_pre + instruct_pattern.format(row["instruction"], row["input"], row["output"])
    return row


dataset_train = dataset_train.map(
    format_chat_template,
    num_proc=4,
)
dataset_val = dataset_val.map(
    format_chat_template,
    num_proc=4,
)

def _prepare_non_packed_dataloader(
    processing_class,
    dataset,
    dataset_text_field: str,
    max_seq_length,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
    validation=False,
):
    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        if not validation:
            outputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
        else:
            element[dataset_text_field] = [x + '[SOURCE]' + y if formatting_func is None else formatting_func(x) + '[SOURCE]' + element["source"] for x, y in zip(element[dataset_text_field], element["source"])]
            outputs = processing_class(
                element[dataset_text_field],
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                #max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            element[dataset_text_field] = [x.split("\n\n### Response:\n")[0] + "\n\n### Response:\n" if formatting_func is None else formatting_func(x.split("\n\n### Response:\n")[0]) + "\n\n### Response:\n" for x, y in zip(element[dataset_text_field], element["source"])]
            inputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                #max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            # remove all instances that do not contain the sublist [106, 2516, 108]
            #ids_to_remove = []
            #for i in range(len(outputs["input_ids"])):
            #    if ','.join([str(x) for x in [106, 2516, 108]]) not in ','.join([str(x) for x in outputs["input_ids"][i]]):
            #        ids_to_remove.append(i)
            #print(ids_to_remove)
            #outputs["input_ids"] = [x for i, x in enumerate(outputs["input_ids"]) if i not in ids_to_remove]
            #outputs["attention_mask"] = [x for i, x in enumerate(outputs["attention_mask"]) if i not in ids_to_remove]
            #inputs["input_ids"] = [x for i, x in enumerate(inputs["input_ids"]) if i not in ids_to_remove]
            
            outputs["labels"] = outputs["input_ids"]
            outputs["input_ids"] = [x[:len(y)] for y, x in zip(inputs["input_ids"], outputs["input_ids"])]
            outputs["attention_mask"] = [[1]*len(x) for x in outputs["input_ids"]]
            return {"input_ids": outputs["input_ids"], "labels": outputs["labels"], "attention_mask": outputs["attention_mask"]}

    signature_columns = ["input_ids", "labels", "attention_mask"]

    if dataset.column_names is not None:  # None for IterableDataset
        extra_columns = list(set(dataset.column_names) - set(signature_columns))
    else:
        extra_columns = []

    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names if remove_unused_columns else None,
        "batch_size": 1000,
    }
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    # filter out all instances where [106, 2516, 108] is not in the input_ids or input_ids is longer than max_seq_length
    tokenized_dataset = tokenized_dataset.filter(lambda x: ','.join([str(x) for x in [109, 6176, 10567, 235292, 108]]) in ','.join([str(x) for x in x["input_ids"]]) and len(x["input_ids"]) <= max_seq_length, num_proc=4)

    # filter out all instances that are longer than max_seq_length
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length, num_proc=4)
    # filter out all instances that have labels longer than max_seq_length
    if validation:
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["labels"]) <= max_seq_length, num_proc=4)

    return tokenized_dataset


dataset_train = _prepare_non_packed_dataloader(
    tokenizer,
    dataset_train,
    "text",
    512,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
)

dataset_val = _prepare_non_packed_dataloader(
    tokenizer,
    dataset_val,
    "text",
    512,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
    validation=True,
)

response_template_with_context = "\n\n### Response:\n"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
print(response_template_ids)
#instr_template_with_context = "\n\n### Response:\n"
#instr_template_ids = tokenizer.encode(instr_template_with_context, add_special_tokens=False)
data_collator_for_training = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)
data_collator_for_eval = DataCollatorForSeq2Seq(tokenizer, max_length=512, padding="max_length", pad_to_multiple_of=8)

data_loader_train = DataLoader(
    dataset_train,
    batch_size=2,
    collate_fn=data_collator_for_training,
)

# print batch from DataLoader
for batch in data_loader_train:
    print(batch["input_ids"][0])
    print(batch["labels"][0])
    print(batch["attention_mask"][0])
    print(tokenizer.decode(batch["input_ids"][0]))
    break

data_loader_val = DataLoader(
    dataset_val,
    batch_size=2,
    collate_fn=data_collator_for_eval,
)

# print batch from DataLoader
for batch in data_loader_val:
    print(batch["input_ids"][0])
    print(batch["labels"][0])
    print(batch["attention_mask"][0])
    print(tokenizer.decode(batch["input_ids"][0]))
    break

