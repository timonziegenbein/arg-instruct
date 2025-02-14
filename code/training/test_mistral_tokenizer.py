from torch.utils.data import DataLoader
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
from datasets import load_dataset
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
        #project='Fine-tune Gemma-2-9b-it on CA Dataset',
        project='Fine-tune Mistral-7B-Instruct-v0.3 on CA Dataset',
        job_type="training",
        anonymous="allow"
    )

#base_model = "/mnt/home/mstahl/argpaca/models/gemma-2-9b-it"
#new_model = "Gemma-2-9b-it-CA52k"
base_model = f"/mnt/home/mstahl/argpaca/models/Mistral-7B-Instruct-v0.3/"
new_model = "Mistral-7B-Instruct-v0.3-CA52k"

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
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token

# Importing the dataset
dataset_name = "timonziegenbein/ca_all_merged"
dataset_train = load_dataset(dataset_name, split="train")
dataset_train = dataset_train.select(range(2))
dataset_val = load_dataset(dataset_name, split="validation")
dataset_val = dataset_val.select(range(2))

def format_chat_template(row):
    row_json = [{"role": "user", "content": row["instruction"] + "\n" + row["input"]},
                {"role": "assistant", "content": row["output"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
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
        print([x for x in element[dataset_text_field]])
        outputs = processing_class(
            element[dataset_text_field] if formatting_func is None else formatting_func(element),
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        if validation:


            #element[dataset_text_field] = [x.split("<start_of_turn>model\n")[0] if formatting_func is None else formatting_func(x.split("<start_of_turn>model\n")[0]) for x in element[dataset_text_field]]
            element[dataset_text_field] = [x.split("[/INST]")[0] + "[/INST]" if formatting_func is None else formatting_func(x.split("[/INST]")[0]) + "[/INST]" for x in element[dataset_text_field]]
            inputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            # remove all instances that do not contain the sublist [106, 2516, 108]
            ids_to_remove = []
            for i in range(len(outputs["input_ids"])):
                #if ','.join([str(x) for x in [106, 2516, 108]]) not in ','.join([str(x) for x in outputs["input_ids"][i]]):
                if ','.join([str(x) for x in [4]]) not in ','.join([str(x) for x in inputs["input_ids"][i]]):
                    ids_to_remove.append(i)
            outputs["input_ids"] = [x for i, x in enumerate(outputs["input_ids"]) if i not in ids_to_remove]
            outputs["attention_mask"] = [x for i, x in enumerate(outputs["attention_mask"]) if i not in ids_to_remove]
            inputs["input_ids"] = [x for i, x in enumerate(inputs["input_ids"]) if i not in ids_to_remove]
            
            outputs["labels"] = outputs["input_ids"]
            #outputs["input_ids"] = [x[:len(y)+3] for y, x in zip(inputs["input_ids"], outputs["input_ids"])]
            outputs["input_ids"] = [x[:len(y)] for y, x in zip(inputs["input_ids"], outputs["input_ids"])]
            outputs["attention_mask"] = [[1]*len(x) for x in outputs["input_ids"]]
            print({"input_ids": outputs["input_ids"], "labels": outputs["labels"], "attention_mask": outputs["attention_mask"]})
            return {"input_ids": outputs["input_ids"], "labels": outputs["labels"], "attention_mask": outputs["attention_mask"]}
        else:
            print({"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]})
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}


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

    #tokenized_dataset = tokenized_dataset.filter(lambda x: ','.join([str(x) for x in [106, 2516, 108]]) in ','.join([str(x) for x in x["input_ids"]]), num_proc=4)
    tokenized_dataset = tokenized_dataset.filter(lambda x: ','.join([str(x) for x in [4]]) in ','.join([str(x) for x in x["input_ids"]]), num_proc=4)

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

#response_template_with_context = "<start_of_turn>model\n"
response_template_with_context = "[/INST]"
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
#instr_template_with_context = "<start_of_turn>user\n"
instr_template_with_context = "[INST]"
instr_template_ids = tokenizer.encode(instr_template_with_context, add_special_tokens=False)
data_collator_for_training = DataCollatorForCompletionOnlyLM(response_template_ids, instr_template_ids, tokenizer=tokenizer, mlm=False)
data_collator_for_eval = DataCollatorForSeq2Seq(tokenizer, max_length=512, padding="longest", pad_to_multiple_of=8)

data_loader_train = DataLoader(
    dataset_train,
    batch_size=2,
    collate_fn=data_collator_for_training,
)

# print batch from DataLoader
for batch in data_loader_train:
    print(batch)
    break

data_loader_val = DataLoader(
    dataset_val,
    batch_size=2,
    collate_fn=data_collator_for_eval,
)

# print batch from DataLoader
for batch in data_loader_val:
    print(batch)
    break
