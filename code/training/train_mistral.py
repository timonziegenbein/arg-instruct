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
#dataset_train = dataset_train.select(range(100))
dataset_val = load_dataset(dataset_name, split="validation")
#dataset_val = dataset_val.select(range(100))

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
            print(ids_to_remove)
            outputs["input_ids"] = [x for i, x in enumerate(outputs["input_ids"]) if i not in ids_to_remove]
            outputs["attention_mask"] = [x for i, x in enumerate(outputs["attention_mask"]) if i not in ids_to_remove]
            inputs["input_ids"] = [x for i, x in enumerate(inputs["input_ids"]) if i not in ids_to_remove]
            
            outputs["labels"] = outputs["input_ids"]
            #outputs["input_ids"] = [x[:len(y)+3] for y, x in zip(inputs["input_ids"], outputs["input_ids"])]
            outputs["input_ids"] = [x[:len(y)+1] for y, x in zip(inputs["input_ids"], outputs["input_ids"])]
            outputs["attention_mask"] = [[1]*len(x) for x in outputs["input_ids"]]
            return {"input_ids": outputs["input_ids"], "labels": outputs["labels"], "attention_mask": outputs["attention_mask"]}
        else:
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
data_collator_for_eval = DataCollatorForSeq2Seq(tokenizer, max_length=512, padding="max_length", pad_to_multiple_of=8)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
    # remove all <unk> tokens
    decoded_preds = [x.replace("<unk>", "").strip() for x in decoded_preds]
    #print(decoded_preds[0:10])
    #decoded_preds_split = [x.split("\nmodel\n")[1].strip() if "\nmodel\n" in x else None for x in decoded_preds]
    decoded_preds_split = [x.split("[/INST]")[1].strip() if "[/INST]" in x else None for x in decoded_preds]
    #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    decoded_labels = [x.replace("<unk>", "").strip() for x in decoded_labels]
    #decoded_labels = [x.split("\nmodel\n")[1].strip() for x in decoded_labels]
    #print(decoded_labels[0:10])
    decoded_labels = [x.split("[/INST]")[1].strip() if "[/INST]" in x else None for x in decoded_labels]

    for decoded_label in decoded_labels:
        if decoded_label is None:
            print(decoded_label)

    for i in range(10):
        print('Before:')
        print(decoded_preds[i])
        print('After:')
        print(decoded_preds_split[i])
        print('Label:')
        print(decoded_labels[i])
        print('-'*50)
    # remove None preds in decoded_labels and corresponding preds
    decoded_labels = [decoded_labels[i] for i in range(len(decoded_labels)) if decoded_preds_split[i] is not None]
    decoded_preds_split = [x for x in decoded_preds_split if x is not None]
    #print(decoded_labels[0:10])
    default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = 0
    for i in range(len(decoded_preds_split)):
        rougeL += default_rouge_scorer.score(decoded_labels[i], decoded_preds_split[i])['rougeL'].fmeasure
    if len(decoded_preds_split) == 0:
        return {"rougeL": 0}
    rougeL /= len(decoded_preds_split)
    print(rougeL)
    return {"rougeL": rougeL}

generation_config = GenerationConfig.from_pretrained(base_model)
generation_config.max_new_tokens = 64
# Hyperparamter
training_arguments = Seq2SeqTrainingArguments(
    #output_dir='/mnt/home/mstahl/argpaca/models/gemma-2-9b-it-ca_gen_merged',
    output_dir='/mnt/home/mstahl/argpaca/models/Mistral-7B-Instruct-v0.3-CA52k',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=30,
    optim="paged_adamw_32bit",
    num_train_epochs=10,
    eval_strategy="steps",
    eval_steps=0.2,
    save_strategy="steps",
    save_steps=0.2,
    do_eval=False,
    logging_steps=1,
    warmup_ratio=0.05,
    logging_strategy="steps",
    learning_rate=5.3547702316584515e-05,
    fp16=False,
    bf16=True,
    group_by_length=True,
    report_to="wandb",
    #use_liger_kernel=True,
    eval_on_start=True,
    predict_with_generate=True,
    generation_config=generation_config,
    generation_max_length=576,
)

class Seq2SeqTrainerWithTwoDataCollators(Seq2SeqTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator_for_training=None,
        train_dataset=None,
        data_collator_for_eval=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator_for_training,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Override self.model.generation_config if a GenerationConfig is specified in args.
        # Priority: args.generation_config > model.generation_config > default GenerationConfig.
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
        
        self.data_collator_for_eval = data_collator_for_eval

    def get_eval_dataloader(self, eval_dataset = None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator_for_eval if self.data_collator_for_eval is not None else self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

# Setting sft parameters
trainer = Seq2SeqTrainerWithTwoDataCollators(
    model_init=model_init,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    #peft_config=peft_config,
    #max_length=1024,
    #dataset_text_field="text",
    processing_class=tokenizer,
    args=training_arguments,
    #packing=False,
    compute_metrics=compute_metrics,
    data_collator_for_training=data_collator_for_training,
    data_collator_for_eval=data_collator_for_eval,
)

trainer.train()
trainer.evaluate()

# Save the fine-tuned model
wandb.finish()

## Save the fine-tuned model
trainer.save_model()

