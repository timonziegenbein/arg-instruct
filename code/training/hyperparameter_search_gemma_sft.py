import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, TrainingArguments, GenerationConfig, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import datasets
import pandas as pd
import gc
import wandb
import wandb_osh
from trl import SFTTrainer
from trl.trainer.utils import peft_module_casting_to_bf16
from accelerate import PartialState
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import random

import sys
sys.path.append('../arg_eval')
from rouge import rouge_scorer

sys.path.append('..')
from tasks.argument_annotated_essays_2 import ClassifyingArgumentComponentsArgumentAnnotatedEssays2
from tasks.argument_annotated_essays_2 import IdentifyingArgumentComponentsArgumentAnnotatedEssays2
from tasks.argument_annotated_essays_2 import IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2
from tasks.argument_annotated_essays_2 import StanceRecognitionArgumentAnnotatedEssays2
from tasks.qt30 import IllocutionaryRelationsIdentificationQT30
from tasks.qt30 import PropositionalRelationsIdentificationQT30
from tasks.f1000rd import PragmaticTaggingF1000rd
from tasks.iac_2 import PredictAgreementIacV2
from tasks.iac_2 import PredictFactualityIacV2
from tasks.iac_2 import PredictNiceIacV2
from tasks.iac_2 import PredictRespectIacV2
from tasks.iac_2 import PredictSarcasmIacV2
from tasks.ibm_rank_30k import QualityAssessmentIbmRank30k
from tasks.ibm_rank_30k import StancePredictionIbmRank30k
from tasks.arguana_counterargs_corpus import SameDebateArgumentsArguanaCounterargsCorpus
from tasks.arguana_counterargs_corpus import SameDebateCountersArguanaCounterargsCorpus
from tasks.arguana_counterargs_corpus import SameDebateOpposingArgumentsArguanaCounterargsCorpus
from tasks.arguana_counterargs_corpus import SameDebateOpposingCountersArguanaCounterargsCorpus
from tasks.aspect_controlled_argument_generation import AspectControlledArgumentGenerationAspectControlledArgumentGeneration
from tasks.debate_sum import ExtractiveSummarizationDebateSum
from tasks.webis_conclugen_21 import ConclusionGenerationWebisConclugen21
test_tasks = [
    ClassifyingArgumentComponentsArgumentAnnotatedEssays2,
    IdentifyingArgumentComponentsArgumentAnnotatedEssays2,
    IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2,
    StanceRecognitionArgumentAnnotatedEssays2,
    IllocutionaryRelationsIdentificationQT30,
    PropositionalRelationsIdentificationQT30,
    PragmaticTaggingF1000rd,
    PredictAgreementIacV2,
    PredictFactualityIacV2,
    PredictNiceIacV2,
    PredictRespectIacV2,
    PredictSarcasmIacV2,
    QualityAssessmentIbmRank30k,
    StancePredictionIbmRank30k,
    SameDebateArgumentsArguanaCounterargsCorpus,
    SameDebateCountersArguanaCounterargsCorpus,
    SameDebateOpposingArgumentsArguanaCounterargsCorpus,
    SameDebateOpposingCountersArguanaCounterargsCorpus,
    AspectControlledArgumentGenerationAspectControlledArgumentGeneration,
    ExtractiveSummarizationDebateSum,
    ConclusionGenerationWebisConclugen21,
] 

device_string = PartialState().process_index

if device_string == "0":
    wandb_osh.set_log_level("ERROR")
    wandb.login()
    run = wandb.init(
        project='Fine-tune Gemma-2-9b-it on CA Dataset',
        job_type="training",
        anonymous="allow"
    )

model_id = "/mnt/home/mstahl/argpaca/models/gemma-2-9b"
new_model = "Gemma-2-9b-seed"

tokenizer = AutoTokenizer.from_pretrained(model_id)

def model_init():
    gc.collect()
    torch.cuda.empty_cache()
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        #target_modules=modules
    )

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
        model_id,
        #quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        device_map={'':device_string}
    ).to(torch_dtype)

    model = get_peft_model(model, peft_config)
    peft_module_casting_to_bf16(model)
    return model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare data
task = ExtractiveSummarizationDebateSum()
if len([i for i in task.instances if i.split == 'dev']) > 0:
    train_instances = [{
            'instruction': i.task_instruction,
            'input': i.input,
            'output': i.output
        } for i in task.instances if i.split == 'train']
    valid_instances = [{
            'instruction': i.task_instruction,
            'input': i.input,
            'output': i.output
        } for i in task.instances if i.split == 'dev']
    random.shuffle(train_instances)
    random.shuffle(valid_instances)
else:
    instances = [{
        'instruction': i.task_instruction,
        'input': i.input,
        'output': i.output
    } for i in task.instances if i.split == 'train']
    random.shuffle(instances)
    train_instances, valid_instances = instances[:round(len(instances)*0.875)], instances[round(len(instances)*0.875):]
print(f"{len(train_instances)} train instances, {len(valid_instances)} validation instances")

train_df = pd.DataFrame.from_dict(train_instances)
valid_df = pd.DataFrame.from_dict(valid_instances)
dataset_train = datasets.Dataset.from_pandas(train_df)
dataset_val = datasets.Dataset.from_pandas(valid_df) 

def tokenize(examples):
    tokenized_inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512)
    tokenized_inputs['labels'] = tokenizer(examples['output'])['input_ids']
    return tokenized_inputs    
    
map_kwargs = {
    "batched": True,
    "batch_size": 1000,
}
tokenized_dataset_train = dataset_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = dataset_val.map(tokenize, **map_kwargs)

print(tokenized_dataset_train[0])

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds_split = [x.split("\nmodel\n")[1].strip() if "\nmodel\n" in x else None for x in decoded_preds]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [x.split("\nmodel\n")[1].strip() for x in decoded_labels]
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
    print(decoded_labels[0:10])
    default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = 0
    for i in range(len(decoded_preds_split)):
        rougeL += default_rouge_scorer.score(decoded_labels[i], decoded_preds_split[i])['rougeL'].fmeasure
    if len(decoded_preds_split) == 0:
        return {"rougeL": 0}
    rougeL /= len(decoded_preds_split)
    print(rougeL)
    return {"rougeL": rougeL}

data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=512, padding="max_length", pad_to_multiple_of=8)

generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 64

training_arguments = Seq2SeqTrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_ratio=0.05,
    logging_strategy="steps",
    learning_rate=2e-4,
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

trainer = Seq2SeqTrainer(
    model_init=model_init,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_val,
    args=training_arguments,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)
trainer.train()


def hp_space(trial):
    # Define hyperparameters to be tuned
    return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 32, step=1),
            #"warmup_steps": trial.suggest_int("warmup_steps", 0, 100, step=10),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 10),
        }

def run_hp_search_optuna(trainer, n_trials, direction, **kwargs):
    import optuna

    def _objective(trial):
        trainer.objective = None
        trainer.train(trial=trial)

        # memory management
        del trainer.model
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate if needed
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)

        return trainer.objective

trainer.run_hp_search_optuna = run_hp_search_optuna

def my_objective(metrics):
    return metrics["eval_rougeL"]

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=20,
    storage='sqlite:///gemma_optuna.db',
    compute_objective=my_objective,
    load_if_exists=True,
    study_name="gemma-seed",
)

print(best_trial)