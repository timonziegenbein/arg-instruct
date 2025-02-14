import json
import pandas as pd
import torch
import numpy as np
# from peft import PeftModel, PeftConfig
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines
from outlines.models.openai import OpenAIConfig
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../tasks')

from samplers import InstanceSampler, BalancedInstanceSampler, SuperNiInstanceSampler

base_path = os.environ['ARGPACA_MAJA']
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

GEN_ARGS = {
    "do_sample": False,
    "top_p": 1.0,
    "top_k": 1,
    "temperature": 0.0,
    "num_return_sequences": 1,
}



class AutoRegressivePredictor:
    def __init__(self, model_name, gen_args, peft_model_name=None):
        # if peft_model_name is not None:
        # peft_config = PeftConfig.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        # self.model = PeftModel.from_pretrained(model, model_name).to("cuda")
        # else:
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.gen_args = gen_args

    def write_to_file(self, file_path, sample, predictions):
        with open(file_path, 'a') as f:
            prediction = predictions[0]
            out_dict = {'id': sample.id, 'input': sample.input, 'prediction': prediction, 'reference': sample.output}
            json.dump(out_dict, f)
            f.write('\n')

    def predict_ds(self, ds, to_file=False, file_path=None, overwrite=False):
        if to_file and not overwrite:
            if os.path.exists(file_path):
                temp_df = pd.read_json(file_path, lines=True)
                exisiting_ids = set(temp_df['id'].tolist())
            ds = ds[[sample.id not in exisiting_ids for sample in ds]]
        if to_file and overwrite:
            with open(file_path, 'w') as f:
                f.write("")
        for sample in tqdm(ds, total=len(ds)):
            sample_predictions = self.predict_sample(sample.input)
            if to_file:
                self.write_to_file(file_path, sample, sample_predictions)

    def predict_batch(self, sample):
        with torch.no_grad():
            input_ids = self.tokenizer(sample, return_tensors="pt").input_ids.to("cuda")
            output = self.model.generate(
                input_ids,
                **self.gen_args,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def predict_sample(self, sample):
        with torch.no_grad():
            prompt = sample
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            self.gen_args["max_new_tokens"] = 512
            self.gen_args["eos_token_id"] = self.tokenizer.eos_token_id
            self.gen_args["pad_token_id"] = self.tokenizer.pad_token_id
            outputs = self.model.generate(
                prompt_input_ids,
                **self.gen_args,
            )
            decoded_outputs = []
            for output in outputs:
                decoded_outputs.append(self.tokenizer.decode(
                    output[len(prompt_input_ids[0]):], skip_special_tokens=True).strip())
            return decoded_outputs


class OutlinePredictor():
    def __init__(self, model_name, max_tokens=None, stop_at=None):
        self.model_name = model_name
        if self.model_name =='gpt-4o-mini':
            config = OpenAIConfig(
                presence_penalty=0,
                frequency_penalty=0,
                top_p=1,
                temperature=0.0,
                seed=42,
            )
            self.model = outlines.models.openai("gpt-4o-mini", config, api_key='#APIKEY')
            self.text_generator = outlines.generate.text(self.model)
        else:
            self.model = outlines.models.transformers(model_name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.choice_generator = None
            self.sampler = outlines.samplers.greedy()
            self.float_generator = outlines.generate.format(self.model, float, self.sampler)
            self.text_generator = outlines.generate.text(self.model, self.sampler)
            self.max_tokens = max_tokens
            self.stop_at = stop_at
        self.current_choice_generator_classes = None


    def write_to_file(self, file_path, sample, predictions):
        with open(file_path, 'a') as f:
            prediction = predictions[0]
            out_dict = {
                    'approach': self.model_name,
                    'id': sample.id,
                    'is_clf': sample.is_clf,
                    'is_reg': sample.is_reg,
                    'input': sample.input,
                    'prediction': prediction,
                    'reference': sample.output,
                    }
            json.dump(out_dict, f)
            f.write('\n')


    def predict_ds(self, ds, to_file=False, file_path=None, overwrite=False):
        if to_file:
            if os.path.exists(file_path) and not overwrite:
                temp_df = pd.read_json(file_path, lines=True)
                exisiting_ids = set(temp_df['id'].tolist())
                print(f'Loaded {len(exisiting_ids)} existing predictions')
                ds = [sample for sample in ds if sample.id not in exisiting_ids]
                print(f'Predicting {len(ds)} new samples')
            else:
                with open(file_path, 'w') as f:
                    f.write("")
        for sample in tqdm(ds, total=len(ds)):
            sample_predictions = self.predict_sample(sample)
            if to_file:
                self.write_to_file(file_path, sample, sample_predictions)

    def predict_sample(self, sample):
        if self.model_name == 'gpt-4o-mini':
            if sample.is_clf or sample.is_reg:
                if self.current_choice_generator_classes != sample.classes:
                    self.choice_generator = outlines.generate.choice(self.model, sample.classes)
                    self.current_choice_generator_classes = sample.classes
                pred = self.choice_generator(sample.input)
            else:
                pred = self.text_generator(sample.input)
        else:
            if sample.is_clf:
                print('clf')
                print(sample.classes)
                if self.current_choice_generator_classes != sample.classes:
                    self.choice_generator = outlines.generate.choice(self.model, sample.classes, self.sampler)
                    self.current_choice_generator_classes = sample.classes
                pred = self.choice_generator(sample.input)
            elif sample.is_reg:
                print('reg')
                pred = self.float_generator(sample.input, max_tokens=10)
            else:
                print('gen')
                pred = self.text_generator(sample.input, stop_at=self.stop_at, max_tokens=int(len(sample.output)*1.25)/4)
        return [pred]


class RandomPredictor(AutoRegressivePredictor):
    def __init__(self, model_name, max_tokens=None, stop_at=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.stop_at = stop_at
        self.num_written = 0

    def write_to_file(self, file_path, sample, predictions):
        with open(file_path, 'a') as f:
            prediction = predictions[0]
            out_dict = {
                    'approach': self.model_name,
                    'id': sample.id,
                    'is_clf': sample.is_clf,
                    'is_reg': sample.is_reg,
                    'input': sample.input,
                    'prediction': prediction,
                    'reference': sample.output,
                    }
            json.dump(out_dict, f)
            f.write('\n')

    def predict_ds(self, ds, to_file=False, file_path=None, overwrite=False):
        if to_file:
            if os.path.exists(file_path) and not overwrite:
                temp_df = pd.read_json(file_path, lines=True)
                exisiting_ids = set(temp_df['id'].tolist())
                print(f'Loaded {len(exisiting_ids)} existing predictions')
                ds = [sample for sample in ds if sample.id not in exisiting_ids]
                print(f'Predicting {len(ds)} new samples')
            else:
                with open(file_path, 'w') as f:
                    f.write("")
        for sample in tqdm(ds, total=len(ds)):
            sample_predictions = self.predict_sample(sample)
            if to_file:
                self.write_to_file(file_path, sample, sample_predictions)

    def predict_sample(self, sample):
        return [np.random.choice(sample.classes)]


class MajorityPredictor(AutoRegressivePredictor):
    def __init__(self, model_name, max_tokens=None, stop_at=None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.stop_at = stop_at
        self.num_written = 0

    def write_to_file(self, file_path, sample, predictions):
        with open(file_path, 'a') as f:
            prediction = predictions[0]
            out_dict = {
                    'approach': self.model_name,
                    'id': sample.id,
                    'is_clf': sample.is_clf,
                    'is_reg': sample.is_reg,
                    'input': sample.input,
                    'prediction': prediction,
                    'reference': sample.output,
                    }
            json.dump(out_dict, f)
            f.write('\n')

    def predict_ds(self, ds, to_file=False, file_path=None, overwrite=False):
        if to_file:
            if os.path.exists(file_path) and not overwrite:
                temp_df = pd.read_json(file_path, lines=True)
                exisiting_ids = set(temp_df['id'].tolist())
                print(f'Loaded {len(exisiting_ids)} existing predictions')
                ds = [sample for sample in ds if sample.id not in exisiting_ids]
                print(f'Predicting {len(ds)} new samples')
            else:
                with open(file_path, 'w') as f:
                    f.write("")
        for sample in tqdm(ds, total=len(ds)):
            sample_predictions = self.predict_sample(sample)
            if to_file:
                self.write_to_file(file_path, sample, sample_predictions)

    def predict_sample(self, sample):
        return [sample.classes[0]]


def get_instruction_pattern(model_name):
    if model_name == 'alpaca':
        instruct_pre = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''
        instruct_pattern = '''### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n'''
        return instruct_pre, instruct_pattern
    elif model_name == 'llama3-8b-instruct' or model_name == 'argpaca-8b' or model_name == 'argpaca-8b-fulldata' or model_name == 'random' or model_name == 'gpt-4o-mini' or model_name == 'majority' or 'gemma-2-9b-ca' in model_name or 'gemma-2-9b-instruct' in model_name or 'gemma-2-9b-alpaca' in model_name or 'gemma-2-9b' in model_name or model_name == 'mean':
        instruct_pre = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n'''
        instruct_pattern = '''### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n'''
        return instruct_pre, instruct_pattern
    elif model_name == 'Meta-Llama-3-8B-Instruct':
        instruct_pre = ''''''
        instruct_pattern = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}\nInput:{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
        return instruct_pre, instruct_pattern
    elif 'gemma-2-9b-it' in model_name or 'gemma-2-9b-ca' in model_name:
        instruct_pre = ''''''
        instruct_pattern = '''<bos><bos><start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n'''
        return instruct_pre, instruct_pattern
    elif model_name == 'Mistral-7B-Instruct-v0.3' or model_name == 'Ministral-8B-Instruct-2410' or 'Mistra' in model_name:
        instruct_pre = ''''''
        instruct_pattern = '''<s>[INST] {}\nInput:{}[/INST]'''
        return instruct_pre, instruct_pattern
    else:
        instruct_pre = ''''''
        instruct_pattern = '''{}\nInput:{}\n\nOutput:'''


def get_default_preds(dataset, model_name, overwrite, balanced_dataset):
    args = parse_args()
    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    save_path = 'predictions_default.jsonl'

    np.random.seed(42)
    instance_sampler = BalancedInstanceSampler(clf_only=True)

    test_instances = []
    for instances in instance_sampler.get_batch('test', 10, shuffle=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            test_instances.append(instance)

    model = AutoRegressivePredictor(args.model_name, GEN_ARGS)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)


def get_outline_preds(dataset, model_name, overwrite, balanced_dataset):
    args = parse_args()
    if 'checkpoint' in model_name:
        model_name = model_name.split('/')[-2] + '-' + model_name.split('/')[-1]

    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    if balanced_dataset:
        save_path = base_path + '/data/inference/predictions_outline_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'balanced')
    else:
        save_path = base_path + '/data/inference/predictions_outline_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'unbalanced')
    np.random.seed(42)
    if dataset == 'ca':
        instance_sampler = InstanceSampler()
    elif dataset == 'superni':
        instance_sampler = SuperNiInstanceSampler()
    else:
        raise ValueError('Invalid dataset name')

    train_instances = []
    for instances in instance_sampler.get_batch('train', 100, 1, balanced=True, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            train_instances.append(instance)
    print(len(train_instances))

    test_instances = []
    for instances in instance_sampler.get_batch('test', 100, 1, balanced=balanced_dataset, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            test_instances.append(instance)
    print(len(test_instances))

    # group instances based on task_name and add all possible outputs for a group to each instance
    if model_name == 'gpt-4o-mini':
        instances_for_classes = train_instances if dataset == 'ca' else test_instances
        task_groups = {}
        for instance in [instance for instance in instances_for_classes if instance.is_clf or instance.is_reg]:
            if instance.task_name not in task_groups:
                task_groups[instance.task_name] = []
            task_groups[instance.task_name].append(instance.output)
        for instance in [instance for instance in test_instances if instance.is_clf or instance.is_reg]:
            instance.classes = list(set(task_groups[instance.task_name]))
    else:
        instances_for_classes = train_instances if dataset == 'ca' else test_instances
        task_groups = {}
        for instance in [instance for instance in instances_for_classes if instance.is_clf]:
            if instance.task_name not in task_groups:
                task_groups[instance.task_name] = []
            task_groups[instance.task_name].append(instance.output)
        for instance in [instance for instance in test_instances if instance.is_clf]:
            instance.classes = list(set(task_groups[instance.task_name]))

    model = OutlinePredictor(args.model_name, max_tokens=None, stop_at=None)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)


def get_random_preds(dataset, model_name, overwrite, balanced_dataset):
    args = parse_args()
    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    if balanced_dataset:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'balanced')
    else:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'unbalanced')
    np.random.seed(42)
    if dataset == 'ca':
        instance_sampler = InstanceSampler()
    elif dataset == 'superni':
        instance_sampler = SuperNiInstanceSampler()
    else:
        raise ValueError('Invalid dataset name')

    train_instances = []
    for instances in instance_sampler.get_batch('train', 100, 1, balanced=True, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            train_instances.append(instance)
    print(len(train_instances))

    test_instances = []
    for instances in instance_sampler.get_batch('test', 100, 1, balanced=balanced_dataset, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            test_instances.append(instance)
    print(len(test_instances))

    # group instances based on task_name and add all possible outputs for a group to each instance
    instances_for_classes = train_instances if dataset == 'ca' else test_instances
    task_groups = {}
    for instance in [instance for instance in instances_for_classes]:
        if instance.task_name not in task_groups:
            task_groups[instance.task_name] = []
        task_groups[instance.task_name].append(instance.output)
    for instance in [instance for instance in test_instances]:
        instance.classes = list(set(task_groups[instance.task_name]))

    model = RandomPredictor(args.model_name, max_tokens=None, stop_at=None)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)


def get_majority_preds(dataset, model_name, overwrite, balanced_dataset):
    args = parse_args()
    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    if balanced_dataset:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'balanced')
    else:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'unbalanced')

    np.random.seed(42)
    if dataset == 'ca':
        instance_sampler = InstanceSampler()
    elif dataset == 'superni':
        instance_sampler = SuperNiInstanceSampler()
    else:
        raise ValueError('Invalid dataset name')

    train_instances = []
    for instances in instance_sampler.get_batch('train', 100, 1, balanced=True, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            train_instances.append(instance)
    print(len(train_instances))

    test_instances = []
    for instances in instance_sampler.get_batch('test', 100, 1, balanced=balanced_dataset, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            test_instances.append(instance)
    print(len(test_instances))

    # group instances based on task_name and add all possible outputs for a group to each instance
    instances_for_classes = train_instances if dataset == 'ca' else test_instances
    task_groups = {}
    for instance in [instance for instance in instances_for_classes]:
        if instance.task_name not in task_groups:
            task_groups[instance.task_name] = []
        task_groups[instance.task_name].append(instance.output)
    for instance in [instance for instance in test_instances]:
        # pick majority class from train instances
        instance.classes = [max(task_groups[instance.task_name], key=task_groups[instance.task_name].count)]

    model = MajorityPredictor(args.model_name, max_tokens=None, stop_at=None)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)


def get_mean_preds(dataset, model_name, overwrite, balanced_dataset):
    args = parse_args()
    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    if balanced_dataset:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'balanced')
    else:
        save_path = base_path + '/data/inference/predictions_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'unbalanced')

    np.random.seed(42)
    if dataset == 'ca':
        instance_sampler = InstanceSampler()
    elif dataset == 'superni':
        instance_sampler = SuperNiInstanceSampler()
    else:
        raise ValueError('Invalid dataset name')

    train_instances = []
    for instances in instance_sampler.get_batch('train', 100, 1, balanced=True, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            train_instances.append(instance)
    print(len(train_instances))

    test_instances = []
    for instances in instance_sampler.get_batch('test', 100, 1, balanced=balanced_dataset, shuffle=True, early_stopping=False):
        for instance in instances:
            instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
            instance.task_name = instance.id.split('_')[0]
            test_instances.append(instance)
    print(len(test_instances))

    # group instances based on task_name and add all possible outputs for a group to each instance
    instances_for_classes = train_instances if dataset == 'ca' else test_instances
    task_groups = {}
    for instance in [instance for instance in instances_for_classes]:
        if instance.task_name not in task_groups:
            task_groups[instance.task_name] = []
        task_groups[instance.task_name].append(instance.output)
    for instance in [instance for instance in test_instances]:
        #if task is regression, take mean of all outputs
        if instance.is_reg:
            instance.classes = [str(np.mean([float(x) for x in task_groups[instance.task_name]]))]
        else:
            instance.classes = [max(task_groups[instance.task_name], key=task_groups[instance.task_name].count)]

    model = MajorityPredictor(args.model_name, max_tokens=None, stop_at=None)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)


def get_task_preds(dataset, model_name, overwrite):
    from tasks.upk_sentential_argument_mining import ArgumentIdentificationUKPSententialArgumentMining
    from tasks.argument_annotated_essays_2 import IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2
    from tasks.appropriateness_corpus import InappropriatenessDetectionAppropriatenessCorpus
    from tasks.ibm_rank_30k import QualityAssessmentIbmRank30k
    from tasks.enthymemes_student_essays import ReconstructEnthymemesEnthymemesStudentEssays
    from tasks.debate_sum import ExtractiveSummarizationDebateSum
    tasks = [
        ArgumentIdentificationUKPSententialArgumentMining(),
        IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2(),
        InappropriatenessDetectionAppropriatenessCorpus(),
        QualityAssessmentIbmRank30k(),
        ReconstructEnthymemesEnthymemesStudentEssays(),
        ExtractiveSummarizationDebateSum()
    ]
    args = parse_args()
    if 'checkpoint' in model_name:
        model_name = model_name.split('/')[-2] + '-' + model_name.split('/')[-1]

    instruct_pre, instruct_pattern = get_instruction_pattern(model_name.split('/')[-1])
    save_path = base_path + '/data/inference/predictions_outline_{}_{}_{}.jsonl'.format(model_name.split('/')[-1], dataset, 'full-test')
    np.random.seed(42)

    instances = []
    for task in tasks:
        instances += task.instances

    train_instances = []
    for instance in [instance for instance in instances if instance.split == 'train']:
        instance.task_name = instance.id.split('_')[0]
        train_instances.append(instance)
    print(len(train_instances))

    test_instances = []
    for instance in [instance for instance in instances if instance.split == 'test']:
        instance.input = instruct_pre + instruct_pattern.format(instance.task_instruction, instance.input)
        instance.task_name = instance.id.split('_')[0]
        test_instances.append(instance)
    print(len(test_instances))

    # group instances based on task_name and add all possible outputs for a group to each instance
    instances_for_classes = train_instances
    task_groups = {}
    for instance in [instance for instance in instances_for_classes if instance.is_clf]:
        if instance.task_name not in task_groups:
            task_groups[instance.task_name] = []
        task_groups[instance.task_name].append(instance.output)
    for instance in [instance for instance in test_instances if instance.is_clf]:
        instance.classes = list(set(task_groups[instance.task_name]))

    model = OutlinePredictor(args.model_name, max_tokens=None, stop_at=None)
    model.predict_ds(test_instances, to_file=True, file_path=save_path, overwrite=overwrite)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default='default')
    parser.add_argument('--dataset', type=str, default='default')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--balanced_dataset', action='store_true', default=False)
    parser.add_argument('--full-test', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_test:
        get_task_preds(args.dataset, args.model_name, args.overwrite)
    elif args.mode == 'default':
        get_default_preds(args.dataset, args.model_name, args.overwrite, args.balanced_dataset)
    elif args.mode == 'outline':
        get_outline_preds(args.dataset, args.model_name, args.overwrite, args.balanced_dataset)
    elif args.mode == 'random':
        get_random_preds(args.dataset, 'random', args.overwrite, args.balanced_dataset)
    elif args.mode == 'majority':
        get_majority_preds(args.dataset, 'majority', args.overwrite, args.balanced_dataset)
    elif args.mode == 'mean':
        get_mean_preds(args.dataset, 'mean', args.overwrite, args.balanced_dataset)
    else:
        raise ValueError('Invalid argument combination')
