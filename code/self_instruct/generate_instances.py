# adapted from https://github.com/yizhongw/self-instruct/blob/main/self_instruct/generate_instances.py

import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from transformers import AutoTokenizer
from templates.instance_gen_template import output_first_template_for_clf, input_first_template_for_gen, output_first_template_for_reg

import sys
sys.path.insert(0, '..')
from samplers import InstanceGenerationSampler

random.seed(42)
base_path = os.environ['ARGPACA_MAJA']


def clean_text(text):
    # This regex will match one or more occurrences of newline characters
    return re.sub(r's*(\n)\s*(\n)+', '\n', text)


def encode_prompt(seed_instances, task_type):
    """Encode multiple prompt instructions into a single string."""
    # group instances by task instruction
    grouped_instances = {}
    for seed_instance in seed_instances:
        if seed_instance.task_instruction not in grouped_instances:
            grouped_instances[seed_instance.task_instruction] = []
        grouped_instances[seed_instance.task_instruction].append(seed_instance)

    # sample 10 tasks
    grouped_instances = dict(random.sample(grouped_instances.items(), min(10, len(grouped_instances))))

    model_path = "Meta-Llama-3-70B" 
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    num_prefix_tokens = 0

    if task_type == 'clf':
        prefix = '''Given the classification task definition and the class labels, generate an input that corresponds to each of the class labels. If the task doesn't require input, just generate possible class labels.\n\n'''
        num_prefix_tokens += tokenizer.encode(prefix, return_tensors="pt").shape[1]
        for task_instruction, seed_instances in grouped_instances.items():
            task_prefix = '''Task: ''' + seed_instance.task_instruction + "\n"
            for seed_instance in seed_instances:
                task_prefix += '''Class label: ''' + clean_text(seed_instance.output) + "\n"
                task_prefix += '''Input: ''' + clean_text(seed_instance.input) + "\n"
            num_prefix_tokens += tokenizer.encode(task_prefix, return_tensors="pt").shape[1]
            if num_prefix_tokens > 3072:
                break
            else:
                prefix += task_prefix
            prefix += "\n"
    elif task_type == 'reg':
        prefix = '''Given the regression task definition and the score range, generate an input that corresponds to the lower bound, upper bound and a score in the middle of the range. If the task doesn't require input, just generate possible scores.\n\n''' 
        num_prefix_tokens += tokenizer.encode(prefix, return_tensors="pt").shape[1]
        for task_instruction, seed_instances in grouped_instances.items():
            task_prefix = '''Task: ''' + seed_instance.task_instruction + "\n"
            for seed_instance in seed_instances:
                task_prefix += '''Score: ''' + clean_text(seed_instance.output) + "\n"
                task_prefix += '''Input: ''' + clean_text(seed_instance.input) + "\n"
            num_prefix_tokens += tokenizer.encode(task_prefix, return_tensors="pt").shape[1]
            if num_prefix_tokens > 3072:
                break
            else:
                prefix += task_prefix
            prefix += "\n"
    else:
        prefix = '''Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.\n\n'''
        num_prefix_tokens += tokenizer.encode(prefix, return_tensors="pt").shape[1]
        for task_instruction, seed_instances in grouped_instances.items():
            task_prefix = '''Task: ''' + seed_instance.task_instruction + "\n"
            for seed_instance in seed_instances:
                task_prefix += '''Input: ''' + clean_text(seed_instance.input) + "\n"
                task_prefix += '''Output: ''' + clean_text(seed_instance.output) + "\n"
            num_prefix_tokens += tokenizer.encode(task_prefix, return_tensors="pt").shape[1]
            if num_prefix_tokens > 3072:
                break
            else:
                prefix += task_prefix
            prefix += "\n"
    prefix += '''Task:'''
    return prefix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default=base_path + "/data/self_instruct_llm_generations_maja/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--regression_tasks_only",
        action="store_true",
        help="If specified, only do for regression tasks.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sampler = InstanceGenerationSampler().get_batch('train')

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)
    print(f"Loaded {len(tasks)} tasks from {args.input_file}")

    task_types = {}
    tasks_filtered = []
    with open(os.path.join(args.batch_dir, "is_clf_or_not_filtered_Meta-Llama-3-70B.jsonl")) as fin:
        for line in fin:
            data = json.loads(line)
            
            # filter tasks
            tasks_filtered.append([t for t in tasks if t["instruction"] == data["instruction"]][0])
          
            if data["is_classification"].strip() in ["Yes", "yes", "YES"]:
                task_types[data["instruction"]] = 'clf'
            elif data["is_regression"].strip() in ["Yes", "yes", "YES"]:
                task_types[data["instruction"]] = 'reg'
            else:
                task_types[data["instruction"]] = 'gen'
    tasks = tasks_filtered
    print(f"Filtering resulted in {len(tasks)} tasks")

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_types[task["instruction"]] == 'clf']

    if args.regression_tasks_only:
        tasks = [task for task in tasks if task_types[task["instruction"]] == 'reg']

    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_types[task["instruction"]] == 'gen']

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    
    from llm_requests import make_requests
    
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "raw_instances", "instance_metadata", "instruction_metadata",
                         "most_similar", "avg_similarity_score"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=True) + "\n")
            else:
                prompts = []
                #seed_instances_batch = next(sampler)
                for task in batch:
                    #if task_types[task["instruction"]] == 'clf':
                    #    prompt_instances = [t for t in seed_instances_batch if t.is_clf]
                    #elif task_types[task["instruction"]] == 'reg':
                    #    prompt_instances = [t for t in seed_instances_batch if t.is_reg]
                    #else:
                    #    prompt_instances = [t for t in seed_instances_batch if not t.is_clf and not t.is_reg]
                    #prefix = encode_prompt(prompt_instances, task_types[task["instruction"]])
                    #prompt = prefix + " " + task["instruction"].strip() + "\n"
                    if task_types[task["instruction"]] == 'clf':
                        prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                    elif task_types[task["instruction"]] == 'reg':
                        prompt = output_first_template_for_reg + " " + task["instruction"].strip() + "\n"
                    else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\n"

                    prompts.append(prompt)
                results = make_requests(prompts, max_tokens=1024, greedy=True, presence_penalty=1.5)
                #    engine=args.engine,
                #    prompts=prompts,
                #    # because the clf template is longer, we need to decrease the max_tokens
                #    max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                #    temperature=0,
                #    top_p=0,
                #    frequency_penalty=0,
                #    presence_penalty=1.5,
                #    stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                #    logprobs=1,
                #    n=1,
                #    best_of=1,
                #    api_key=args.api_key,
                #    organization=args.organization)
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"].split('Task:')[0].strip()
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "raw_instances", "instance_metadata", "instruction_metadata",
                         "most_similar", "avg_similarity_score"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=True) + "\n")
            progress_bar.update(len(batch))
