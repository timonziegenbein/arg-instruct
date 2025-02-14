# adapted from https://github.com/yizhongw/self-instruct/blob/main/self_instruct/identify_clf_or_not.py

import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
import codecs

import sys
sys.path.insert(0, '..')

from samplers import InstructionSampler

random.seed(42)

base_path = os.environ['ARGPACA_MAJA']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        #required=True,
        default=base_path + "/data/self_instruct_llm_generations_maja/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
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
    
    instruction_sampler = InstructionSampler()
    clf_seed_instructions = instruction_sampler.get_all_by_type(is_clf=True, split="train")
    reg_seed_instructions = instruction_sampler.get_all_by_type(is_reg=True, split="train")
    gen_seed_instructions = instruction_sampler.get_all_by_type(is_gen=True, split="train")
    
    from llm_requests import make_requests

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]

    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_Meta-Llama-3-70B.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with codecs.open(output_path, "r", encoding='utf-8') as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")  


    progress_bar = tqdm.tqdm(total=len(lines))
    with codecs.open(output_path, "w", encoding='utf-8') as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification", "is_regression"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=True) + "\n")
            else:
               
                #prefix = templates[args.template]
                #prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                #results = make_requests(prompts, max_tokens=3, greedy=True)
                
                # regression
                prefix = '''Can the following task be regarded as a regression task with ordinal output?\n\n\n'''
                reg_seed_instructions = random.sample(reg_seed_instructions, 10)
                non_reg_seed_instructions = random.sample(clf_seed_instructions, 5) + random.sample(gen_seed_instructions, 5)
                random.shuffle(non_reg_seed_instructions)
                
                reg_check_instructions = {x: 'Yes' if x in reg_seed_instructions else 'No' for x in reg_seed_instructions+non_reg_seed_instructions}
                for inst, is_reg in sorted(reg_check_instructions.items(), key=lambda x: random.random()):
                    prefix += f"Task: {inst}\nIs it regression? {is_reg}\n\n"
                prefix += '''Task:'''

                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it regression?" for d in batch]
                reg_results = make_requests(prompts, max_tokens=3, greedy=True)     
                
                # classification
                prefix = '''Can the following task be regarded as a classification task with finite output labels?\n\n\n'''
                clf_seed_instructions = random.sample(clf_seed_instructions, 10)
                non_clf_seed_instructions = random.sample(reg_seed_instructions, 5) + random.sample(gen_seed_instructions, 5)
                random.shuffle(non_clf_seed_instructions)
                
                clf_check_instructions = {x: 'Yes' if x in clf_seed_instructions else 'No' for x in clf_seed_instructions+non_clf_seed_instructions}
                for inst, is_clf in sorted(clf_check_instructions.items(), key=lambda x: random.random()):
                    prefix += f"Task: {inst}\nIs it classification? {is_clf}\n\n"
                prefix += '''Task:'''

                prompts = [prefix + " " + d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                clf_results = make_requests(prompts, max_tokens=3, greedy=True) 
                
                for i in range(len(batch)):
                    #print(batch)
                    #print(results)
                    data = batch[i]
                    if reg_results[i]["response"] is not None:
                        data["is_regression"] = reg_results[i]["response"].split("Task")[0].split("\n")[0].strip() # extract Yes/No
                    else:
                        data["is_regression"] = ""
                    
                    if clf_results[i]["response"] is not None:
                        data["is_classification"] = clf_results[i]["response"].split("Task")[0].split("\n")[0].strip() # extract Yes/No
                    else:
                        data["is_classification"] = ""
                        
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"],
                        "is_regression": data["is_regression"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification", "is_regression"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=True) + "\n")
            progress_bar.update(len(batch))
