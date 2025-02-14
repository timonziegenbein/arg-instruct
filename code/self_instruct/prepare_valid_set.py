import os
import json
import argparse
import glob
import re
import random
import tqdm
import pandas as pd


random.seed(123)
base_path = os.environ['ARGPACA_MAJA']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampled_instances_file",
        nargs="+",
        default=base_path + "/data/self_instruct_llm_generations_maja/finetuning/sampled_generated_instances_47191.json",
        type=str,
        help="The input files that contains the sampled machine generated instances."
    )
    parser.add_argument(
        "--output_dir",
        default=base_path + "/data/self_instruct_llm_generations_maja/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for finetuning."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_generated_instances = []
    with open(os.path.join(args.output_dir, "all_generated_instances.json"), "r") as fin:
        all_generated_instances = json.loads(fin.readline())
    print(f"All generated instances: {len(all_generated_instances)}")

    sampled_instances = []
    with open(os.path.join(args.output_dir, args.sampled_instances_file), "r") as fin:
        sampled_instances = json.loads(fin.readline())           
    print(f"Sampled instances for training: {len(sampled_instances)}")    

    potential_valid_instances = [inst for inst in all_generated_instances if not inst in sampled_instances]
    unique_valid_instructions = set([it["instruction"] for it in potential_valid_instances])
    print(f"Potential instances for validation: {len(potential_valid_instances)}")
    print(f"Unique valid instructions: {len(unique_valid_instructions)}")

    # sample one instance per instruction (5.778 or 10%)
    sampled_valid_instances = []
    for instruction in random.sample(list(unique_valid_instructions), 5778):
        sampled_valid_instances.extend(random.sample([it for it in potential_valid_instances if it["instruction"] == instruction], 1))    
    print(f"Sampled {len(sampled_valid_instances)} valid instances.")
    
    # commented out for safety ;)
    #with open(os.path.join(args.output_dir, f"sampled_generated_val_instances_{len(sampled_valid_instances)}.json"), "w") as fout:
        #fout.write(json.dumps(sampled_valid_instances))
    