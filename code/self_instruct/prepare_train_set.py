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
        "--output_dir",
        default=base_path + "/data/self_instruct_llm_generations_maja/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for finetuning."
    )
    parser.add_argument(
        "--max_size",
        default=52500,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_generated_instances = []
    with open(os.path.join(args.output_dir, "all_generated_instances.json"), "r") as fin:
        all_generated_instances = json.loads(fin.readline())
    print(f"All generated instances: {len(all_generated_instances)}")

    # not valid instructions
    validation_instances = []
    with open(os.path.join(args.output_dir, "sampled_generated_val_instances_5778.json"), "r") as fin:
        validation_instances = json.loads(fin.readline())           
    valid_instructions = [i['instruction'] for i in validation_instances]
    print(f"Instances for validation: {len(validation_instances)}")    
    print(f"Instruction for validation: {len(valid_instructions)}")    
    
    potential_train_instances = [i for i in all_generated_instances if not i['instruction'] in valid_instructions]
    print(f"Potential instances for training: {len(potential_train_instances)}")    
    
    unique_train_instructions = set([it["instruction"] for it in potential_train_instances])
    print(f"Unique train instructions: {len(unique_train_instructions)}")

    # sample one instance per instruction, maximal max_size
    max_size = min(args.max_size, len(unique_train_instructions))
    print(f'Sampling {max_size} training instances...')
        
    sampled_train_instances = []
    for instruction in random.sample(list(unique_train_instructions), max_size):
        sampled_train_instances.extend(random.sample([it for it in potential_train_instances if it["instruction"] == instruction], 1))    
    print(f"Sampled {len(sampled_train_instances)} train instances.")
    
    with open(os.path.join(args.output_dir, f"sampled_generated_train_instances_{len(sampled_train_instances)}.json"), "w") as fout:
        fout.write(json.dumps(sampled_train_instances))
    