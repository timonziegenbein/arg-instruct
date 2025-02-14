import os
import numpy as np
import pandas as pd
import json
import random
import sys
sys.path.insert(0, '..')
from samplers import InstanceSampler, BalancedInstanceSampler, SuperNiInstanceSampler

TEST_TASKS_NAMES = [
    'argument-annotated-essays-2', 'qt30', 'f1000rd', 'iac-v2', 'ibm-rank-30k', 'arguana-counterargs-corpus', 'aspect-controlled-argument-generation', 'debate-sum', 'webis-conclugen-21'
]

if __name__ == '__main__':
    instance_sampler = BalancedInstanceSampler()
    train_instances = []
    num_instances = 0
    for instances in instance_sampler.get_batch('train', 1, 1000, balanced=True, shuffle=True):
        num_instances += len(instances)
        for instance in instances:
            if instance.dataset_names[0] in TEST_TASKS_NAMES:
                train_instances.append({
                    'instruction': instance.task_instruction,
                    'input': instance.input,
                    'output': instance.output
                })
        if len(train_instances) >= 57204:
            break
    print(f"Loaded {num_instances} train instances")

    random.shuffle(train_instances)
    df = pd.DataFrame.from_dict(train_instances)
    train_df = df[:52004]
    valid_df = df[52004:]
    train_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca52k_seed", f"train.json"), orient='records', lines=True)
    valid_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca52k_seed", f"valid.json"), orient='records', lines=True)
    print(f"Saved {len(train_df)} train instances in total")
    print(f"Saved {len(valid_df)} valid instances in total")

    instance_sampler = BalancedInstanceSampler()
    train_instances = []
    num_instances = 0
    for instances in instance_sampler.get_batch('train', 1, 1000, balanced=True, shuffle=True):
        num_instances += len(instances)
        for instance in instances:
            if instance.dataset_names[0] not in TEST_TASKS_NAMES:
                train_instances.append({
                    'instruction': instance.task_instruction,
                    'input': instance.input,
                    'output': instance.output
                })
        if len(train_instances) >= 57204:
            break
    print(f"Loaded {num_instances} train instances")

    random.shuffle(train_instances)
    df = pd.DataFrame.from_dict(train_instances)
    train_df = df[:52004]
    valid_df = df[52004:]
    train_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca52k", f"train.json"), orient='records', lines=True)
    valid_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca52k", f"valid.json"), orient='records', lines=True)
    print(f"Saved {len(train_df)} train instances in total")
    print(f"Saved {len(valid_df)} valid instances in total")


    #valid_df = pd.read_json("/mnt/home/mstahl/argpaca/data/self_instruct_llm_generations_maja/finetuning/sampled_generated_val_instances_5778.json")
    #print(valid_df.head())
    #print(f"Loaded {len(valid_df)} valid instances")
    #valid_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca52k", f"valid.json"), orient='records', lines=True)
    #print(f"Saved {len(valid_df)} valid instances in total")

    #train_df = pd.read_json("/mnt/home/mstahl/argpaca/data/self_instruct_llm_generations_maja/finetuning/sampled_generated_train_instances_52445.json")
    #print(train_df.head())
    #print(f"Loaded {len(train_df)} train instances")
    #train_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca_gen52k", f"train.json"), orient='records', lines=True)
    #print(f"Saved {len(train_df)} train instances in total")

    #valid_df = pd.read_json("/mnt/home/mstahl/argpaca/data/self_instruct_llm_generations_maja/finetuning/sampled_generated_val_instances_5778.json")
    #print(valid_df.head())
    #print(f"Loaded {len(valid_df)} valid instances")
    #valid_df.to_json(os.path.join("/mnt/home/tziegenb/argpaca/data/train/ca_gen52k", f"valid.json"), orient='records', lines=True)
    #print(f"Saved {len(valid_df)} valid instances in total")
