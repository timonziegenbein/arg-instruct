import os
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
    for instances in instance_sampler.get_batch('train', 1, 620, balanced=True, shuffle=True):
        num_instances += len(instances)
        for instance in instances:
            if instance.dataset_names[0] not in TEST_TASKS_NAMES:
                train_instances.append({
                    'instruction': instance.task_instruction,
                    'input': instance.input,
                    'output': instance.output
                })
        if len(train_instances) >= 52004:
            break
    print(f"Loaded {num_instances} instances")
    print(f"Save {len(train_instances)} instances in total")

    random.shuffle(train_instances)
    with open(os.path.join("/bigwork/nhwpstam/argpaca/data/train", f"ca_finetuning_data_{len(train_instances)}.json"), "w") as fout:
        fout.write(json.dumps(train_instances))
