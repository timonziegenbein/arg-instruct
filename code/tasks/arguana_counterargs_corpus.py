import numpy as np
import pandas as pd
import json
import os
import csv

import sys
sys.path.insert(0, '..')
from task import Task, Instance


base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/arguana-counterargs-corpus/03-pairs-best-counter-task/'
arg_path = base_path + '/data/arguana-counterargs-corpus/02-extracted-arguments/'


class SameDebateOpposingCountersArguanaCounterargsCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='same-debate-opposing-counters_arguana-counterargs-corpus',
            task_instruction='Given an argument, which of the candidates is the best counterargument to it? All counters in the same debate with stance opposite to the given argument are candidates. The task is to find the best counterargument among all counters to the argument’s stance.',
            dataset_names=['arguana-counterargs-corpus'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        np.random.seed(42)
        for split in ['training', 'validation', 'test']:
            split_label = split
            if split_label == 'validation':
                split_label = 'dev'
            elif split_label == 'training':
                split_label = 'train'

            data = pd.read_csv(ds_path + split + '/01-debate-opposing-counters.tsv', sep='\t', header=None)

            argument = ''
            candidates = []
            true_candidate = ''

            for i, row in data.iterrows():
                with open(arg_path + row[0]) as f:
                    new_argument = f.readlines()

                if argument == '' or new_argument == argument:
                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates.append(candidate)

                    label = row[2]
                    if label == True:
                        true_candidate = candidate

                elif new_argument != argument:
                    candidate_string = ''
                    np.random.shuffle(candidates)
                    for j in range(len(candidates)):
                        candidate_string += f"Candidate {j+1}: \"{''.join(candidates[j]).strip()}\"\n"
                    instance = Instance(
                        input=f"Argument: {''.join(argument)}\n{candidate_string}",
                        output=f"{''.join(true_candidate)}",
                        split=split_label,
                    )
                    self.instances.append(instance)

                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates = [candidate]

                    label = row[2]
                    if label == True:
                        true_candidate = candidate
                    else:
                        true_candidate = ''


class SameDebateCountersArguanaCounterargsCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='same-debate-counters_arguana-counterargs-corpus',
            task_instruction='Given an argument, which of the candidates is the best counterargument to it? All counters in the same debate irrespective of their stance are candidates. The task is to find the best counterargument among all on-topic arguments phrased as counters.',
            dataset_names=['arguana-counterargs-corpus'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        np.random.seed(42)
        for split in ['training', 'validation', 'test']:
            split_label = split
            if split_label == 'validation':
                split_label = 'dev'
            elif split_label == 'training':
                split_label = 'train'

            data = pd.read_csv(ds_path + split + '/02-debate-counters.tsv', sep='\t', header=None)

            argument = ''
            candidates = []
            true_candidate = ''

            for i, row in data.iterrows():
                with open(arg_path + row[0]) as f:
                    new_argument = f.readlines()

                if argument == '' or new_argument == argument:
                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates.append(candidate)

                    label = row[2]
                    if label == True:
                        true_candidate = candidate

                elif new_argument != argument:
                    candidate_string = ''
                    np.random.shuffle(candidates)
                    for j in range(len(candidates)):
                        candidate_string += f"Candidate {j+1}: \"{''.join(candidates[j]).strip()}\"\n"
                    instance = Instance(
                        input=f"Argument: {''.join(argument)}\n{candidate_string}",
                        output=f"{''.join(true_candidate)}",
                        split=split_label,
                    )
                    self.instances.append(instance)

                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates = [candidate]

                    label = row[2]
                    if label == True:
                        true_candidate = candidate
                    else:
                        true_candidate = ''


class SameDebateOpposingArgumentsArguanaCounterargsCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='same-debate-opposing-argument_arguana-counterargs-corpus',
            task_instruction='Given an argument, which of the candidates is the best counterargument to it? All arguments in the same debate with opposite stance are candidates. The task is to find the best among all on-topic counterarguments.',
            dataset_names=['arguana-counterargs-corpus'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        np.random.seed(42)
        for split in ['training', 'validation', 'test']:
            split_label = split
            if split_label == 'validation':
                split_label = 'dev'
            elif split_label == 'training':
                split_label = 'train'

            data = pd.read_csv(ds_path + split + '/03-debate-opposing-arguments.tsv', sep='\t', header=None)

            argument = ''
            candidates = []
            true_candidate = ''

            for i, row in data.iterrows():
                with open(arg_path + row[0]) as f:
                    new_argument = f.readlines()

                if argument == '' or new_argument == argument:
                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates.append(candidate)

                    label = row[2]
                    if label == True:
                        true_candidate = candidate

                elif new_argument != argument:
                    candidate_string = ''
                    np.random.shuffle(candidates)
                    for j in range(len(candidates)):
                        candidate_string += f"Candidate {j+1}: \"{''.join(candidates[j]).strip()}\"\n"
                    instance = Instance(
                        input=f"Argument: {''.join(argument)}\n{candidate_string}",
                        output=f"{''.join(true_candidate)}",
                        split=split_label,
                    )
                    self.instances.append(instance)

                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates = [candidate]

                    label = row[2]
                    if label == True:
                        true_candidate = candidate
                    else:
                        true_candidate = ''


class SameDebateArgumentsArguanaCounterargsCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='same-debate-argument_arguana-counterargs-corpus',
            task_instruction='Given an argument, which of the candidates is the best counterargument to it? All arguments in the same debate irrespective of their stance are candidates. The task is to find the best counterargument among all on-topic arguments.',
            dataset_names=['arguana-counterargs-corpus'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        np.random.seed(42)
        for split in ['training', 'validation', 'test']:
            split_label = split
            if split_label == 'validation':
                split_label = 'dev'
            elif split_label == 'training':
                split_label = 'train'

            data = pd.read_csv(ds_path + split + '/04-debate-arguments.tsv', sep='\t', header=None)

            argument = ''
            candidates = []
            true_candidate = ''

            for i, row in data.iterrows():
                with open(arg_path + row[0]) as f:
                    new_argument = f.readlines()

                if argument == '' or new_argument == argument:
                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates.append(candidate)

                    label = row[2]
                    if label == True:
                        true_candidate = candidate

                elif new_argument != argument:
                    candidate_string = ''
                    np.random.shuffle(candidates)
                    for j in range(len(candidates)):
                        candidate_string += f"Candidate {j+1}: \"{''.join(candidates[j]).strip()}\"\n"
                    instance = Instance(
                        input=f"Argument: {''.join(argument)}\n{candidate_string}",
                        output=f"{''.join(true_candidate)}",
                        split=split_label,
                    )
                    self.instances.append(instance)

                    argument = new_argument
                    with open(arg_path + row[1]) as f:
                        candidate = f.readlines()
                    candidates = [candidate]

                    label = row[2]
                    if label == True:
                        true_candidate = candidate
                    else:
                        true_candidate = ''

if __name__ == '__main__':
    task = SameDebateOpposingCountersArguanaCounterargsCorpus()
    task.instances.sort(key=lambda x: len(x.input))
    for i in range(10):
        print(task.instances[i].input)
        print(task.instances[i].output)
        print(task.instances[i].split)
        print()
