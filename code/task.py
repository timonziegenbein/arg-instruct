import os
import hashlib
import numpy as np
import pandas as pd

INSTRUCTION_TEMPLATE_INPUT_OUTPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

INSTRUCTION_TEMPLATE_OUTPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}
"""

base_path = os.environ['ARGPACA_MAJA']
_CACHE_DIR = base_path + '/data/cache/'

class Instance:
    def __init__(self, input, output, split):
        self.input = input
        self.output = output
        self.split = split  # train, dev, test

    def apply_template(self):
        input_output = hasattr(self, 'input')
        if input_output and self.input:
            return INSTRUCTION_TEMPLATE_INPUT_OUTPUT.format(self.task_instruction, self.input, self.output)
        else:
            return INSTRUCTION_TEMPLATE_OUTPUT.format(self.task_instruction, self.output)

    def __str__(self):
        return f'in: {self.input}\nout: {self.output}\nsplit: {self.split}'


class Task:
    def __init__(self, task_name, task_instruction, dataset_names, is_clf=False, is_reg=False, load_data=True, from_cache=True):
        self.task_name = task_name
        self.task_instruction = task_instruction
        self.dataset_names = dataset_names # deprecated
        self.instances = []
        self.is_clf = is_clf
        self.is_reg = is_reg

        if load_data:
            if not from_cache or not os.path.exists(_CACHE_DIR + self.task_name + '.parquet.gzip'):
                self.load_data()
                self._to_parquet()
            else:
                self._from_parquet()

            self.update_instances_with_task_properties()

    def update_instances_with_task_properties(self):
        for i, instance in enumerate(self.instances):
            instance.id = self.task_name + '_instance{}'.format(hashlib.md5(' '.join([instance.input, instance.output]).encode('utf-8')).hexdigest())
            instance.task_instruction = self.task_instruction
            instance.is_clf = self.is_clf
            instance.is_reg = self.is_reg
            instance.dataset_names = self.dataset_names

    def load_data(self):
        raise NotImplementedError

    def get_batch(self, split, batch_size=1, balanced=False, shuffle=False, early_stopping=False):
        if balanced and self.is_clf:
            return self._get_balanced_output_batch(split, batch_size, shuffle, 'clf', early_stopping)
        elif balanced and self.is_reg:
            return self._get_balanced_output_batch(split, batch_size, shuffle, 'reg')
        else:
            return self._get_single_instance_batch(split, batch_size, shuffle)

    def _get_balanced_output_batch(self, split, batch_size, shuffle, task_type, early_stopping=False):
        if task_type == 'clf':
            # get all output types
            output_types = sorted(set([instance.output for instance in self.instances]))
            # split instances by output types
            split_instances = [[
                instance for instance in self.instances if instance.output == output_type and instance.split == split] for output_type in output_types]
        elif task_type == 'reg':
            # split into batch_size number of chunks
            split_instances = np.array_split(sorted([i for i in self.instances if i.split == split], key=lambda x: float(x.output)), batch_size)

        # shuffle split_instances
        if shuffle:
            split_instances = [np.random.permutation(x) for x in split_instances]

        total_instances = sum([len(instances) for instances in split_instances])
        batch = []
        if early_stopping:
            for i in range(min([len(instances) for instances in split_instances])):
                batch += [instances[i] for instances in split_instances]
                if len(batch) >= batch_size or len(batch) == total_instances:
                    yield batch
                    batch = []
        else:
            for i in range(max([len(instances) for instances in split_instances])):
                batch += [instances[i] for instances in split_instances if i < len(instances)]
                if len(batch) >= batch_size or len(batch) == total_instances:
                    yield batch
                    batch = []

    def _get_single_instance_batch(self, split, batch_size, shuffle):
        if shuffle:
            self.instances = np.random.permutation(self.instances)
        split_instances = [instance for instance in self.instances if instance.split == split]
        if split_instances:
            batch = []
            for i in range(len(split_instances)):
                batch += [split_instances[i]]
                if len(batch) >= batch_size or len(batch) == len(split_instances):
                    yield batch
                    batch = []
        else:
            yield []

    def _to_parquet(self):
        df = pd.DataFrame([vars(instance) for instance in self.instances])
        df.to_parquet(_CACHE_DIR + self.task_name + '.parquet.gzip', compression='gzip', index=False)

    def _from_parquet(self):
        df = pd.read_parquet(_CACHE_DIR + self.task_name + '.parquet.gzip')
        self.instances = [Instance(**row) for i, row in df.iterrows()]
