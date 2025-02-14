import numpy as np
import pandas as pd
import csv
import os

import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/debate-sum/debate2019.csv'


class ExtractiveSummarizationDebateSum(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='extractive-summarization_debate-dum',
            task_instruction='Create a word-level extractive summary of the argument by “underlining” and/or “highlighting” the evidence in such a way to support the argument being made.',
            dataset_names=['debate-sum'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        data = pd.read_csv(ds_path)
        inds = list(range(len(data)))
        np.random.shuffle(inds)
        train_inds = inds[:int(len(inds)*0.7)]
        dev_inds = inds[int(len(inds)*0.7):int(len(inds)*0.8)]
        test_inds = inds[int(len(inds)*0.8):]

        for i, row in data.iterrows():
            split = 'train'
            if i in dev_inds:
                split = 'dev'
            elif i in test_inds:
                split = 'test'
            instance = Instance(
                input=row['Full-Document'],
                output=row['Extract'],
                split=split,
            )
            self.instances.append(instance)

if __name__ == '__main__':
    task = ExtractiveSummarizationDebateSum()
    # sort instaces by input length
    task.instances.sort(key=lambda x: len(x.input))
    for i in range(10):
        print(task.instances[i+8000].input)
        print(task.instances[i+8000].output)
        print(task.instances[i+8000].split)
        print()
