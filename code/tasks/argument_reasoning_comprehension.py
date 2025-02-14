from task import Task, Instance
import pandas as pd
import numpy as np
import os

base_path = os.environ['ARGPACA_MAJA']

class ArgumentReasoningComprehension(Task):
    def __init__(self, **kwargs):
        super().__init__('argument-reasoning-comprehension_argument-reasoning-comprehension',
                         'Given an argument consisting of a claim and a reason, select the correct warrant that explains reasoning of this particular argument. There are only two options given and only one answer is correct.', ['argument-reasoning-comprehension'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/argument-reasoning-comprehension-task/'
        for split in ['train', 'dev', 'test']:
            df = pd.read_csv(ds_path + split + '-full.txt', sep='\t')
            for i, row in df.iterrows():
                instance = Instance(
                    'Topic: ' + row['debateTitle'] + '\nAdditional Info: ' + row['debateInfo'] + '\nClaim: ' + row['claim'] +
                    '\nReason: ' + row['reason'] + '\nWarrant 1: ' +
                    row['warrant0'] + '\nWarrant 2: ' + row['warrant1'],
                    [row['warrant0'], row['warrant1']][row['correctLabelW0orW1']],
                    split,
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = ArgumentReasoningComprehension()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)

    print('+'*50)

    batch = task.get_batch(split='dev')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)

    print('+'*50)

    batch = task.get_batch(split='test')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
