from task import Task, Instance
import numpy as np
import os
import json

base_path = os.environ['ARGPACA_MAJA']

class PragmaticTaggingF1000rd(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='pragmatic-tagging_f1000rd',
            task_instruction='Select the pragmatic category (the communicative purpose) for each sentence of the given peer review. The pragmatic categories are Recap (summarizes the content without evaluating it), Strength (express an explicit positive opinion), Weakness (express an explicit negative opinion), Todo (recommendations and questions), Other and Structure (labeling headers and other elements added by the review to structure the text).',
            dataset_names=['f1000rd'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/f1000rd/'
        train_file = 'train_inputs_full.json'
        with open(os.path.join(ds_path, train_file)) as f:
            data = json.load(f)
            
        np.random.shuffle(data)
        train_data = data[:int(len(data)*0.7)]
        dev_data = data[int(len(data)*0.7):int(len(data)*0.8)]
        test_data = data[int(len(data)*0.8):]
            
        for elem in train_data:
            instance = Instance(
                input='\n'.join(elem['sentences']).replace('\n\n', '\n'),
                output='\n'.join(elem['labels']),
                split='train',
            )
            self.instances.append(instance)
        for elem in dev_data:
            instance = Instance(
                input='\n'.join(elem['sentences']).replace('\n\n', '\n'),
                output='\n'.join(elem['labels']),
                split='dev',
            )
            self.instances.append(instance)
        for elem in test_data:
            instance = Instance(
                input='\n'.join(elem['sentences']).replace('\n\n', '\n'),
                output='\n'.join(elem['labels']),
                split='test',
            )
            self.instances.append(instance)


if __name__ == '__main__':
    task = PragmaticTaggingF1000rd()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
