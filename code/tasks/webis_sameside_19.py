from task import Task, Instance
import numpy as np
import csv
import os

base_path = os.environ['ARGPACA_MAJA']

class SameSideStanceClassificationWebisSameside19(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='same-side-stance-classification_webis-sameside-19',
            task_instruction='Given two arguments on a topic, decide whether they are on the same side or not.',
            dataset_names=['webis-sameside-19'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/webis_sameside_19/cross-topics/'

        with open(os.path.join(ds_path, 'training.csv'), newline='') as f:
            reader = csv.reader(f)
            train_data = list(reader)

        for elem in train_data[1:]:  # skip header
            instance = Instance(
                input="Argument 1: {}\n\nArgument 2: {}".format(elem[1], elem[3]),
                output="Same side" if elem[6] == 'True' else "Not the same side",
                split='train',
            )
            self.instances.append(instance)

        with open(os.path.join(ds_path, 'test.csv'), newline='') as f:
            reader = csv.reader(f)
            test_data = list(reader)

        for elem in test_data[1:]:  # skip header
            instance = Instance(
                input="Argument 1: {}\n\nArgument 2: {}".format(elem[2], elem[3]),
                output="Same side" if elem[4] == 'True' else "Not the same side",
                split='test',
            )
            self.instances.append(instance)


if __name__ == '__main__':
    task = SameSideStanceClassificationWebisSameside19()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
