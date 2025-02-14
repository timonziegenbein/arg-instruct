import numpy as np
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

class ReasonIdentificationReasonIdentificationAndClassification(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='reason-identification_reason-identification-and-classification',
            task_instruction='Identify the reasons in the given argumentative text.',
            dataset_names=['reason-identification-and-classification'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/reason-identification-and-classification/reason/reason'
        splits = {}

        for file in os.listdir(os.path.join(ds_path, 'folds')):
            if file.endswith('-1') or file.endswith('-2') or file.endswith('-3') or file.endswith('-4'):
                with open(os.path.join(ds_path, 'folds', file), encoding='latin-1') as f:
                    split = f.readlines()
                    f.close()
                split = [l.split(' ')[0] for l in split]
                for s in split:
                    splits[s] = 'train'
            elif file.endswith('-5'):
                with open(os.path.join(ds_path, 'folds', file), encoding='latin-1') as f:
                    split = f.readlines()
                    f.close()
                split = [l.split(' ')[0] for l in split]
                for s in split:
                    splits[s] = 'test'

        # Load all files in the abotion, gayRights, marijuana, and obama folders
        for folder in ['abortion', 'gayRights', 'marijuana', 'obama']:
            for file in os.listdir(os.path.join(ds_path, folder)):
                with open(os.path.join(ds_path, folder, file), encoding='latin-1') as f:
                    data = f.readlines()
                    f.close()

                data = [x.replace('\n', '').replace('Line##', '')
                        for x in data if x != '\n' and not x.startswith('Label')]
                instance = Instance(
                    input=data[0],
                    output='\n'.join(['Reason {}'.format(i) + ': ' + x for i, x in enumerate(data[1:])]),
                    split=splits[file.split('.')[0]],
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = ReasonIdentificationReasonIdentificationAndClassification()
    task.load_data()
