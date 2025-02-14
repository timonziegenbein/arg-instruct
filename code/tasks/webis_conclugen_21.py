import numpy as np
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance


base_path = os.environ['ARGPACA_MAJA']

class ConclusionGenerationWebisConclugen21(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='conclusion-generation_webis-conclugen-21',
            task_instruction='Generate an informative conclusion for the given argumentative text.',
            dataset_names=['webis-conclugen-21'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/webis-conclugen-2021/base'

        for split in ['train', 'val', 'test']:
            with open(os.path.join(ds_path, f'{split}.source')) as f:
                source_data = f.readlines()
                f.close()
            with open(os.path.join(ds_path, f'{split}.target')) as f:
                target_data = f.readlines()
                f.close()

            for source, target in zip(source_data, target_data):
                instance = Instance(
                    input=source.replace('\n', ''),
                    output=target.replace('\n', ''),
                    split=split if split != 'val' else 'dev',
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = ConclusionGenerationWebisConclugen21()
    for i in range(10):
        print(task.instances[i].input)
        print(task.instances[i].output)
        print(task.instances[i].split)
        print()
