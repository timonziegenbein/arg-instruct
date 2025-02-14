from task import Task, Instance
import numpy as np
import csv
import os

base_path = os.environ['ARGPACA_MAJA']


class NoveltyClassificationArgsvalidnovel(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='novelty-classification_argsvalidnovel',
            task_instruction='Argument conclusions are novel when they contain novel premise-related content and/or combination of the content in the premises in a way that goes beyond what is stated in the premise. Is the conclusion novel?',
            dataset_names=['argsvalidnovel'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/argsvalidnovel/TaskA_'

        for split in ['train', 'dev', 'test']:
            with open(ds_path + split + '.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

                for elem in data[1:]:
                    if elem[5] == '1':
                        instance = Instance(
                            input='Premise: {}\nConclusion: {}'.format(elem[1], elem[2]),
                            output='Yes',
                            split=split,
                        )
                        self.instances.append(instance)
                    elif elem[5] == '-1':
                        instance = Instance(
                            input='Premise: {}\nConclusion: {}'.format(elem[1], elem[2]),
                            output='No',
                            split=split,
                        )
                        self.instances.append(instance)
                    else:
                        pass
                        # ignore somewhat cases


class ValidityClassificationArgsvalidnovel(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='validity-classification_argsvalidnovel',
            task_instruction='Argument conclusions are valid if they follow from the premise, meaning a logical inference links the premise to the conclusion. Is the conclusion valid?',
            dataset_names=['argsvalidnovel'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/argsvalidnovel/TaskA_'

        for split in ['train', 'dev', 'test']:
            with open(ds_path + split + '.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

                for elem in data[1:]:
                    if elem[3] == '1':
                        instance = Instance(
                            input='Premise: {}\nConclusion: {}'.format(elem[1], elem[2]),
                            output='Yes',
                            split=split,
                        )
                        self.instances.append(instance)
                    elif elem[3] == '-1':
                        instance = Instance(
                            input='Premise: {}\nConclusion: {}'.format(elem[1], elem[2]),
                            output='No',
                            split=split,
                        )
                        self.instances.append(instance)
                    else:
                        pass
                        # ignore somewhat cases


class RealtiveNoveltyClassificationArgsvalidnovel(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='relative-novelty-classification_argsvalidnovel',
            task_instruction='Argument conclusions are novel when they contain novel premise-related content and/or combination of the content in the premises in a way that goes beyond what is stated in the premise. Given the conclusions below: Is conclusion A better than conclusion B in terms of novelty?',
            dataset_names=['argsvalidnovel'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/argsvalidnovel/TaskB_'

        for split in ['train', 'dev', 'test']:
            with open(ds_path + split + '.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

                for elem in data[1:]:
                    if elem[8] == '-1':
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='Yes',
                            split=split,
                        )
                        self.instances.append(instance)
                    elif elem[8] == '1':
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='No',
                            split=split,
                        )
                        self.instances.append(instance)
                    else:
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='They are equally novel',
                            split=split,
                        )
                        self.instances.append(instance)


class RealtiveValidityClassificationArgsvalidnovel(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='relative-validity-classification_argsvalidnovel',
            task_instruction='Argument conclusions are valid if they follow from the premise, meaning a logical inference links the premise to the conclusion. Given the conclusions below: Is conclusion A better than conclusion B in terms of validity?',
            dataset_names=['argsvalidnovel'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/argsvalidnovel/TaskB_'

        for split in ['train', 'dev', 'test']:
            with open(ds_path + split + '.csv', newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

                for elem in data[1:]:
                    if elem[4] == '-1':
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='Yes',
                            split=split,
                        )
                        self.instances.append(instance)
                    elif elem[4] == '1':
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='No',
                            split=split,
                        )
                        self.instances.append(instance)
                    else:
                        instance = Instance(
                            input='Premise: {}\nConclusion A: {}\nConclusion B: {}'.format(elem[1], elem[2], elem[3]),
                            output='They are equally valid',
                            split=split,
                        )
                        self.instances.append(instance)
