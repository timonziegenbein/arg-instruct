import numpy as np
import csv
import os
import json
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/argument_annotated_essays_2/'


def get_splits():
    with open(ds_path + 'train-test-split.csv', newline='') as f:
        split_data = list(csv.reader(f))
        split_dict = {}
        for line in split_data[1:]:
            key, value = line[0].split(';')
            split_dict[key] = value.replace('"', '')
    return split_dict


def get_prompts():
    with open(ds_path + 'prompts.csv', encoding='unicode_escape') as f:
        prompt_data = list(csv.reader(f))
        prompt_dict = {}
        for line in prompt_data[1:]:
            key, value = line[0].split(';')
            for i in range(1, len(line)):
                value += line[i]
            prompt_dict[key] = value.replace('"', '')
        return prompt_dict


def get_data():
    data_dict = {}
    files = os.listdir(ds_path + 'brat-project-final/')

    txt_files = [f for f in files if '.txt' in f]
    for txt_file in txt_files:
        ann_file = txt_file[:-4] + '.ann'
        with open(ds_path + 'brat-project-final/' + txt_file) as f:
            essay = f.read()

        with open(ds_path + 'brat-project-final/' + ann_file) as tsv:
            annos = list(csv.reader(tsv, delimiter="\t"))

        data_dict[txt_file[:-4]] = (essay, annos)
    return data_dict


class IdentifyingArgumentComponentsArgumentAnnotatedEssays2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='identifying-argument-components_argument-annotated-essays-2',
            task_instruction='Identify all argumentative text spans in the following essay.',
            dataset_names=['argument-annotated-essays-2'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        split_dict = get_splits()
        data_dict = get_data()

        for eid in data_dict.keys():
            text, annos = data_dict[eid]
            arg_annos = [a for a in annos if a[0][0] == 'T']
            arg_texts = [a[2] for a in arg_annos]

            instance = Instance(
                input='Essay: {}'.format(text),
                output='\n'.join(arg_texts),
                split=split_dict[eid].lower(),
            )
            self.instances.append(instance)


class ClassifyingArgumentComponentsArgumentAnnotatedEssays2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classifying-argument-components_argument-annotated-essays-2',
            task_instruction='Given the following essay as context, and a list of argumentative components extracted from the essay. Label each argumentative component as "major claim", "claim", or "premise".',
            dataset_names=['argument-annotated-essays-2'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        split_dict = get_splits()
        data_dict = get_data()

        for eid in data_dict.keys():
            text, annos = data_dict[eid]
            arg_annos = [a for a in annos if a[0][0] == 'T']
            arg_texts = [a[2] for a in arg_annos]

            output = ''
            filtered_annos = [a[2] for a in annos if 'MajorClaim' in a[1]]
            if len(filtered_annos) > 0:
                output += 'Major claims:\n' + '\n'.join(filtered_annos)
            filtered_annos = [a[2] for a in annos if 'Claim' in a[1]]
            if len(filtered_annos) > 0:
                output += '\n\nClaims:\n' + '\n'.join(filtered_annos)
            filtered_annos = [a[2] for a in annos if 'Premise' in a[1]]
            if len(filtered_annos) > 0:
                output += '\n\nPremises:\n' + '\n'.join(filtered_annos)

            instance = Instance(
                input='Essay: {}\nArgumentative components: {}'.format(text, '\n'.join(arg_texts)),
                output=output,
                split=split_dict[eid].lower(),
            )
            self.instances.append(instance)


class IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='identifying-argumentative-relations_argument-annotated-essays-2',
            task_instruction='Are the following two argumentative components (AC1 and AC2), which come from the same essay, connected in the sense that AC1 either supports or attacks AC2? (Yes or No)',
            dataset_names=['argument-annotated-essays-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        split_dict = get_splits()
        data_dict = get_data()
        self.instances = []

        for eid in data_dict.keys():
            text, annos = data_dict[eid]
            # match annos based on line number (paragraph)
            paragraph_annos = [[] for _ in range(len(text.split('\n')))]
            line_ranges = []
            start = 0
            for i, line in enumerate(text.split('\n')):
                line_ranges.append((start, start + len(line)))
                start += len(line) + 1
            for anno in annos:
                if anno[0][0] == 'T':
                    char_start, char_end = int(anno[1].split()[1]), int(anno[1].split()[2])
                    for i, (start, end) in enumerate(line_ranges):
                        if char_start >= start and char_start < end:
                            paragraph_annos[i].append(anno)
                            break
            print(paragraph_annos)

            for p_annos in paragraph_annos:
                arg_annos = [a for a in p_annos if a[0][0] == 'T']
                arg_premises = [a for a in p_annos if a[0][0] == 'T' and 'Premise' in a[1]]
                rel_annos = [a[1].split() for a in annos if a[0][0] == 'R']
                rel_binary = [(a[1][5:], a[2][5:]) for a in rel_annos]

                for i, arg1 in enumerate(arg_annos):
                    for j, arg2 in enumerate(arg_annos):
                        if i != j:
                            output = "No"
                            for rel in rel_binary:
                                if (arg1[0] == rel[0] and arg2[0] == rel[1]):
                                    output = "Yes"

                            instance = Instance(
                                input='Argumentative Component 1 (AC1): "{}", Argumentative Component 2 (AC2): "{}"'.format(
                                    arg1[2], arg2[2]),
                                output=output,
                                split=split_dict[eid].lower(),
                            )
                            self.instances.append(instance)



class StanceRecognitionArgumentAnnotatedEssays2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='stance-recognition_argument-annotated-essays-2',
            task_instruction='Does the following argumentative component "attack" or "support" the target argumentative component?',
            dataset_names=['argument-annotated-essays-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        split_dict = get_splits()
        data_dict = get_data()

        for eid in data_dict.keys():
            text, annos = data_dict[eid]
            arg_annos = [a for a in annos if a[0][0] == 'T']
            rel_annos = [a[1].split() for a in annos if a[0][0] == 'R']

            for rel_anno in rel_annos:
                out_anno = [a for a in arg_annos if a[0] == rel_anno[1][5:]][0]
                in_anno = [a for a in arg_annos if a[0] == rel_anno[2][5:]][0]

                instance = Instance(
                    input='Argumentative component: "{}", target argumentative component: "{}"'.format(
                        out_anno[2], in_anno[2]),
                    output=rel_anno[0][:-1],  # "attack" or "support"
                    split=split_dict[eid].lower(),
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = IdentifyingArgumentativeRelationsArgumentAnnotatedEssays2(from_cache=False)
    print(len([i for i in task.instances if i.split == 'train']))
    print(len([i for i in task.instances if i.split == 'dev']))
    print(len([i for i in task.instances if i.split == 'test']))
    print(len([i for i in task.instances if i.split == 'train' and i.output == 'Yes']))
    print(len([i for i in task.instances if i.split == 'train' and i.output == 'No']))
    print(len([i for i in task.instances if i.split == 'dev' and i.output == 'Yes']))
    print(len([i for i in task.instances if i.split == 'dev' and i.output == 'No']))
    print(len([i for i in task.instances if i.split == 'test' and i.output == 'Yes']))
    print(len([i for i in task.instances if i.split == 'test' and i.output == 'No']))
