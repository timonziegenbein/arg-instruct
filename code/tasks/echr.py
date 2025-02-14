from task import Task, Instance
import numpy as np
import os
import json
import random
random.seed(92833)

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/echr/ECHR_Corpus.json'

with open(ds_path, 'r') as f:
    data = json.load(f)

split = list(range(len(data)))
test_split = random.sample(split, 8)
split = [i for i in split if not i in test_split]
dev_split = random.sample(split, 8)


class ArgumentClauseRecognitionEchr(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argument-clause-recognition_echr',
            # is the clause is annotated with any of the two argument types or with the non-argument type
            task_instruction='Does the given clause belong to an argument?',
            dataset_names=['echr'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        with open(ds_path, 'r') as f:
            data = json.load(f)

        for i in range(len(data)):
            split = 'train'
            if i in test_split:
                split = 'test'
            elif i in dev_split:
                split = 'dev'

            document = data[i]
            arguments = document['arguments']
            conclusion_ids = [x['conclusion'] for x in arguments]
            premise_ids = [item for sublist in [x['premises'] for x in arguments] for item in sublist]

            for clause in document['clauses']:
                label = 'No'
                if clause['_id'] in conclusion_ids or clause['_id'] in premise_ids:
                    label = 'Yes'

                instance = Instance(
                    input=document['text'][clause['start']:clause['end']-1],
                    output=label,
                    split=split,
                )
                self.instances.append(instance)


class ClauseRelationPredictionEchr(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='clause-relation-prediction_echr',
            task_instruction='Given a pair of argument clauses coming from the same document, predict if they are members of the same argument or not.',
            dataset_names=['echr'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        with open(ds_path, 'r') as f:
            data = json.load(f)

        for i in range(len(data)):
            split = 'train'
            if i in test_split:
                split = 'test'
            elif i in dev_split:
                split = 'dev'

            document = data[i]
            arguments = document['arguments']
            flat_arguments = []
            for arg in arguments:
                arg_list = [arg['conclusion']]
                arg_list.extend(arg['premises'])
                flat_arguments.append(arg_list)

            for j in range(len(document['clauses'])-5):
                for k in range(1, 6):  # "we only consider the pairs of arguments that are no more than five sentences apart"
                    c1 = document['clauses'][j]
                    c2 = document['clauses'][j+k]

                    label = 'Not members of the same argument'

                    for flat_arg in flat_arguments:
                        if c1['_id'] in flat_arg and c2['_id'] in flat_arg:
                            label = 'Members of the same argument'
                            break

                    instance = Instance(
                        input=f"Clause 1: {document['text'][c1['start']:c1['end']-1]}\nClause 2: {document['text'][c2['start']:c2['end']-1]}",
                        output=label,
                        split=split,
                    )
                    self.instances.append(instance)


class PremiseRecognitionEchr(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='premise-recognition_echr',
            task_instruction='Is the following argument clause a premise?',
            dataset_names=['echr'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        with open(ds_path, 'r') as f:
            data = json.load(f)

        for i in range(len(data)):
            split = 'train'
            if i in test_split:
                split = 'test'
            elif i in dev_split:
                split = 'dev'

            document = data[i]
            clauses = document['clauses']

            for argument in document['arguments']:
                # premises
                for p_id in argument['premises']:
                    clause = [c for c in clauses if c['_id'] == p_id][0]

                    instance = Instance(
                        input=document['text'][clause['start']:clause['end']-1],
                        output='Yes',
                        split=split,
                    )
                    self.instances.append(instance)

                # conclusion
                clause = [c for c in clauses if c['_id'] == argument['conclusion']][0]
                instance = Instance(
                    input=document['text'][clause['start']:clause['end']-1],
                    output='No',
                    split=split,
                )
                self.instances.append(instance)


class ConclusionRecognitionEchr(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='conlusion-recognition_echr',
            task_instruction='Is the following argument clause an argument conclusion?',
            dataset_names=['echr'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        with open(ds_path, 'r') as f:
            data = json.load(f)

        for i in range(len(data)):
            split = 'train'
            if i in test_split:
                split = 'test'
            elif i in dev_split:
                split = 'dev'

            document = data[i]
            clauses = document['clauses']

            for argument in document['arguments']:
                # premises
                for p_id in argument['premises']:
                    clause = [c for c in clauses if c['_id'] == p_id][0]

                    instance = Instance(
                        input=document['text'][clause['start']:clause['end']-1],
                        output='No',
                        split=split,
                    )
                    self.instances.append(instance)

                # conclusion
                clause = [c for c in clauses if c['_id'] == argument['conclusion']][0]
                instance = Instance(
                    input=document['text'][clause['start']:clause['end']-1],
                    output='Yes',
                    split=split,
                )
                self.instances.append(instance)
