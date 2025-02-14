import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance


base_path = os.environ['ARGPACA_MAJA']

class ClaimRevisionImprovementClaimRevisionPairCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='claim-revision-improvement_claim-revision-pair-corpus',
            task_instruction='''Compare the given two versions of the same claim and determine which one is better (Claim 1 or Claim 2).''',
            dataset_names=['claim-revision-pair-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/claim-rev-pairs/eacl21_base.csv'
        df = pd.read_csv(ds_path)

        instances = [i for i in range(len(df))]
        np.random.seed(42)
        np.random.shuffle(instances)
        train_instances = set(instances[:int(len(instances)*0.7)])
        dev_instances = set(instances[int(len(instances)*0.7):int(len(instances)*0.8)])

        for i, row in df.iterrows():
            instance = Instance(
                input='Claim 1: ' + row['v1_text'] + '\n' + 'Claim 2: ' + row['v2_text'],
                output='Claim 1' if row['label'] == 0 else 'Claim 2',
                split='train' if i in train_instances else 'dev' if i in dev_instances else 'test'
            )
            self.instances.append(instance)


class SubotimalClaimDetectionClaimRevisionPairCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='suboptimal-claim-detection_claim-revision-pair-corpus',
            task_instruction='''Given an argumentative claim, decide whether it is in need of further revision or can be considered to be phrased more or less optimally (Suboptimal or Optimal).''',
            dataset_names=['claim-revision-pair-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/claim-rev-pairs/acl23_revised.csv'
        df = pd.read_csv(ds_path)

        for i, row in df.iterrows():
            instance = Instance(
                input=row['claim_text'],
                output='Suboptimal' if row['label'] != 0 else 'Optimal',
                split=row['data_split']
            )
            self.instances.append(instance)


class ClaimImprovementSuggestionsClaimRevisionPairCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='claim-improvement-suggestions_claim-revision-pair-corpus',
            task_instruction='''Given an argumentative claim, select the type required type of quality improvement from the defined set (Typo/grammar correction, Clarification, Link correction/addition) that should be improved when revising the claim.''',
            dataset_names=['claim-revision-pair-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/claim-rev-pairs/acl23_revised.csv'
        df = pd.read_csv(ds_path, sep=',')
        labels = ['Typo or grammar correction', 'Clarified claim', 'Corrected or added links', 'Clarified argument']

        for i, row in df.iterrows():
            if row['revision_type'] in labels:
                if 'Clarified' in row['revision_type']:
                    row['revision_type'] = 'Clarification'
                elif 'link' in row['revision_type']:
                    row['revision_type'] = 'Link correction/addition'
                elif 'Typo' in row['revision_type']:
                    row['revision_type'] = 'Typo or grammar correction'
                instance = Instance(
                    input=row['claim_text'],
                    output=row['revision_type'],
                    split=row['data_split']
                )
                self.instances.append(instance)


class ClaimOptimizationClaimRevisionPairCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='claim-optimization_claim-revision-pair-corpus',
            task_instruction='''Given as input an argumentative claim, potentially along with context information on the debate, rewrite the claim such that it improves in terms of text quality and/or argument quality, and preserves the meaning as far as possible.''',
            dataset_names=['claim-revision-pair-corpus'],
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/claim-rev-pairs/acl23_revised.csv'
        df = pd.read_csv(ds_path, sep=',')

        for claim_id, group in df.groupby('claim_id'):
            group = group.sort_values(by='revision_id')
            for i in range(len(group)-1):
                instance = Instance(
                    input=group.iloc[i]['claim_text'],
                    output=group.iloc[i+1]['claim_text'],
                    split=group.iloc[i+1]['data_split']
                )
                self.instances.append(instance)


if __name__ == '__main__':
    #task = ClaimRevisionImprovementClaimRevisionPairCorpus()
    #task = SubotimalClaimDetectionClaimRevisionPairCorpus()
    task = ClaimImprovementSuggestionsClaimRevisionPairCorpus(from_cache=False)
    #task = ClaimOptimizationClaimRevisionPairCorpus()
    for i in range(10):
        print(task.instances[i].input)
        print(task.instances[i].output)
        print(task.instances[i].split)
        print()
