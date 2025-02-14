from task import Task, Instance
import numpy as np
import pandas as pd
import csv
import os

base_path = os.environ['ARGPACA_MAJA']
ds_path = base_path + '/data/ibm-rank-30k/arg_quality_rank_30k.csv'
data = pd.read_csv(ds_path)


class QualityAssessmentIbmRank30k(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='quality-assessment_ibm-rank-30k',
            task_instruction='How high is the likelihood (0 - 1) that you would recommend your friend to use the following argument as is in a speech supporting/contesting the topic, regardless of your personal opinion?',
            dataset_names=['ibm-rank-30k'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        for i, row in data.iterrows():
            instance = Instance(
                input=row['argument'],
                output=str(row['WA']),
                split=row['set'],
            )
            self.instances.append(instance)


class StancePredictionIbmRank30k(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='stance-prediction_ibm-rank-30k',
            task_instruction='Mark the stance of the argument towards the topic as pro or con.',
            dataset_names=['ibm-rank-30k'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        for i, row in data.iterrows():
            label = 'pro'
            if row['stance_WA'] == -1:
                label = 'con'
            instance = Instance(
                input=f"Argument: {row['argument']}; Topic: {row['topic']}",
                output=label,
                split=row['set'],
            )
            self.instances.append(instance)
