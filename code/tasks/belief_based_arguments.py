import pandas as pd
import numpy as np
import ast
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

issue_map = {
    'Con': 1,
    'Not Saying': 0,
    'No Opinion': 2,
    'Undecided': 0,
    'Pro': 3,
}

reverse_issue_map = {v: k for k, v in issue_map.items()}


big_issues = ['Abortion',
              'Affirmative Action',
              'Animal Rights',
              'Barack Obama',
              'Border Fence',
              'Capitalism',
              'Civil Unions',
              'Death Penalty',
              'Drug Legalization',
              'Electoral College',
              'Environmental Protection',
              'Estate Tax',
              'European Union',
              'Euthanasia',
              'Federal Reserve',
              'Flat Tax',
              'Free Trade',
              'Gay Marriage',
              'Global Warming Exists',
              'Globalization',
              'Gold Standard',
              'Gun Rights',
              'Homeschooling',
              'Internet Censorship',
              'Iran-Iraq War',
              'Labor Union',
              'Legalized Prostitution',
              'Medicaid & Medicare',
              'Medical Marijuana',
              'Military Intervention',
              'Minimum Wage',
              'National Health Care',
              'National Retail Sales Tax',
              'Occupy Movement',
              'Progressive Tax',
              'Racial Profiling',
              'Redistribution',
              'Smoking Ban',
              'Social Programs',
              'Social Security',
              'Socialism',
              'Stimulus Spending',
              'Term Limits',
              'Torture',
              'United Nations',
              'War in Afghanistan',
              'War on Terror',
              'Welfare']


class StancePredictionBeliefBasedArguments(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='stance-prediction_belief-based-arguments',
            task_instruction='Predict the stance (pro, con, or unknown) of the user on the corresponding big issue from the text of the claim.',
            dataset_names=['belief-based-arguments'],
            is_clf=True,
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/belief-based-argument-generation/preprocessed_data'

        for split in ['train', 'valid', 'test']:
            df = pd.read_csv(f'{ds_path}/{split}_df.csv')
            for index, row in df.iterrows():
                instance = Instance(
                    input=row['opinion_txt'],
                    output=row['stance'] if row['stance'] in ['pro', 'con'] else 'unknown',
                    split=split if split != 'valid' else 'dev'
                )
                self.instances.append(instance)


class BeliefBasedClaimGenerationBeliefBasedArguments(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='belief-based-claim-generation_belief-based-arguments',
            task_instruction='Given a controversial topic and a set of beliefs, generate an argumentative claim tailored to the beliefs.',
            dataset_names=['belief-based-arguments'],
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/belief-based-argument-generation/preprocessed_data'

        for split in ['train', 'valid', 'test']:
            df = pd.read_csv(f'{ds_path}/{split}_df.csv')
            # read list from big_issues columns
            df['big_issues'] = df['big_issues'].apply(lambda x: ast.literal_eval(x.replace(' ', ',')))
            for index, row in df.iterrows():
                instance = Instance(
                    input='Topic: {}\n\nBeliefs:\n'.format(row['topic']) + '\n'.join(
                        ['{}: {}'.format(issue, reverse_issue_map[int(b)]) for b, issue in zip(row['big_issues'], big_issues)]),
                    output=row['opinion_txt'],
                    split=split if split != 'valid' else 'dev'
                )
                self.instances.append(instance)


if __name__ == '__main__':
    task = StancePredictionBeliefBasedArguments(from_cache=False)
    for i in range(10):
        print(task.instances[i].input)
        print(task.instances[i].output)
        print(task.instances[i].split)
        print()
