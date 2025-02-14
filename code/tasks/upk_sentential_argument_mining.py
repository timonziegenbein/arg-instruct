from task import Task, Instance
import numpy as np
import pandas as pd
import os
import csv

base_path = os.environ['ARGPACA_MAJA']

class ArgumentIdentificationUKPSententialArgumentMining(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argument-identification_UKP-sentential-argument-mining',
            task_instruction='Given a sentence and a topic, classify the sentence as a “supporting argument” or “opposing argument” if it includes a relevant reason for supporting or opposing the topic, or as a “non-argument” if it does not include a reason or is not relevant to the topic.',
            dataset_names=['UKP-sentential-argument-mining'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/UKP_sentential_argument_mining/data/'
        for file in ['abortion.tsv', 'cloning.tsv', 'death_penalty.tsv', 'gun_control.tsv', 'marijuana_legalization.tsv', 'minimum_wage.tsv', 'nuclear_energy.tsv']:
            data = pd.read_csv(ds_path + file, sep='\t')
            for i, row in data.iterrows():
                if row['annotation'] == 'NoArgument':
                    label = 'non-argument'
                elif row['annotation'] == 'Argument_for':
                    label = 'supporting argument'
                elif row['annotation'] == 'Argument_against':
                    label = 'opposing argument'

                split = row['set']
                if split == 'val':
                    split = 'dev'
                instance = Instance(
                    input=f"Sentence: {row['sentence']}\nTopic: {row['topic']}",
                    output=label,
                    split=split,
                )
                self.instances.append(instance)

        test_data = pd.read_csv(ds_path + 'school_uniforms.tsv', sep='\t', quoting=csv.QUOTE_NONE)
        for i, row in test_data.iterrows():
            if row['set'] == 'test':
                if row['annotation'] == 'NoArgument':
                    label = 'non-argument'
                elif row['annotation'] == 'Argument_for':
                    label = 'supporting argument'
                elif row['annotation'] == 'Argument_against':
                    label = 'opposing argument'

                instance = Instance(
                    input=f"Sentence: {row['sentence']}\nTopic: {row['topic']}",
                    output=label,
                    split='test',
                )
                self.instances.append(instance)
