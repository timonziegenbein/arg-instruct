from task import Task, Instance
import pandas as pd
import numpy as np

ds_path = '/bigwork/nhwpstam/argpaca/data/enthymemes_student_essays/'


class DetectEnthymemesEnthymemesStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'detect-enthymemes_enthymemes-student-essays',
            'An enthymeme is defined here as any missing argumentative discourse unit (ADU) that would complete the logic of a written argument. Is there a problematic enthymematic gap at the position marked with "<mask>" in the following argument?',
            ['enthymemes-student-essays'], 
            is_clf=True,
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        for split in ['train', 'valid', 'test']:
            data = pd.read_excel(ds_path + 'gap_corpus_split.xlsx', sheet_name=split)
            if split == 'valid':
                split = 'dev'

            # labels: 0 for positive examples (actual gap), 1 for negative examples (random gap)
            # Gap (Indicator): 1 for cases in which no gap was created

            for index, row in data.iterrows():
                label = 'No'
                if row['labels'] == 0:
                    label = 'Yes'

                input_text = row['Para with Gap']
                if row['No Gap (Indicator)'] == 1:
                    input_text = str(row['Before Gap']) + ' <mask> ' + str(row['After Gap'])

                instance = Instance(
                    input=input_text,
                    output=label,
                    split=split
                )
                self.instances.append(instance)


class ReconstructEnthymemesEnthymemesStudentEssays(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'reconstruct-enthymemes_enthymemes-student-essays',
            'An enthymeme is defined here as any missing argumentative discourse unit (ADU) that would complete the logic of a written argument. Given the following argument with such a gap, generate a new ADU that fills the gap indicated with "<mask>".',
            ['enthymemes-student-essays'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        for split in ['train', 'valid', 'test']:
            data = pd.read_excel(ds_path + 'gap_corpus_split.xlsx', sheet_name=split)
            if split == 'valid':
                split = 'dev'

            # labels: 0 for positive examples (actual gap), 1 for negative examples (random gap)
            # Gap (Indicator): 1 for cases in which no gap was created

            data = data[(data['labels'] == 0) & (data['No Gap (Indicator)'] == 0)]  # correct gap and gap exists
            for index, row in data.iterrows():
                instance = Instance(
                    input=row['Para with Gap'],
                    output=row['Gap'],
                    split=split
                )
                self.instances.append(instance)
