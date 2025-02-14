import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

class PredictAgreementIacV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='predict-agreement_iac-v2',
            task_instruction='Given a topic, a quote (a statement) and a response to the quote, on a scale from -5 to 5, decide to what extent the response agrees or disagrees with the quote. -5 means strong disagreement, 0 means neutral, and 5 means strong agreement.',
            dataset_names=['iac-v2'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        df = pd.read_csv(base_path + '/data/iac-v2/fourforums.csv')
        topics = df['topic'].unique()
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]

        for i, row in df.iterrows():
            instance = Instance(
                input='Topic: ' + row['topic'][2:-1].capitalize() + '\n' + 'Quote: ' +
                row['quote'][2:-1] + '\n' + 'Response: ' + row['response'][2:-1],
                output=str(row['disagree_agree']),
                split='train' if row['topic'] in train_topics else 'dev' if row['topic'] in dev_topics else 'test'
            )
            self.instances.append(instance)


class PredictRespectIacV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='predict-respect_iac-v2',
            task_instruction='Given a topic, a quote (a statement) and a response to the quote, on a scale from -5 to 5, decide to what extent the response is attacking or respectful. -5 means strong attacking, 0 means neutral, and 5 means strong respectful.',
            dataset_names=['iac-v2'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        df = pd.read_csv(base_path + '/data/iac-v2/fourforums.csv')
        topics = df['topic'].unique()
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]

        for i, row in df.iterrows():
            instance = Instance(
                input='Topic: ' + row['topic'][2:-1].capitalize() + '\n' + 'Quote: ' +
                row['quote'][2:-1] + '\n' + 'Response: ' + row['response'][2:-1],
                output=str(row['attacking_respectful']),
                split='train' if row['topic'] in train_topics else 'dev' if row['topic'] in dev_topics else 'test'
            )
            self.instances.append(instance)


class PredictFactualityIacV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='predict-factuality_iac-v2',
            task_instruction='Given a topic, a quote (a statement) and a response to the quote, on a scale from -5 to 5, decide to what extent the response is emotional or factual. -5 means strong emotional, 0 means neutral, and 5 means strong factual.',
            dataset_names=['iac-v2'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        df = pd.read_csv(base_path + '/data/iac-v2/fourforums.csv')
        topics = df['topic'].unique()
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]

        for i, row in df.iterrows():
            instance = Instance(
                input='Topic: ' + row['topic'][2:-1].capitalize() + '\n' + 'Quote: ' +
                row['quote'][2:-1] + '\n' + 'Response: ' + row['response'][2:-1],
                output=str(row['emotion_fact']),
                split='train' if row['topic'] in train_topics else 'dev' if row['topic'] in dev_topics else 'test'
            )
            self.instances.append(instance)


class PredictNiceIacV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='predict-nice_iac-v2',
            task_instruction='Given a topic, a quote (a statement) and a response to the quote, on a scale from -5 to 5, decide to what extent the response is nasty or nice. -5 means strong nasty, 0 means neutral, and 5 means strong nice.',
            dataset_names=['iac-v2'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        df = pd.read_csv(base_path + '/data/iac-v2/fourforums.csv')
        topics = df['topic'].unique()
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]

        for i, row in df.iterrows():
            instance = Instance(
                input='Topic: ' + row['topic'][2:-1].capitalize() + '\n' + 'Quote: ' +
                row['quote'][2:-1] + '\n' + 'Response: ' + row['response'][2:-1],
                output=str(row['nasty_nice']),
                split='train' if row['topic'] in train_topics else 'dev' if row['topic'] in dev_topics else 'test'
            )
            self.instances.append(instance)


class PredictSarcasmIacV2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='predict-sarcasm_iac-v2',
            task_instruction='Given a topic, a quote (a statement) and a response to the quote, decide to what degree (between 0 and 100) the response is sarcastic. 0 means not sarcasic and 100 means very sarcastic.',
            dataset_names=['iac-v2'],
            is_reg=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        df = pd.read_csv(base_path + '/data/iac-v2/fourforums.csv')
        topics = df['topic'].unique()
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]

        for i, row in df.iterrows():
            instance = Instance(
                input='Topic: ' + row['topic'][2:-1].capitalize() + '\n' + 'Quote: ' +
                row['quote'][2:-1] + '\n' + 'Response: ' + row['response'][2:-1],
                output=str(row['sarcasm_yes']*100),
                split='train' if row['topic'] in train_topics else 'dev' if row['topic'] in dev_topics else 'test'
            )
            self.instances.append(instance)


if __name__ == '__main__':
    task = PredictAgreementIacV2()
    task.load_data()
    task = PredictRespectIacV2()
    task.load_data()
    task = PredictFactualityIacV2()
    task.load_data()
    task = PredictNiceIacV2()
    task.load_data()
    task = PredictSarcasmIacV2()
