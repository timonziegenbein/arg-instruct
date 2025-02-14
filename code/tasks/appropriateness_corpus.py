from task import Task, Instance
import pandas as pd
import numpy as np
import os

base_path = os.environ['ARGPACA_MAJA']

class InappropriatenessDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'inappropriateness-detection_appropriateness-corpus',
            'An argument is appropriate if the used language supports the creation of credibility and emotions as well as if it is proportional to its topic. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument is Appropriate or Inappropriate.',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Inappropriate' if row['Inappropriateness'] == 1 else 'Appropriate',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class ToxicEmotionsDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'toxic-emotions-detection_appropriateness-corpus',
            'An argument has toxic emotions if the emotions appealed to are deceptive or their intensities do not provide room for critical evaluation of the topic by the reader. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument is Toxic or Not Toxic.',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Toxic' if row['Toxic Emotions'] == 1 else 'Not Toxic',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class MissingCommitmentDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'missing-commitment-detection_appropriateness-corpus',
            'An argument is missing commitment if the topic is not taken seriously or openness other’s arguments is absent. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument Lacks Commitment or Does Not Lack Commitment',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Lacks Commitment' if row['Missing Commitment'] == 1 else 'Does Not Lack Commitment',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class MissingIntelligibilityDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'missing-intelligibility-detection_appropriateness-corpus',
            'An argument is not intelligible if its meaning is unclear or irrelevant to the topic or if its reasoning is not understandable. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument Lacks Intelligibility or Does Not Lack Intelligibility',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Lacks Intelligibility' if row['Missing Intelligibility'] == 1 else 'Does Not Lack Intelligibility',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class OtherInappropriatenessDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'other-inappropriateness-detection_appropriateness-corpus',
            'An argument is inappropriate if it contains severe orthographic errors or for reasons that are not Toxic Emotions, Missing Commitment or Missing Intelligibility. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument is Inappropriate Due to Other Reasons or Not Inappropriate Due to Other Reasons',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Inappropriate Due to Other Reasons' if row['Other Reasons'] == 1 else 'Not Inappropriate Due to Other Reasons',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class ExcessiveIntensityDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'excessive-intensity-detection_appropriateness-corpus',
            'An argument has excessive intensity if the emotions appealed to by are unnecessarily strong for the discussed issue. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument has Excessive Intensity or Does Not Have Excessive Intensity',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Has Excessive Intensity' if row['Excessive Intensity'] == 1 else 'Does Not Have Excessive Intensity',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class EmotionalDeceptionDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'emotional-deception-detection_appropriateness-corpus',
            'An argument is emotionally deceptive if the emotions appealed to are used as deceptive tricks to win, derail, or end the discussion. Decide whether the argument is Emotionally Deceptive or Not Emotionally Deceptive',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Is Emotionally Deceptive' if row['Emotional Deception'] == 1 else 'Is Not Emotionally Deceptive',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class MissingSeriousnessDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'missing-seriousness-detection_appropriateness-corpus',
            'An argument is missing seriousness if it is either trolling others by suggesting (explicitly or implicitly) that the issue is not worthy of being discussed or does not contribute meaningfully to the discussion. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument Lacks Seriousness or Does Not Lack Seriousness',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                if row['Inappropriateness'] == 1:
                    instance = Instance(
                        'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                        'Lacks Seriousness' if row['Missing Commitment'] == 1 else 'Does Not Lack Seriousness',
                        'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                    )
                    self.instances.append(instance)


class MissingOpennessDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'missing-openness-detection_appropriateness-corpus',
            'An argument is missing openness if it displays an unwillingness to consider arguments with opposing viewpoints and does not assess the arguments on their merits but simply rejects them out of hand. Decide whether the argument Lacks Openness or Does Not Lack Openness',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                if row['Inappropriateness'] == 1:
                    instance = Instance(
                        'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                        'Lacks Openness' if row['Missing Commitment'] == 1 else 'Does Not Lack Openness',
                        'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                    )
                    self.instances.append(instance)


class UnclearMeaningDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'unclear-meaning-detection_appropriateness-corpus',
            'An argument has unclear meaning if its content is vague, ambiguous, or implicit, such that it remains unclear what is being said about the issue (it could also be an unrelated issue). Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument has Unclear Meaning or Does Not Have Unclear Meaning',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Has Unclear Meaning' if row['Unclear Meaning'] == 1 else 'Does Not Have Unclear Meaning',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class MissingRelevanceDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'missing-relevance-detection_appropriateness-corpus',
            'An argument is missing relevance if it does not discuss the issue, but derails the discussion implicitly towards a related issue or shifts completely towards a different issue. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument Lacks Relevance or Does Not Lack Relevance',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Lacks Relevance' if row['Missing Relevance'] == 1 else 'Does Not Lack Relevance',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class ConfusingReasoningDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'confusing-reasoning-detection_appropriateness-corpus',
            'An argument has confusing reasoning if its components (claims and premises) seem not to be connected logically. Decide whether the argument has Confusing Reasoning or Does Not Have Confusing Reasoning',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Has Confusing Reasoning' if row['Confusing Reasoning'] == 1 else 'Does Not Have Confusing Reasoning',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class DetrimentalOrthographyDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'detrimental-orthography-detection_appropriateness-corpus',
            'An argument is detrimental orthography if it has serious spelling and/or grammatical errors, negatively affecting its readability. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument has Detrimental Orthography or Does Not Have Detrimental Orthography',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                instance = Instance(
                    'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                    'Has Detrimental Orthography' if row['Detrimental Orthography'] == 1 else 'Does Not Have Detrimental Orthography',
                    'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                )
                self.instances.append(instance)


class ReasonUnclassifiedDetectionAppropriatenessCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            'reason-unclassified-detection_appropriateness-corpus',
            'An argument is unclassified if it is inappropriate because of reasons not covered by Detrimental Orthography, Toxic Emotions, Missing Commitment and Missing Intelligibility. Given the following argument and the topic of the debate the argument appeared in. Decide whether the argument is Unclassified or Not Unclassified',
            ['appropriateness-corpus'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/appropriateness-corpus/'
        df = pd.read_csv(ds_path + 'appropriateness_corpus_conservative_w_folds.csv')
        for split in ['TRAIN', 'VALID', 'TEST']:
            for i, row in df[df['fold0.0'] == split].iterrows():
                if row['Inappropriateness'] == 1:
                    instance = Instance(
                        'Topic: ' + row['issue'] + '\nArgument: ' + row['post_text'],
                        'Is Unclassified' if row["Reason Unclassified"] == 1 else 'Is Not Unclassified',
                        'train' if split == 'TRAIN' else 'dev' if split == 'VALID' else 'test',
                    )
                    self.instances.append(instance)


if __name__ == '__main__':
    task = InappropriatenessDetectionAppropriatenessCorpus()
    print(task.instances[0].apply_template())
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
    print('+'*50)
    batch = task.get_batch(split='dev')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
    print('+'*50)
    batch = task.get_batch(split='test')
    for instance in next(batch):
        print(instance)
