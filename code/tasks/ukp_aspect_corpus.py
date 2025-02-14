import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

TOPIC_SPLITS = {
    'train': ['Wind power', 'Nanotechnology', '3d printing', 'Cryptocurrency', 'Virtual reality', 'Gene editing', 'Public surveillance', 'Genetic diagnosis', 'Geoengineering', 'Gmo', 'Organ donation', 'Recycling', 'Offshore drilling', 'Robotic surgery', 'Cloud storing', 'Electric cars', 'Stem cell research'],
    'dev': ['Hydrogen fuel cells', 'Electronic voting', 'Drones', 'Solar energy'],
    'test': ['Tissue engineering', 'Big data', 'Fracking', 'Social networks', 'Net neutrality', 'Hydroelectric dams', 'Internet of things']
}


class ArgumentSimilarityUKPAspectCorpus(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='argument-similarity_ukp-aspect-corpus',
            task_instruction='''Decide, whether the two sentences are similar or not, based on the given topic. Choose one of the following options: Different Topic/Can’t decide (DTORCD): Either one or both of the sentences belong to a topic different than the given one, or you can’t understand one or both sentences. If you choose this option, you need to very briefly explain, why you chose it (e.g.“The second sentence is not grammatical”, “The first sentence is from a different topic” etc.). No Similarity (NS): The two arguments belong to the same topic, but they don’t show any similarity, i.e. they speak aboutcompletely different aspects of the topic. Some Similarity (SS): The two arguments belong to the same topic, showing semantic similarity on a few aspects, but thecentral message is rather different, or one argument is way less specific than the other. High Similarity (HS): The two arguments belong to the same topic, and they speak about the same aspect, e.g. using different words.''',
            dataset_names=['ukp-aspect-corpus'], 
            is_clf=True,
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        ds_path = base_path + '/data/ukp-aspect/UKP_ASPECT.tsv'
        df = pd.read_csv(ds_path, sep='\t')

        for split in ['train', 'dev', 'test']:
            for topic in TOPIC_SPLITS[split]:
                topic_df = df[df['topic'] == topic]
                for _, row in topic_df.iterrows():
                    instance = Instance(
                        input='Topic: ' + row['topic'] + '\n' + 'Sentence 1: ' +
                        row['sentence_1'] + '\n' + 'Sentence 2: ' + row['sentence_2'],
                        output='Different Topic/Can’t decide (DTORCD)' if row['label'] == 'DTORCD' else 'No Similarity (NS)' if row[
                            'label'] == 'NS' else 'Some Similarity (SS)' if row['label'] == 'SS' else 'High Similarity (HS)',
                        split=split
                    )
                    self.instances.append(instance)


if __name__ == '__main__':
    task = ArgumentSimilarityUKPAspectCorpus()
    task.load_data()
