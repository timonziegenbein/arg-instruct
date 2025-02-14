from task import Task, Instance
import numpy as np
import pandas as pd
import json
import os
import csv

base_path = os.environ['ARGPACA_MAJA']

topics = ['abortion', 'cloning', 'death_penalty', 'gun_control',
          'marijuana_legalization', 'minimum_wage', 'nuclear_energy', 'school_uniforms']
test_topics = ['gun_control', 'school_uniforms']
dev_topics = ['death_penalty']

ds_path = base_path + '/data/aspect-controlled-argument-generation/'
sources_generated = ['generated_arguments/common-crawl-en/', 'generated_arguments/redditcomments-en/']
sources_orig = ['cc_training_data_1.1/', 'reddit_training_data_1.1/']


class AspectControlledArgumentGenerationAspectControlledArgumentGeneration(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='aspect-controlled-argument-generation_aspect-controlled-argument-generation',
            task_instruction='Generate an argument that follows the given topic, stance and argument aspect.',
            dataset_names=['aspect-controlled-argument-generation'], **kwargs)
        np.random.seed(42)

    def load_data(self):
        np.random.seed(92384)
        for topic in topics:

            counter = 0
            split = 'train'
            if topic in dev_topics:
                split = 'dev'
            elif topic in test_topics:
                split = 'test'

            control_codes = []
            for sg in sources_generated:
                path = os.path.join(ds_path, sg, topic, 'generation_data/')
                with open(path + 'control_codes.jsonl', 'r') as f:
                    control_codes.extend([json.loads(l) for l in f.readlines()])

            data = []
            for sg in sources_generated:
                path = os.path.join(ds_path, sg, topic, 'generation_data/')

                con_data = []
                with open(path + 'generated_training_data_CON_0.jsonl', 'r') as f:
                    con_data.extend([json.loads(l) for l in f.readlines()])
                    data.extend(con_data)
                pro_data = []
                with open(path + 'generated_training_data_PRO_0.jsonl', 'r') as f:
                    pro_data.extend([json.loads(l) for l in f.readlines()])
                    data.extend(pro_data)

            orig_data = []
            for so in sources_orig:
                orig_data = []
                with open(os.path.join(ds_path, so, topic, 'processed/merged.jsonl'), 'r') as f:
                    orig_data.extend([json.loads(l) for l in f.readlines()])
                    data.extend(orig_data)

            for cc in control_codes:
                data_topic = [elem for elem in data if 'topic' in elem.keys() and cc['topic'] == elem['topic']]
                data_stance = [elem for elem in data_topic if
                               ('_against' in elem['stance'] and cc['stance'] == 'CON') or
                               ('_for' in elem['stance'] and cc['stance'] == 'PRO') or
                               (elem['stance'] == cc['stance'])
                               ]
                data_aspect = [elem for elem in data_stance if
                               ('aspect' in elem.keys() and elem['aspect'] == cc['aspect']) or
                               ('aspect_string' in elem.keys() and cc['aspect'] in elem['aspect_string'])
                               ]

                if len(data_aspect) > 0:
                    instance = Instance(
                        input=f"Topic: {cc['topic']}\nStance: {cc['stance']}\nAspect: {cc['aspect']}",
                        output=str(np.random.choice(data_aspect)['sent']),
                        split=split
                    )
                    self.instances.append(instance)
