import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']
DS_PATH = base_path + '/data/argkp/kpm_data/'
TEST_PATH = base_path + '/data/argkp/test_data/'


class KeyPointMatchingArgKP(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='key-point-matching_argkp',
            task_instruction='A small set of talking points, termed key points can be used to form a concise summary from a large collection of arguments on a given topic. Does the following key point match the given argument?',
            dataset_names=['argkp'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        for subset in ['train', 'dev', 'test']:
            # adapted from 'track_1_kp_matching.py'
            arguments_file = os.path.join(DS_PATH, f"arguments_{subset}.csv")
            key_points_file = os.path.join(DS_PATH, f"key_points_{subset}.csv")
            labels_file = os.path.join(DS_PATH, f"labels_{subset}.csv")
            if subset == "test":
                arguments_file = os.path.join(TEST_PATH, f"arguments_{subset}.csv")
                key_points_file = os.path.join(TEST_PATH, f"key_points_{subset}.csv")
                labels_file = os.path.join(TEST_PATH, f"labels_{subset}.csv")

            arguments_df = pd.read_csv(arguments_file)
            key_points_df = pd.read_csv(key_points_file)
            labels_file_df = pd.read_csv(labels_file)

            for index, row in arguments_df.iterrows():
                arg_id, argument, topic, stance = row
                key_points = key_points_df[(key_points_df["stance"] == stance) & (key_points_df["topic"] == topic)]
                for index2, row2 in key_points.iterrows():
                    kp_id, key_point, _, _ = row2

                    if len(labels_file_df[(labels_file_df['arg_id'] == arg_id) & (labels_file_df['key_point_id'] == kp_id)]) > 0:
                        label = labels_file_df[(labels_file_df['arg_id'] == arg_id) & (
                            labels_file_df['key_point_id'] == kp_id)]['label'].values[0]

                        instance = Instance(
                            input='Key point: {}\nArgument: {}'.format(argument, key_point),
                            output="Yes" if label == 1 else "No",
                            split=subset,
                        )
                        self.instances.append(instance)


class KeyPointGenerationArgKP(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='key-point-generation_argkp',
            task_instruction='A small set of talking points, termed key points can be used to form a concise summary from a large collection of arguments on a given topic. Generate multiple key points on the given topic and provide the stance of each key point towards the topic.',
            dataset_names=['argkp'],
            **kwargs)
        np.random.seed(42)

    def load_data(self):
        for subset in ['train', 'dev', 'test']:
            # adapted from 'track_1_kp_matching.py'
            arguments_file = os.path.join(DS_PATH, f"arguments_{subset}.csv")
            key_points_file = os.path.join(DS_PATH, f"key_points_{subset}.csv")
            labels_file = os.path.join(DS_PATH, f"labels_{subset}.csv")
            if subset == "test":
                arguments_file = os.path.join(TEST_PATH, f"arguments_{subset}.csv")
                key_points_file = os.path.join(TEST_PATH, f"key_points_{subset}.csv")
                labels_file = os.path.join(TEST_PATH, f"labels_{subset}.csv")

            arguments_df = pd.read_csv(arguments_file)
            key_points_df = pd.read_csv(key_points_file)

            for desc, group in arguments_df.groupby(["topic"]):
                topic = desc[0]
                key_points = key_points_df[(key_points_df["topic"] == topic)]

                output = ''
                for _, row in key_points.iterrows():
                    kp = row.iloc[1]
                    stance = "pro" if row.iloc[3] == 1 else "con"
                    output += "{} ({})\n".format(kp, stance)

                instance = Instance(
                    input='Topic: {}'.format(topic),
                    output=output[:-1],  # remove last line break
                    split=subset,
                )
                self.instances.append(instance)


if __name__ == '__main__':
    # task = KeyPointMatchingArgKP()
    # print(task.instances[0].apply_template())
    # batch = task.get_batch(split='train')
    # for instance in next(batch):
    #    print(instance)
    # print('-'*50)
    # for instance in next(batch):
    #    print(instance)
    task = KeyPointGenerationArgKP()
    task.load_data()
    batch = task.get_batch(split='train')
    for instance in next(batch):
        print(instance)
    print('-'*50)
    for instance in next(batch):
        print(instance)
