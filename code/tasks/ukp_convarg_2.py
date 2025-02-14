import numpy as np
import random
import os
import sys
import glob
import pandas as pd
sys.path.insert(0, '..')
from task import Task, Instance

base_path = os.environ['ARGPACA_MAJA']

LABELID2LABEL = {
        'o8_1': 'Argument X has more details, information, facts, or examples / more reasons / better reasoning / goes deeper / is more specific',
        'o8_4': 'Argument X is balanced, objective, discusses several viewpoints / well-rounded / tackles flaws in opposing views',
        'o8_5': 'Argument X has better credibility / reliability / confidence',
        'o8_6': 'Explanation is highly topic-specific and addresses the content of Argument X in detail',
        'o9_1': 'Argument X is clear, crisp, to the point / well written',
        'o9_2': 'Argument X sticks to the topic',
        'o9_3': 'Argument X has provoking question / makes you think',
        'o9_4': 'Argument X is well thought of / has smart remarks / higher complexity',
        'o5_1': 'Argument X is attacking opponent / abusive',
        'o5_2': 'Argument X has language issues / bad grammar / uses humor, jokes, or sarcasm',
        'o5_3': 'Argument X is unclear, hard to follow',
        'o6_1': 'Argument X provides no facts / not enough support / not credible evidence / no clear explanation',
        'o6_2': 'Argument X has no reasoning / less or insufficient reasoning',
        'o6_3': 'Argument X uses irrelevant reasons / irrelevant information',
        'o7_1': 'Argument X is not an argument / is only opinion / is rant',
        'o7_2': 'Argument X is non-sense / has no logical sense / confusing',
        'o7_3': "Argument X is off topic / doesn't address the issue",
        'o7_4': 'Argument X is generally weak / vague',
}

FIXEDID2LABEL = {
        'o8_1': 'Argument A is more convincing because it has more details, information, facts, examples, reasons, better arguments, goes deeper or is more specific.',
        'o8_4': 'Argument A is more convincing because it is more balanced, objective, discusses several points of view, well-rounded or addresses flaws in opposing views.',
        'o8_5': 'Argument A is more convincing because it has better credibility, reliability or confidence.',
        'o8_6': 'Argument A is more convincing because of topic-specific reasons.',
        'o9_1': 'Argument A is more convincing because it is clear, crisp, to the point or well written.',
        'o9_2': 'Argument A is more convincing because it sticks to the topic.',
        'o9_3': 'Argument A is more convincing because it provokes thought.',
        'o9_4': 'Argument A is more convincing because it is well thought out, has smart remarks or is more complex.',
        'o5_1': 'Argument B is less convincing because it is attacking, abusive or disrespectful.',
        'o5_2': 'Argument B is less convincing because it has language issues, bad grammar, uses humor, jokes or sarcasm.',
        'o5_3': 'Argument B is less convincing because it is unclear, or hard to follow.',
        'o6_1': 'Argument B is less convincing because it provides no facts, not enough support, not credible evidence or no clear explanation.',
        'o6_2': 'Argument B is less convincing because it has no reasoning or less reasoning.',
        'o6_3': 'Argument B is less convincing because it uses irrelevant reasons or irrelevant information.',
        'o7_1': 'Argument B is less convincing because it is only opinion, or a rant.',
        'o7_2': 'Argument B is less convincing because it is non-sense, has no logical sense or is confusing.',
        'o7_3': "Argument B is less convincing because it is off topic or doesn't address the issue.",
        'o7_4': 'Argument B is less convincing because it is generally weak or vague.',
}


class ClassifyMoreConvincingArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-convincing-argument_ukp-convarg-2',
            task_instruction='Given the following two arguments (Argument A and Argument B), determine which of the two is more convincing.',
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.seed(42)
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            rand_pick = random.randint(0, 1)
            if rand_pick == 0:
                instance = Instance(
                    input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                    output='Argument A',
                    split=topic_split[row['issue']],
                )
                self.instances.append(instance)
            elif rand_pick == 1:
                instance = Instance(
                    input='Argument A: {}\nArgument B: {}'.format(row['less_conv_arg'], row['more_conv_arg']),
                    output='Argument B',
                    split=topic_split[row['issue']],
                )
                self.instances.append(instance)


class ClassifyMoreDetailsArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-details-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o8_1'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o8_1' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreBalancedArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-balanced-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o8_4'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o8_4' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreCredibleArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-credible-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o8_5'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o8_5' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreTopicSpecificArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-topic-specific-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o8_6'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o8_6' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreClearArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-clear-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o9_1'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o9_1' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreOnTopicArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-on-topic-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o9_2'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o9_2' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreProvokingArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-provoking-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o9_3'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o9_3' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyMoreSmartArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-more-smart-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o9_4'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o9_4' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessAttackingArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-attacking-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o5_1'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o5_1' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessLanguageIssuesArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-language-issues-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o5_2'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o5_2' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessUnclearArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-unclear-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o5_3'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o5_3' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessFactsArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-facts-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o6_1'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, **kwargs)
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o6_1' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessReasoningArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-reasoning-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o6_2'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        
    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o6_2' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyLessRelevantReasonsArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-less-relevant-reasons_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o6_3'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o6_3' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyNotAnArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-not-an-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o7_1'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o7_1' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyNonSenseArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-nonsense-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o7_2'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o7_2' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyOffTopicArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-off-topic-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' +
                                                                FIXEDID2LABEL['o7_3'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir = base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2 = glob.glob(ukp2_dir+'*')

        topics = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file = file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics = list(set(topics))
        np.random.shuffle(topics)
        train_topics = topics[:int(len(topics)*0.7)]
        dev_topics = topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics = topics[int(len(topics)*0.8):]
        topic_split = {}
        for topic in train_topics:
            topic_split[topic] = 'train'
        for topic in dev_topics:
            topic_split[topic] = 'dev'
        for topic in test_topics:
            topic_split[topic] = 'test'

        dfs = []
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df = pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance'] = file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2 = pd.concat(dfs)
        df_ukp2['less_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id'] = df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance = Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o7_3' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)


class ClassifyGenerallyWeakArgumentUKPConvArg2(Task):
    def __init__(self, **kwargs):
        super().__init__(
            task_name='classify-generally-weak-argument_ukp-convarg-2',
            task_instruction='Consider the two arguments below (Argument A and Argument B). Would you agree with the following statement?' + FIXEDID2LABEL['o7_4'],
            dataset_names=['ukp-convarg-2'],
            is_clf=True, 
            **kwargs
        )
        np.random.seed(42)

    def load_data(self):
        ukp2_dir=base_path + '/data/emnlp2016-empirical-convincingness/data/CSV-format/'
        rel_files_ukp2=glob.glob(ukp2_dir+'*')

        topics=[]
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_file=file.split('/')[-1].split('_')[0]
                topics.append(tmp_file)
        topics=list(set(topics))
        np.random.shuffle(topics)
        train_topics=topics[:int(len(topics)*0.7)]
        dev_topics=topics[int(len(topics)*0.7):int(len(topics)*0.8)]
        test_topics=topics[int(len(topics)*0.8):]
        topic_split={}
        for topic in train_topics:
            topic_split[topic]='train'
        for topic in dev_topics:
            topic_split[topic]='dev'
        for topic in test_topics:
            topic_split[topic]='test'

        dfs=[]
        for file in rel_files_ukp2:
            if 'LICENSE.txt' not in file:
                tmp_df=pd.read_csv(file, sep='\t', names=['pair_id', 'gold_label', 'more_conv_arg', 'less_conv_arg'])
                tmp_df['issue']=file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[0]
                tmp_df['stance']=file.split('/')[-1].split('.csv')[0].split('.xml')[0].split('_')[1]
                dfs.append(tmp_df)
        df_ukp2=pd.concat(dfs)
        df_ukp2['less_conv_arg_id']=df_ukp2['pair_id'].apply(lambda x: x.split('_')[0])
        df_ukp2['more_conv_arg_id']=df_ukp2['pair_id'].apply(lambda x: x.split('_')[1])

        for index, row in df_ukp2.iterrows():
            instance=Instance(
                input='Argument A: {}\nArgument B: {}'.format(row['more_conv_arg'], row['less_conv_arg']),
                output='Yes' if 'o7_4' in row['gold_label'] else 'No',
                split=topic_split[row['issue']],
            )
            self.instances.append(instance)

if __name__ == '__main__':
    task=ClassifyMoreConvincingArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreDetailsArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreBalancedArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreCredibleArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreTopicSpecificArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreClearArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreOnTopicArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreProvokingArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyMoreSmartArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessAttackingArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessLanguageIssuesArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessUnclearArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessFactsArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessReasoningArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyLessRelevantReasonsArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyNotAnArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyNonSenseArgumentUKPConvArg2()
    task.load_data()
    task=ClassifyOffTopicArgumentUKPConvArg2()
    task.load_data()
    task=ClaffiyGenerallyWeakArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyLessFactsArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyLessReasoningArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyLessRelevantReasonsArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyNotAnArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyNonSenseArgumentUKPConvArg2()
    task.load_data()
    task = ClassifyOffTopicArgumentUKPConvArg2()
    task.load_data()
    task = ClaffiyGenerallyWeakArgumentUKPConvArg2()
    task.load_data()
