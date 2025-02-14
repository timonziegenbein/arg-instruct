import numpy as np
import pandas as pd
import os
from rouge import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import rankdata
from scipy.stats import wilcoxon

base_path = os.environ['ARGPACA_MAJA']

TEST_DATASET_NAMES = [
    'argument-annotated-essays-2', 'qt30', 'f1000rd', 'iac-v2', 'ibm-rank-30k', 'arguana-counterargs-corpus', 'aspect-controlled-argument-generation', 'debate-sum', 'webis-conclugen-21'
]

VALID_DATASET_NAMES = [
    'aaugwd', 'icle-argument-strength', 'argumentation-synthesis'
]


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# adapted the flowing from Squad v1.1 evaluation, without removing the articles.


if __name__ == "__main__":
    prediction_files = {
        #'01_gemma_base': base_path + '/data/inference/predictions_outline_gemma-2-9b_ca_balanced.jsonl',
        #'02_gemma_CAseed': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-checkpoint-2216_ca_balanced.jsonl',
        ##'03_gemma_CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_gen_new-checkpoint-2246_ca_balanced.jsonl',
        ##'04_gemma_Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-alpaca52k-checkpoint-1138_ca_balanced.jsonl',
        #'05_gemma_CAseed+CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+ca_gen52k_new-checkpoint-2222_ca_balanced.jsonl',
        #'07_gemma_CAseed+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+alpaca52k-checkpoint-1123_ca_balanced.jsonl',
        ##'06_gemma_CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_gen+alpaca52k-checkpoint-1333_ca_balanced.jsonl',
        #'08_gemma_CAseed+CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_combined-checkpoint-3624_ca_balanced.jsonl',

        '09_gemma-it-ours': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct_ca_balanced.jsonl',
        '10_gemma-it-ours_CAseed': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k-checkpoint-5537_ca_balanced.jsonl',
        #'11_gemma-it-ours_CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_gen-checkpoint-4560_ca_balanced.jsonl',
        #'12_gemma_it-ours-Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-alpaca52k-checkpoint-2260_ca_balanced.jsonl',
        '13_gemma-it-ours_CAseed+CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+ca52k_gen-checkpoint-1119_ca_balanced.jsonl',
        #'15_gemma-it-ours_CAseed+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+alpaca52k-checkpoint-2246_ca_balanced.jsonl',
        #'14_gemma-it-ours_CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_gen+alpaca52k-checkpoint-3399_ca_balanced.jsonl',
        #'16_gemma-it-ours_CAseed+CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_combined-checkpoint-906_ca_balanced.jsonl',

        #'17_gemma_it': base_path + '/data/inference/predictions_outline_gemma-2-9b-it_ca_balanced.jsonl',


        #'ArgInstruct': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+ca52k_gen-checkpoint-1119_ca_full-test.jsonl',
    }
    data = {
        'task': [],
        'task_type': [],
        'task_is_ca': [],
        'dataset': [],
        'approach': [],
        'split': [],
        'metric_values': []
    }

    min_pred_len = min([len(pd.read_json(prediction_files[prediction_file], lines=True)) for prediction_file in prediction_files])
    print(f'Using {min_pred_len} predictions per approach')

    for prediction_file in prediction_files:
        print(f'Processing {prediction_file}')
        temp_df = pd.read_json(prediction_files[prediction_file], lines=True).drop_duplicates(subset=['id'])
        temp_df = temp_df[:min_pred_len]
        if 'superni' in prediction_file:
            temp_df['task'] = temp_df['id'].apply(lambda x: '_'.join(x.split('_')[0:1]))
        else:
            temp_df['task'] = temp_df['id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
        temp_df['id'] = temp_df['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        print(f'Loaded {len(temp_df)} predictions for {len(temp_df["task"].unique())} tasks')
        # print num predictions per task
        for task in temp_df['task'].unique():
            print(f'{task}: {len(temp_df[temp_df["task"] == task])}')
        for task in temp_df['task'].unique():
            task_df = temp_df[temp_df['id'].str.startswith(task)]
            task_type = 'reg' if task_df['is_reg'].iloc[0] else 'clf' if task_df['is_clf'].iloc[0] else 'gen'
            task_is_ca = 'superni' if 'superni' in prediction_file else 'ca'
            try:
                dataset = task_df['id'].iloc[0].split('_')[1]
            except:
                dataset = 'ERROR'
            approach = prediction_file.split('/')[-1].split('.')[0]
            tmp_metric_values = []
            if task_type == 'gen':
                for i, row in task_df.iterrows():
                    tmp_metric_values.append(default_rouge_scorer.score(str(row['reference']), str(row['prediction'][:int(len(row['reference'])*1.25)]))['rougeL'].fmeasure)
            elif task_type == 'clf':
                for i, row in task_df.iterrows():
                    if row['reference'] == row['prediction']:
                        tmp_metric_values.append(1)
                    else:
                        tmp_metric_values.append(0)
            elif task_type == 'reg':
                if task_type == 'reg':
                    float_predictions = [float(x) for x in task_df['prediction']]
                    float_references = [float(x) for x in task_df['reference']]
                    for pred, ref in zip(float_predictions, float_references):
                        tmp_metric_values.append(-1*abs(pred - ref))

            data['task'].append(task)
            data['task_type'].append(task_type)
            data['task_is_ca'].append(task_is_ca)
            data['dataset'].append(dataset)
            data['split'].append('test' if dataset in TEST_DATASET_NAMES else 'train')#if dataset not in VALID_DATASET_NAMES else 'valid')
            data['approach'].append(approach)
            data['metric_values'].append(tmp_metric_values)

    df = pd.DataFrame(data)

    # comoute significance for each task
    for approach_1 in df['approach'].unique():
        for approach_2 in df['approach'].unique():
            if approach_1 == approach_2:
                continue
            #print(f'Comparing {approach_1} and {approach_2} for task {task}')
            num_significant_train = 0
            num_significant_test = 0
            #for task in df['task'].unique():
            for task_type in df['task_type'].unique():
                for split in df['split'].unique():
                    #task_df = df[df['task'] == task]
                    task_df = df[(df['task_type'] == task_type) & (df['split'] == split)]
                    approach_1_metric_values = task_df[task_df['approach'] == approach_1]['metric_values'].tolist()
                    approach_2_metric_values = task_df[task_df['approach'] == approach_2]['metric_values'].tolist()
                    # flatten lists
                    approach_1_metric_values = [item for sublist in approach_1_metric_values for item in sublist]
                    approach_2_metric_values = [item for sublist in approach_2_metric_values for item in sublist]
                    wilcoxon_result = wilcoxon(approach_1_metric_values, approach_2_metric_values, alternative='less', method='exact')
                    if wilcoxon_result.pvalue < 0.05:
                        print(f'{approach_1} vs {approach_2} for task_type {task_type} and split {split} is significant')
                    else:
                        print(f'{approach_1} vs {approach_2} for task_type {task_type} and split {split} is NOT significant')
            #percentage_significant_train = num_significant_train / len(df['task'][df['split'] == 'train'].unique())
            #percentage_significant_test = num_significant_test / len(df['task'][df['split'] == 'test'].unique())
            #print(f'{approach_1} vs {approach_2}: {percentage_significant_train} significant on train, {percentage_significant_test} significant on test')

