import string
import numpy as np
import pandas as pd
import json
import argparse
import os
from rouge import rouge_scorer
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import rankdata

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
        #'alpaca3': '../inference/predictions_default.jsonl',
        #'baseline_random': '../inference/predictions_random.jsonl',
        #'alpaca3_superni': base_path + '/data/inference/predictions_outline_llama3-8b-instruct_superni.jsonl',

        #'alpaca3_ca': base_path + '/data/inference/predictions_outline_llama3-8b-instruct_ca.jsonl',
        #'alpaca_it_ca': base_path + '/data/inference/predictions_outline_alpaca_ca_balanced.jsonl',
        #'llama3_it_ca': base_path + '/data/inference/predictions_outline_Meta-Llama-3-8B-Instruct_ca_balanced.jsonl',
        #'mistral_it_ca': base_path + '/data/inference/predictions_outline_Mistral-7B-Instruct-v0.3_ca_balanced.jsonl',
        #'mistral_arg_ca_52k': base_path + '/data/inference/predictions_outline_MistralArg-52k_ca_balanced.jsonl',
        #'mistral_arg_ca_104k': base_path + '/data/inference/predictions_outline_MistralArg-104k-500step_ca_balanced.jsonl',
        #'mistral_arg_ca_840k': base_path + '/data/inference/predictions_outline_MistralArg-840k-500step_ca_balanced.jsonl',
        #'mistral_arg_ca_840k_4000': base_path + '/data/inference/predictions_outline_MistralArg-840k-4000step_ca_balanced.jsonl',
        #'mistral_arg_ca_52k_2000_e7': base_path + '/data/inference/predictions_outline_MistralArg-52k-2000step-e7lr_ca_balanced.jsonl',
        #'mistral_52k_ca_0500steps': base_path + '/data/inference/predictions_outline_MistralArg-52k-2000step-checkpoint_000500_ca_balanced.jsonl',
        #'mistral_52k_ca_1000steps': base_path + '/data/inference/predictions_outline_MistralArg-52k-2000step-checkpoint_001000_ca_balanced.jsonl',
        #'mistral_52k_ca_1500steps': base_path + '/data/inference/predictions_outline_MistralArg-52k-2000step-checkpoint_001500_ca_balanced.jsonl',
        #'mistral_52k_ca_2000steps': base_path + '/data/inference/predictions_outline_MistralArg-52k-2000step-checkpoint_002000_ca_balanced.jsonl',
        #'mistral_arg_ca_52k_4000': base_path + '/data/inference/predictions_outline_MistralArg-52k-4000step_ca_balanced.jsonl',
        #'ministral_it_ca': base_path + '/data/inference/predictions_outline_Ministral-8B-Instruct-2410_ca_balanced.jsonl',
        ##'argpaca_ca': base_path + '/data/inference/predictions_outline_argpaca-8b_ca_balanced.jsonl',
        #'gpt-4o-mini_ca': base_path + '/data/inference/predictions_outline_gpt-4o-mini_ca_balanced.jsonl',
        #'gemma_052k_gen_1500steps': base_path + '/data/inference/predictions_outline_models-checkpoint-1500_ca_balanced.jsonl',
        #'gemma_052k_all_0500steps': base_path + '/data/inference/predictions_outline_models-checkpoint-500_ca_balanced.jsonl',
        #'gemma_156k_all_1290steps': base_path + '/data/inference/predictions_outline_models-checkpoint-1290_ca_balanced.jsonl',
        #'gemma_156k_all_3888steps': base_path + '/data/inference/predictions_outline_models-checkpoint-3888_ca_balanced.jsonl',
        #'gemma_ca52k_ca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-checkpoint-848_ca_balanced.jsonl',
        #'gemma_ca52k_ca_2': base_path + '/data/inference/predictions_outline_models-checkpoint-848_ca_balanced.jsonl',
        #'gemma_ca52k_ca_0275': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-hp-checkpoint-275_ca_balanced.jsonl',
        #'gemma_ca52k_ca_0263': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-hp-checkpoint-263_ca_balanced.jsonl',
        #'gemma_ca52k_ca_1500': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-hp-checkpoint-1500_ca_balanced.jsonl',
        #'gemma_ca52k_ca_2000': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-hp-checkpoint-2000_ca_balanced.jsonl',
        #'2_gemma_CAseed+extraTraining': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+ca_gen52k-checkpoint-1110_ca_balanced.jsonl',
        #'3_gemma_CAgen_round2': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_gen_new_round2-checkpoint-1123_ca_balanced.jsonl',
        #'4_gemma_CAseed+CAgen_round2': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+ca52k_gen_new_round2-checkpoint-1111_ca_balanced.jsonl',

        #'alpaca3_superni': base_path + '/data/inference/predictions_outline_llama3-8b-instruct_superni.jsonl',
        #'alpaca_superni': base_path + '/data/inference/predictions_outline_alpaca_superni.jsonl',
        #'gemma_superni': base_path + '/data/inference/predictions_outline_gemma-2-9b-it_superni.jsonl',
        #'llama3_superni': base_path + '/data/inference/predictions_outline_Meta-Llama-3-8B-Instruct_superni.jsonl',
        #'mistral_superni': base_path + '/data/inference/predictions_outline_Mistral-7B-Instruct-v0.3_superni.jsonl',


        #'00_random_ca': base_path + '/data/inference/predictions_random_ca_balanced.jsonl',
        #'00_majority_ca': base_path + '/data/inference/predictions_majority_ca_balanced.jsonl',

        #'01_gemma_base': base_path + '/data/inference/predictions_outline_gemma-2-9b_ca_balanced.jsonl',
        #'02_gemma_CAseed': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k-checkpoint-2216_ca_balanced.jsonl',
        #'03_gemma_CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_gen_new-checkpoint-2246_ca_balanced.jsonl',
        #'04_gemma_Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-alpaca52k-checkpoint-1138_ca_balanced.jsonl',
        #'05_gemma_CAseed+CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+ca_gen52k_new-checkpoint-2222_ca_balanced.jsonl',
        #'06_gemma_CAseed+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k+alpaca52k-checkpoint-1123_ca_balanced.jsonl',
        #'07_gemma_CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_gen+alpaca52k-checkpoint-1333_ca_balanced.jsonl',
        #'08_gemma_CAseed+CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_combined-checkpoint-3624_ca_balanced.jsonl',

        #'09_gemma-it-ours': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct_ca_balanced.jsonl',
        #'10_gemma-it-ours_CAseed': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k-checkpoint-5537_ca_balanced.jsonl',
        #'11_gemma-it-ours_CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_gen-checkpoint-4560_ca_balanced.jsonl',
        ##'12_gemma_it-ours-Alpaca': SAME AS gemma-it-ours
        #'13_gemma-it-ours_CAseed+CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+ca52k_gen-checkpoint-1119_ca_balanced.jsonl',
        #'14_gemma-it-ours_CAseed+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+alpaca52k-checkpoint-2246_ca_balanced.jsonl',
        #'15_gemma-it-ours_CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_gen+alpaca52k-checkpoint-3399_ca_balanced.jsonl',
        #'16_gemma-it-ours_CAseed+CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k_combined-checkpoint-906_ca_balanced.jsonl',

        #'17_gemma_it': base_path + '/data/inference/predictions_outline_gemma-2-9b-it_ca_balanced.jsonl',

        '00_gemma-it-ours_CAseed+CAgen': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct-ca52k+ca52k_gen-checkpoint-1119_superni_balanced.jsonl',
        '01_gemma_CAseed+CAgen+Alpaca': base_path + '/data/inference/predictions_outline_gemma-2-9b-ca52k_combined-checkpoint-3624_superni_balanced.jsonl',
        '02_gemma-it-ours': base_path + '/data/inference/predictions_outline_gemma-2-9b-instruct_superni_balanced.jsonl',
        '03_gemma_base': base_path + '/data/inference/predictions_outline_gemma-2-9b_superni_balanced.jsonl',
    }
    data = {
        'task': [],
        'task_type': [],
        'task_is_ca': [],
        'dataset': [],
        'approach': [],
        'split': [],
        'micro_f1': [],
        'macro_f1': [],
        #'mase': [],
        'rougeL': [],
        'exact_matches': [],
    }

    min_pred_len = min([len(pd.read_json(prediction_files[prediction_file], lines=True)) for prediction_file in prediction_files])
    print(f'Using {min_pred_len} predictions per approach')

    # get mean of regression predictions from random Approach
    mean_df = pd.read_json(base_path + '/data/inference/predictions_mean_ca_balanced.jsonl', lines=True)
    mean_df = mean_df[:min_pred_len]
    mean_df['task'] = mean_df['id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    mean_df['id'] = mean_df['id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    mean_df = mean_df[mean_df['is_reg']]
    mean_df['prediction'] = mean_df['prediction'].apply(lambda x: float(x))
    mean_df = mean_df.groupby('task').agg({'prediction': 'mean'}).reset_index()
    mean_dict = dict(zip(mean_df['task'], mean_df['prediction']))

    for prediction_file in prediction_files:
        print(f'Processing {prediction_file}')
        temp_df = pd.read_json(prediction_files[prediction_file], lines=True).drop_duplicates(subset=['id'])
        temp_df = temp_df[:min_pred_len]
        if 'superni' in prediction_files[prediction_file]:
            temp_df['task'] = temp_df['id'].apply(lambda x: x.split('_')[0])
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
            task_is_ca = 'superni' if 'superni' in prediction_files[prediction_file] else 'ca'
            try:
                dataset = task_df['id'].iloc[0].split('_')[1]
            except:
                dataset = 'ERROR'
            approach = prediction_file.split('/')[-1].split('.')[0]
            if task_type == 'gen':
                exact_matches = 0
                rougeL = 0
                for i, row in task_df.iterrows():
                    if row['prediction'] == row['reference']:
                        exact_matches += 1
                    rougeL += default_rouge_scorer.score(str(row['reference']), str(row['prediction'][:int(len(row['reference'])*1.25)]))['rougeL'].fmeasure
                exact_matches /= len(task_df)
                rougeL /= len(task_df)
            else:
                exact_matches, rougeL = None, None
            if task_type == 'clf':
                #print('Approach:', approach)
                #print('Task:', task)
                #print('Reference distribution:', task_df['reference'].value_counts())
                #print('Prediction distribution:', task_df['prediction'].value_counts())
                _, _, micro_f1, _ = precision_recall_fscore_support(task_df['reference'], task_df['prediction'], average='micro')
                _, _, macro_f1, _ = precision_recall_fscore_support(task_df['reference'], task_df['prediction'], average='macro')
                #print('Micro-F1:', micro_f1)
                #print('Macro-F1:', macro_f1)
                #print('-'*50)
            else:
                micro_f1, macro_f1 = None, None
            try:
                if task_type == 'reg':
                    float_predictions = [float(x) for x in task_df['prediction']]
                    float_references = [float(x) for x in task_df['reference']]
                    mase = np.mean(abs(np.array(float_references) - np.array(float_predictions)))/np.mean(abs(np.array(float_references) - np.mean(mean_dict[task])))
                else:
                    mase = None
            except:
                mase = 'ERROR'
            data['task'].append(task)
            data['task_type'].append(task_type)
            data['task_is_ca'].append(task_is_ca)
            data['dataset'].append(dataset)
            data['split'].append('test' if dataset in TEST_DATASET_NAMES else 'train')#if dataset not in VALID_DATASET_NAMES else 'valid')
            data['approach'].append(approach)
            data['micro_f1'].append(micro_f1)
            data['macro_f1'].append(macro_f1)
            data['rougeL'].append(rougeL)
            data['exact_matches'].append(exact_matches)

    # print len of all lists in data
    for key in data:
        print(f'{key}: {len(data[key])}')

    # calculate approach ranking for each task and metric
    ranks = {}
    for task in set(data['task']):
        for metric in ['micro_f1', 'macro_f1', 'rougeL', 'exact_matches']:
            task_df = pd.DataFrame(data)
            task_df = task_df[task_df['task'] == task]
            task_df['rank'] = rankdata(task_df[metric], method='dense')
            for i, row in task_df.iterrows():
                if row['approach'] not in ranks:
                    ranks[row['approach']] = {}
                if row['task'] not in ranks[row['approach']]:
                    ranks[row['approach']][row['task']] = {}
                if metric != 'mase':
                    ranks[row['approach']][row['task']][metric] = max(task_df['rank']) - row['rank'] + 1
                else:
                    ranks[row['approach']][row['task']][metric] = row['rank']

    df = pd.DataFrame(data)
    df['macro_f1_rank'] = df.apply(lambda x: ranks[x['approach']][x['task']]['macro_f1'] if x['task_type'] == 'clf' else None, axis=1)
    df['micro_f1_rank'] = df.apply(lambda x: ranks[x['approach']][x['task']]['micro_f1'] if x['task_type'] == 'clf' else None, axis=1)
    #df['mase_rank'] = df.apply(lambda x: ranks[x['approach']][x['task']]['mase'] if x['task_type'] == 'reg' else None, axis=1)
    df['rougeL_rank'] = df.apply(lambda x: ranks[x['approach']][x['task']]['rougeL'] if x['task_type'] == 'gen' else None, axis=1)
    df['exact_matches_rank'] = df.apply(lambda x: ranks[x['approach']][x['task']]['exact_matches'] if x['task_type'] == 'gen' else None, axis=1)
    df.to_csv('evaluation_results_balanced_superni.csv', index=False)

    # calculate average values for each approach 
    pivot_df = df.pivot_table(index='approach', values=['micro_f1', 'micro_f1_rank', 'rougeL', 'rougeL_rank'], aggfunc='mean', columns='split')
    # undo the multiindex
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df[['micro_f1_train', 'micro_f1_rank_train', 'rougeL_train', 'rougeL_rank_train']]
    # calc mean of ranks
    pivot_df['mean_rank_train'] = pivot_df[['micro_f1_rank_train', 'rougeL_rank_train']].mean(axis=1)
    # round to 2 decimal places
    pivot_df.to_csv('evaluation_results_balanced_pivot_task_superni.csv')

    # drop all columns that are not needed
    df = df.drop(columns=['task', 'task_type', 'task_is_ca', 'macro_f1', 'exact_matches', 'macro_f1_rank', 'exact_matches_rank'])
    dataset_df = df.groupby(['approach', 'dataset', 'split']).mean().reset_index()

    ranks = {}
    for task in set(dataset_df['dataset']):
        for metric in ['micro_f1_rank','rougeL_rank']:
            task_df = dataset_df.copy()
            task_df = task_df[task_df['dataset'] == task]
            task_df = task_df.sort_values(by=metric, ascending=False)
            task_df['rank'] = rankdata(task_df[metric], method='dense')
            for i, row in task_df.iterrows():
                if row['approach'] not in ranks:
                    ranks[row['approach']] = {}
                if row['dataset'] not in ranks[row['approach']]:
                    ranks[row['approach']][row['dataset']] = {}
                ranks[row['approach']][row['dataset']][metric] = row['rank']

    dataset_df['micro_f1_rank'] = dataset_df.apply(lambda x: ranks[x['approach']][x['dataset']]['micro_f1_rank'], axis=1)
    dataset_df['rougeL_rank'] = dataset_df.apply(lambda x: ranks[x['approach']][x['dataset']]['rougeL_rank'], axis=1)
    # drop dataset columns
    dataset_df = dataset_df.drop(columns=['dataset'])
    # pivot the dataset_df
    pivot_df = dataset_df.pivot_table(index='approach', values=['micro_f1', 'micro_f1_rank', 'rougeL', 'rougeL_rank'], aggfunc='mean', columns='split')
    # undo the multiindex
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df[['micro_f1_train', 'micro_f1_rank_train', 'rougeL_train', 'rougeL_rank_train']]
    # create rank of ranks
    #pivot_df['micro_f1_rank_train'] = rankdata(pivot_df['micro_f1_rank_train'], method='dense')
    #pivot_df['rougeL_rank_train'] = rankdata(pivot_df['rougeL_rank_train'], method='dense')
    # calc mean of ranks
    pivot_df['mean_rank_train'] = pivot_df[['micro_f1_rank_train',  'rougeL_rank_train']].mean(axis=1)
    # round to 2 decimal places
    #pivot_df = pivot_df.round(2)
    pivot_df.to_csv('evaluation_results_balanced_pivot_dataset_superni.csv')
