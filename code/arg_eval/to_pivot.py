import pandas as pd
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv('evaluation_results_balanced.csv')
    # drop all columns that are not needed
    df = df.drop(columns=['task', 'task_type', 'task_is_ca', 'micro_f1', 'exact_matches', 'micro_f1_rank', 'exact_matches_rank'])
    dataset_df = df.groupby(['approach', 'dataset']).mean().reset_index()
    # calculate approach ranking for each dataset and metric
    ranks = {}
    for task in set(dataset_df['dataset']):
        for metric in ['macro_f1_rank', 'mase_rank', 'rougeL_rank']:
            task_df = dataset_df.copy()
            task_df = task_df[task_df['dataset'] == task]
            task_df = task_df.sort_values(by=metric, ascending=False)
            task_df['rank'] = np.arange(1, len(task_df)+1)
            for i, row in task_df.iterrows():
                if row['approach'] not in ranks:
                    ranks[row['approach']] = {}
                if row['dataset'] not in ranks[row['approach']]:
                    ranks[row['approach']][row['dataset']] = {}
                if metric != 'mase':
                    ranks[row['approach']][row['dataset']][metric] = row['rank']
                else:
                    ranks[row['approach']][row['dataset']][metric] = len(task_df) - row['rank'] + 1
    dataset_df['macro_f1_rank'] = dataset_df.apply(lambda x: ranks[x['approach']][x['dataset']]['macro_f1_rank'], axis=1)
    dataset_df['mase_rank'] = dataset_df.apply(lambda x: ranks[x['approach']][x['dataset']]['mase_rank'], axis=1)
    dataset_df['rougeL_rank'] = dataset_df.apply(lambda x: ranks[x['approach']][x['dataset']]['rougeL_rank'], axis=1)
    # drop dataset columns
    dataset_df = dataset_df.drop(columns=['dataset'])
    # pivot the dataset_df
    pivot_df = dataset_df.pivot_table(index='approach', values=['macro_f1', 'macro_f1_rank', 'mase', 'mase_rank', 'rougeL', 'rougeL_rank'], aggfunc='mean')
    # calc mean of ranks
    pivot_df['mean_rank'] = pivot_df[['macro_f1_rank', 'mase_rank', 'rougeL_rank']].mean(axis=1)
    # round to 2 decimal places
    pivot_df = pivot_df.round(2)
    print(pivot_df)
