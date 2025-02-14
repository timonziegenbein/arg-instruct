from datasets import load_dataset
from datasets import concatenate_datasets

ds_general = load_dataset("alexl83/AlpacaDataCleaned", split='train')
# create validation set from general dataset
ds_general = ds_general.train_test_split(test_size=0.1, shuffle=True, seed=42)

ds_ca = load_dataset("timonziegenbein/ca52k")
ds_ca_gen = load_dataset("timonziegenbein/ca_gen52k")

ds_general = ds_general.map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'source': 'alpaca'})
ds_ca = ds_ca.map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'source': 'ca'})
ds_ca_gen = ds_ca_gen.map(lambda x: {'instruction': x['instruction'], 'input': x['input'], 'output': x['output'], 'source': 'ca_gen'})

# merge datasets
ds_train = concatenate_datasets([ds_general['train'], ds_ca['train']])
ds_train = concatenate_datasets([ds_train, ds_ca_gen['train']])

ds_valid = concatenate_datasets([ds_general['test'], ds_ca['validation']])
ds_valid = concatenate_datasets([ds_valid, ds_ca_gen['validation']])

# sample 52000 instances for training and 5200 instances for validation
ds_train = ds_train.shuffle(seed=42).select(range(52000))
ds_valid = ds_valid.shuffle(seed=42).select(range(5200))

# save datasets
ds_train = ds_train.to_pandas()
ds_valid = ds_valid.to_pandas()

ds_train.to_json("/mnt/home/tziegenb/argpaca/data/train/ca_all_merged52k/train.json", orient='records', lines=True)
ds_valid.to_json("/mnt/home/tziegenb/argpaca/data/train/ca_all_merged52k/valid.json", orient='records', lines=True)
