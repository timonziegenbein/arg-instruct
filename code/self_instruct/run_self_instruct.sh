#!/bin/bash
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python bootstrap_instructions.py --num_instructions_to_generate 60000
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python identify_clf_or_not.py
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python filter_is_clf_or_not.py
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python generate_instances.py
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python prepare_for_finetuning.py
HOME=/bigwork/nhwpziet HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python combine_ca_and_alpaca_data.py
