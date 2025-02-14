#!/bin/bash   
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m generate_instruction generate_instruction_following_data
