import os
import json
import tqdm
import argparse
from collections import OrderedDict

import sys
sys.path.insert(0, '..')
base_path = os.environ['ARGPACA_MAJA']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        #required=True,
        default=base_path + "/data/self_instruct_llm_generations_maja/",
        help="The directory where the batch is stored.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    clf_or_not = []
    with open(os.path.join(args.batch_dir, f"is_clf_or_not_Meta-Llama-3-70B.jsonl")) as f:
        for line in f:
            clf_or_not.append(json.loads(line))
              
    print(f"Found {len(clf_or_not)} is_clf_or_not instances")
                   
    filtered = []
    for x in clf_or_not:
        if not (x['is_classification'] == 'Yes' and x['is_regression'] == 'Yes'):
            filtered.append(x)
                   
    print(f"Filtering lead to {len(filtered)} instances")
            
    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_filtered_Meta-Llama-3-70B.jsonl")
    with open(output_path, "w") as fout:
        for line in filtered:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")
