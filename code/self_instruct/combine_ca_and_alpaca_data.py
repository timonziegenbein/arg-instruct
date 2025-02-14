import os
import json
import random
import argparse

random.seed(147)
base_path = os.environ['ARGPACA_MAJA']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=base_path + "/data/self_instruct_llm_generations_maja/finetuning/",
        type=str,
        help="The output dir to save the cleaned version of the generated instances, so that it can be used for finetuning."
    )


if __name__ == "__main__":
    args = parse_args()

    with open(os.path.join(args.output_dir, "sampled_generated_instances_52004.json"), "r") as f:
        arg_data = json.load(f)

    with open("../stanford_alpaca/alpaca_data.json", "r") as f:
        alpaca_data = json.load(f)

    print(f'Loaded {len(arg_data)} argpaca instances and {len(alpaca_data)} alpaca instances.')

    data_all = alpaca_data
    data_all.extend(arg_data)
    random.shuffle(data_all)

    print(f"Save {len(data_all)} instances in total")

    with open(os.path.join(args.output_dir, f"sampled_generated_instances_and_alpaca_{len(data_all)}.json"), "w") as fout:
        fout.write(json.dumps(data_all))
