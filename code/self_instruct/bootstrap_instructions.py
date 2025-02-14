# adapted from https://github.com/yizhongw/self-instruct/blob/main/self_instruct/bootstrap_instructions.py

import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
#from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer

import sys
sys.path.insert(0, '..')

from samplers import BalancedInstanceSampler, InstructionSampler, SuperNiInstructionSampler

random.seed(42)

base_path = os.environ['ARGPACA_MAJA']

def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of computational argumentation classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of computational argumentation tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_llm_response(response):
    raw_instructions = re.split(r"\n\d+\s?\. ", response)
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 200: # Argpaca edit; original: 150
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        # required=True,
        default=base_path + "/data/self_instruct_llm_generations_maja/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instruction_sampler = InstructionSampler()
    superni_instruction_sampler = SuperNiInstructionSampler()
    from llm_requests import make_requests
    # seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    # if args.use_clf_seed_tasks_only:
    #    seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = instruction_sampler.get_all(split="train")
    superni_instructions = superni_instruction_sampler.get_all()
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions,
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                prompt_instructions += random.sample(seed_instructions,
                                                     args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                batch_inputs.append(prompt)
            results = make_requests(batch_inputs, presence_penalty=2.0, max_tokens=1024)
            instructions = []
            all_metadata = []
            for result in results:
                result["response"] = post_process_llm_response(result["response"])
                new_instructions = result["response"]
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            prefix = '''Does the following task fall into the field of computational argumentation?\n\n'''
            # randomly sample a 10 seed instructions
            argumentative_seed_instructions = random.sample(seed_instructions, 10)
            non_argumentative_seed_instructions = random.sample(superni_instructions, 10)
            argumentative_check_instructions = {x: 'Yes' if x in argumentative_seed_instructions else 'No' for x in argumentative_seed_instructions+non_argumentative_seed_instructions}
            for inst, is_arg in sorted(argumentative_check_instructions.items(), key=lambda x: random.random()):
                prefix += f"Task: {inst}\nIs it argumentative? {is_arg}\n\n"
            prefix += '''Task:'''

            prompts = [prefix + " " + d.strip() + "\n" + "Is it argumentative?" for d in instructions]
            arg_results = make_requests(prompts, max_tokens=3, greedy=True)

            for inst, metadata, arg_result in zip(instructions, all_metadata, arg_results):
                
                # is is CS related?
                if arg_result["response"] is not None:
                    is_arg = arg_result["response"].split("Task")[0].split("\n")[0].strip() # extract Yes/No
                    if is_arg in ["Yes", "yes", "YES"]:
                
                        # is it unsimilar to previous ones?
                        rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in seed_instructions + machine_instructions]
                        if not max(rouge_scores) > 0.7:
                            
                            all_instructions = seed_instructions + machine_instructions
                            most_similar_instructions = {
                                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                            }

                            machine_instructions.append(inst)
                            fout.write(json.dumps({
                                "instruction": inst,
                                "most_similar": most_similar_instructions,
                                "avg_similarity_score": float(np.mean(rouge_scores)),
                                "metadata": metadata,
                                "request_idx": request_idx
                            }) + "\n")
                            progress_bar.update(1)
            request_idx += 1
