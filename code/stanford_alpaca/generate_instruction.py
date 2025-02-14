"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire

# import BlancedInstanceSampler from samplers.py in parent directory
import sys
sys.path.insert(0, '..')

from samplers import BalancedInstanceSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datetime import datetime


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        print(f"Instruction: {instruction}")
        print(f"Input: {input}")
        print(f"Output: {output}")
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input == None else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["response"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        #if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
        #    continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
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
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    num_instructions_to_generate=50000,
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=32,
):
    #seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    #seed_instruction_data = [
    #    {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
    #    for t in seed_tasks
    #]
    #print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    balanced_instance_sampler = BalancedInstanceSampler()
    balanced_seed_instructions = next(balanced_instance_sampler.get_batch('train', 1))
    print(len(balanced_seed_instructions))
    task_names = set([instance.id.split('_')[0] for instance in balanced_seed_instructions])
    seed_instructions = []
    for task_name in task_names:
        for instance in balanced_seed_instructions:
            if instance.id.split('_')[0] == task_name:
                seed_instructions.append(instance)
                break
    seed_instruction_data = [
        {"instruction": s.task_instruction, "input": s.input, "output": s.output}
        for s in seed_instructions
    ]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    model = LLM("meta-llama/Meta-Llama-3-70B", tensor_parallel_size=4, max_model_len=4096)
    tokenizer = model.get_tokenizer()

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)


        #decoding_args = utils.OpenAIDecodingArguments(
        #    temperature=temperature,
        #    n=1,
        #    max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        #    top_p=top_p,
        #    stop=["\n20", "20.", "20."],
        #)
        request_start = time.time()
        #results = utils.openai_completion(
        #    prompts=batch_inputs,
        #    model_name=model_name,
        #    batch_size=request_batch_size,
        #    decoding_args=decoding_args,
        #    logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        #)

        outputs = model.generate(
            batch_inputs,
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=3072,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            )
        )

        results = []
        for j, (prompt, output) in enumerate(zip(batch_inputs, outputs)):
            data = {
                "prompt": prompt,
                "response": output.outputs[0].text,
                "created_at": str(datetime.now()),
            }
            results.append(data)


        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            #with Pool(num_cpus) as p:
            #    rouge_scores = p.map(
            #        partial(rouge_scorer._score_lcs, new_instruction_tokens),
            #        all_instruction_tokens,
            #    )

            rouge_scores = [partial(rouge_scorer._score_lcs, new_instruction_tokens)(all_instruction_tokens[i]) for i in range(len(all_instruction_tokens))]
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
