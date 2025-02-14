import json
import tqdm
import os
import random
from datetime import datetime
import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

'''model_path = "Meta-Llama-3-70B" #/bigwork/nhwpstam/argpaca/models/Meta-Llama-3-8B"  # Meta-Llama-3-70B
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)



def make_requests(prompts, max_tokens=1024  # , temperature=0.7, top_p=0.5, frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
                  ):
    response = None
    target_length = max_tokens

    inputs = tokenizer(
        prompts,
        padding='max_length',
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    results = []
    for j, prompt in enumerate(prompts):
        data = {
            "prompt": prompt,
            "response": tokenizer.decode(outputs[j], skip_special_tokens=True)[len(prompt):],
            "created_at": str(datetime.now()),
        }
        results.append(data)
    return results'''


base_path = os.environ['ARGPACA_MAJA']
#model = LLM("meta-llama/Meta-Llama-3-70B", tensor_parallel_size=4, max_model_len=4096)
model = LLM(base_path + "/code/self_instruct/Meta-Llama-3-70B", tensor_parallel_size=4, max_model_len=4096)#4, max_model_len=4096) #meta-llama/
tokenizer = model.get_tokenizer()


def make_requests(prompts, max_tokens=1024, temperature=0.6, top_p=0.9, presence_penalty=0.0, greedy=False
                  ):

#    conversations = tokenizer.apply_chat_template(
#        [{'role': 'user', 'content': prompt} for prompt in prompts],
#        tokenize=False,
#    )

    if greedy:
        top_k = 1
    else:
        top_k = -1

    outputs = model.generate(
#        [conversations],
        prompts,
        SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
    )

    results = []
    for j, (prompt, output) in enumerate(zip(prompts, outputs)):
        data = {
            "prompt": prompt,
            #"response": output.outputs[0].text.replace('<|start_header_id|>assistant<|end_header_id|>\n\n', ''),
            "response": output.outputs[0].text,
            "created_at": str(datetime.now()),
        }
        results.append(data)
    return results
