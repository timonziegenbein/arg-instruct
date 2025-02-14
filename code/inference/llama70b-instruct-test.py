from vllm import LLM, SamplingParams

#sampling_params = SamplingParams(temperature=0.6, top_p=0.9)
# Use LLaMA 3 70B on 4 GPUs
llm = LLM("meta-llama/Meta-Llama-3-7B-Instruct", tensor_parallel_size=2, max_model_len=4096)

tokenizer = llm.get_tokenizer()

conversations = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'What are the most popular quantization techniques for LLMs?'}],
    tokenize=False,
)

outputs = llm.generate(
    [conversations],
    SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
    )
)

for output in outputs:
    prompt = output.prompt
    generated_text = ' '.join([o.text for o in output.outputs])
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
