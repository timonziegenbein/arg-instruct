from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict

model_name = "/bigwork/nhwpstam/argpaca/models/llama3-8b-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Which planet is closer to the Sun: Mercury or Earth?"
completions = ("Mercury", "Earth")

pred = predict(prompt, completions, model_and_tokenizer=(model, tokenizer))
print(pred)
# Mercury
