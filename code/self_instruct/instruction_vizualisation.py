import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)
import json
import tqdm

import benepar, spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("The time for action is now. It's never too late to do something.")

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    

    
def find_root_verb_and_its_dobj(tree_root):
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None

def find_root_verb_and_its_dobj_in_string(s):
    doc = nlp(s)
    last_sent = list(doc.sents)[-1]
    return find_root_verb_and_its_dobj(last_sent.root)


generated_data_path = "../../data/self_instruct_llm_generations_maja/finetuning/sampled_generated_train_instances_52445.json" #machine_generated_instructions.jsonl"
machine_generated_tasks = []
with open(generated_data_path, 'r') as fin:
    machine_generated_tasks= json.loads(fin.readline())
    
instructions = set([task["instruction"] for task in machine_generated_tasks])
print(len(instructions))

raw_phrases = []
for instruction in tqdm.tqdm(instructions):
    try:b
        verb, noun = find_root_verb_and_its_dobj_in_string(instruction)
        raw_phrases.append({
            "verb": verb,
            "noun": noun,
            "instruction": instruction
        })
    except Exception as e:
        print(e)
        print(instruction)
        
raw_phrases_df = pd.DataFrame(raw_phrases)
raw_phrases_df.to_csv('raw_phrases_last_sent.csv')