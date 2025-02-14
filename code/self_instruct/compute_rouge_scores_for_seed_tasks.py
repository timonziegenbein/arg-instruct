from rouge_score import rouge_scorer
import numpy as np
from bert_score import score
import sys
sys.path.insert(0, '..')

from samplers import BalancedInstanceSampler, InstructionSampler

if __name__ == '__main__':
    instruction_sampler = InstructionSampler()
    seed_instructions = instruction_sampler.get_all()
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    rouge_scores = []
    for i, inst_a in enumerate(seed_instructions):
        for j, inst_b in enumerate(seed_instructions):
            if j > i:
                rouge_score = scorer.score(inst_a, inst_b)["rougeL"].fmeasure
                rouge_scores.append(rouge_score)
    print(f"Max Rouge-L score between seed instructions: {max(rouge_scores)}")
    print(f"Min Rouge-L score between seed instructions: {min(rouge_scores)}")
    print(f"Average Rouge-L score between seed instructions: {sum(rouge_scores) / len(rouge_scores)}")
    print(f"95% percentile of Rouge-L score between seed instructions: {np.percentile(rouge_scores, 95)}")
    print(f"5% percentile of Rouge-L score between seed instructions: {np.percentile(rouge_scores, 5)}")

    instance_list_a = []
    instance_list_b = []
    for i, inst_a in enumerate(seed_instructions):
        for j, inst_b in enumerate(seed_instructions):
            if j > i:
                instance_list_a.append(inst_a)
                instance_list_b.append(inst_b)
    precision, recall, f1 = score(instance_list_a, instance_list_b, lang='en', verbose=True, rescale_with_baseline=True)
    print(f"Max BERTScore between seed instructions: {max(f1)}")
    print(f"Min BERTScore between seed instructions: {min(f1)}")
    print(f"Average BERTScore between seed instructions: {sum(f1) / len(f1)}")
    print(f"95% percentile of BERTScore between seed instructions: {np.percentile(f1, 95)}")
    print(f"5% percentile of BERTScore between seed instructions: {np.percentile(f1, 5)}")
