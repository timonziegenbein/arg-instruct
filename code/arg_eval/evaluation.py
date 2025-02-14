import string
import numpy as np
import pandas as pd
import json
import argparse
from rouge import rouge_scorer
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support


class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word.
        # But for the first word of a sentence, there is no space before it.
        # So, we remove all the added spaces ("Ġ").
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
xlingual_tokenizer = GPTTokenizer()
xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer)

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, references=None, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def match(prediction, ground_truth, references=None, xlingual=False):
    references = list(set([l for r in references for l in r]))  # flatten the list of lists
    return (normalize_answer(prediction) in [normalize_answer(gt) for gt in references])


def no_match(prediction, ground_truth, references, xlingual=False):
    references = list(set([l for r in references for l in r]))  # flatten the list of lists
    return (normalize_answer(prediction) not in [normalize_answer(gt) for gt in references])


def rouge(prediction, ground_truth, references=None, xlingual=False):
    if xlingual:
        scorer = xlingual_rouge_scorer
    else:
        scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, references=None, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, references=references, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, predictions_for_matching, xlingual=False):
    assert len(predictions) == len(
        references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_matches, evaluatable_instances, rougeL = 0, 0, 0

    references_flat = np.array(references).flatten()
    predictions_flat = np.array(predictions).flatten()
    predictions_for_matching_flat = np.array(predictions_for_matching).flatten()
    references_flat = np.array([normalize_answer(r) for r in references_flat])
    predictions_flat = np.array([normalize_answer(p) for p in predictions_flat])
    predictions_for_matching_flat = np.array([normalize_answer(p) for p in predictions_for_matching_flat])

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        references_flat, predictions_flat, average='macro', labels=list(set(references_flat)), zero_division=0)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        references_flat, predictions_flat, average='micro', labels=list(set(references_flat)), zero_division=0)

    # calculate mean absoulte scaled error
    mae = np.mean(np.abs(predictions_flat - references_flat))
    mad = np.mean(np.abs(references_flat - np.mean(references_flat)))
    mase = mae / mad

    no_matches = []
    for pred, gold in zip(predictions_for_matching, references):
        assert isinstance(gold, list)
        no_matches.append(metric_max_over_ground_truths(
            no_match, prediction=pred, ground_truths=gold, references=references, xlingual=xlingual
        ))

    for pred, gold in zip(predictions, references):
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold, references=references, xlingual=xlingual
        )

    matches = np.invert(no_matches)
    macro_precision_match, macro_recall_match, macro_f1_match, _ = precision_recall_fscore_support(
        references_flat[matches], predictions_flat[matches], average='macro', labels=list(set(references_flat)), zero_division=0)
    micro_precision_match, micro_recall_match, micro_f1_match, _ = precision_recall_fscore_support(
        references_flat[matches], predictions_flat[matches], average='micro', labels=list(set(references_flat)), zero_division=0)

    for pred, gold in zip(predictions_flat[matches], references_flat[matches]):
        exact_matches += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=[gold], references=[[x] for x in references_flat[matches]], xlingual=xlingual
        )

    rougeL = 100.0 * rougeL / len(references)
    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision_match": macro_precision_match,
        "macro_recall_match": macro_recall_match,
        "macro_f1_match": macro_f1_match,
        "micro_precision_match": micro_precision_match,
        "micro_recall_match": micro_recall_match,
        "micro_f1_match": micro_f1_match,
        "mase": mase,
        "exact_matches": exact_matches / len(references),
        "evaluatable_instances": (len(references)-sum(no_matches))/len(references),
        "rougeL": rougeL
    }
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, predictions_for_matching, groups, xlingual=False):
    assert len(predictions) == len(references) == len(predictions_for_matching) == len(groups)

    examples_by_group = {}
    for pred, gold, pred_for_match, group in zip(predictions, references, predictions_for_matching, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold, pred_for_match))

    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references, task_predictions_for_matching = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, task_predictions_for_matching, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        help="Jsonl file to write the results to.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prediction_files = [
        '../inference/predictions_default.jsonl',
        '../inference/predictions_cappr.jsonl',
        '../inference/predictions_guidance.jsonl',
        '../inference/predictions_random.jsonl',
        '../inference/predictions_outline.jsonl'
    ]
    data = {
        'task': [],
        'prediction_file': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
        'micro_precision': [],
        'micro_recall': [],
        'micro_f1': [],
        'macro_precision_match': [],
        'macro_recall_match': [],
        'macro_f1_match': [],
        'micro_precision_match': [],
        'micro_recall_match': [],
        'micro_f1_match': [],
        'mase': [],
        'exact_matches': [],
        'evaluatable_instances': [],
        'rougeL': []
    }

    # read default predictions_for_matching_flat
    predictions_for_matching = []
    with open(prediction_files[0]) as fin:
        for line in fin:
            prediction = json.loads(line)
            predictions_for_matching.append(prediction["prediction"])

    for prediction_file in prediction_files:
        tasks = []
        predictions = []
        references = []
        with open(prediction_file) as fin:
            for line in fin:
                prediction = json.loads(line)
                tasks.append(prediction["id"].split("_")[0])
                predictions.append(prediction["prediction"])
                references.append([prediction["reference"]])

        print(f"Loaded {len(predictions)} predictions")
        print(f"Loaded {len(references)} references")

        all_results = {}
        #results = compute_metrics(predictions, references)
        ## print("======== Overall Metrics ========")
        #for metric, value in results.items():
        #    # print(f"{metric}: {value}")
        #    all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            predictions, references, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

    # put into nice pandas with tasks as rows and metrics as columns
        stored_task_prediction_file_combinations = []
        for key, value in all_results.items():
            if key not in data.keys():
                metric, task = key.split("_for_")
            else:
                metric = key
                task = "overall"
            if (task, prediction_file) not in stored_task_prediction_file_combinations:
                data['task'].append(task)
                data['prediction_file'].append(prediction_file)
                stored_task_prediction_file_combinations.append((task, prediction_file))

            data[metric].append(value)

    df = pd.DataFrame(data)
    # group by prediction file and compute the mean save as row with task = "overall"
    mean_scores = df[['prediction_file', 'macro_precision', 'macro_recall', 'macro_f1', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision_match', 'macro_recall_match', 'macro_f1_match', 'micro_precision_match', 'micro_recall_match', 'micro_f1_match', 'mase', 'exact_matches', 'evaluatable_instances', 'rougeL']].groupby('prediction_file').mean().reset_index()
    # add overall rows
    mean_scores['task'] = 'overall'
    df = pd.concat([df, mean_scores], ignore_index=True)
    df.to_json(args.output_file, orient='records', lines=True)

    if True: # This is just to see whether cappr works
        cappr_predictions = []
        default_predictions = []
        references = []
        tasks = []
        for prediction_file in prediction_files:
            with open(prediction_file) as fin:
                for line in fin:
                    prediction = json.loads(line)
                    if "cappr" in prediction_file:
                        cappr_predictions.append([prediction["prediction"]])
                        tasks.append(prediction["id"].split("_")[0])
                        references.append([prediction["reference"]])
                    elif "default" in prediction_file:
                        default_predictions.append(prediction["prediction"])

        print(f"Loaded {len(cappr_predictions)} predictions")
        print(f"Loaded {len(default_predictions)} references")

        all_results = {}
        results = compute_metrics(default_predictions, cappr_predictions, predictions_for_matching)
        # print("======== Overall Metrics ========")
        for metric, value in results.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            default_predictions, cappr_predictions, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value
        print(f"CAPPR vs. Default")
        print(f"Exact matches: {all_results['exact_matches']}")
        print(f"Evaluatable instances: {all_results['evaluatable_instances']}")
        print(f"Agreement: {all_results['exact_matches'] / all_results['evaluatable_instances']}")

        guidance_predictions = []
        default_predictions = []
        references = [] 
        tasks = []
        for prediction_file in prediction_files:
            with open(prediction_file) as fin:
                for line in fin:
                    prediction = json.loads(line)
                    if "guidance" in prediction_file:
                        guidance_predictions.append([prediction["prediction"]])
                        tasks.append(prediction["id"].split("_")[0])
                        references.append([prediction["reference"]])
                    elif "default" in prediction_file:
                        default_predictions.append(prediction["prediction"])

        print(f"Loaded {len(guidance_predictions)} predictions")
        print(f"Loaded {len(default_predictions)} references")

        all_results = {}
        results = compute_metrics(default_predictions, guidance_predictions, predictions_for_matching)
        # print("======== Overall Metrics ========")
        for metric, value in results.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            default_predictions, guidance_predictions, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value
        print(f"Guidance vs. Default")
        print(f"Exact matches: {all_results['exact_matches']}")
        print(f"Evaluatable instances: {all_results['evaluatable_instances']}")
        print(f"Agreement: {all_results['exact_matches'] / all_results['evaluatable_instances']}")


        outline_predictions = []
        default_predictions = []
        references = []
        tasks = []
        for prediction_file in prediction_files:
            with open(prediction_file) as fin:
                for line in fin:
                    prediction = json.loads(line)
                    if "outline" in prediction_file:
                        outline_predictions.append([prediction["prediction"]])
                        tasks.append(prediction["id"].split("_")[0])
                        references.append([prediction["reference"]])
                    elif "default" in prediction_file:
                        default_predictions.append(prediction["prediction"])

        print(f"Loaded {len(outline_predictions)} predictions")
        print(f"Loaded {len(default_predictions)} references")

        all_results = {}
        results = compute_metrics(default_predictions, outline_predictions, predictions_for_matching)
        # print("======== Overall Metrics ========")
        for metric, value in results.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            default_predictions, outline_predictions, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value
        print(f"Outline vs. Default")
        print(f"Exact matches: {all_results['exact_matches']}")
        print(f"Evaluatable instances: {all_results['evaluatable_instances']}")
        print(f"Agreement: {all_results['exact_matches'] / all_results['evaluatable_instances']}")


        # guidance vs. cappr_predictions
        outline_predictions = []
        guidance_predictions = []
        references = []
        tasks = []
        for prediction_file in prediction_files:
            with open(prediction_file) as fin:
                for line in fin:
                    prediction = json.loads(line)
                    if "outline" in prediction_file:
                        outline_predictions.append([prediction["prediction"]])
                        tasks.append(prediction["id"].split("_")[0])
                        references.append([prediction["reference"]])
                    elif "guidance" in prediction_file:
                        guidance_predictions.append(prediction["prediction"])

        print(f"Loaded {len(outline_predictions)} predictions")
        print(f"Loaded {len(guidance_predictions)} references")

        all_results = {}
        results = compute_metrics(guidance_predictions, outline_predictions, predictions_for_matching)
        # print("======== Overall Metrics ========")
        for metric, value in results.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            guidance_predictions, outline_predictions, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value
        print(f"Outline vs. Guidance")
        print(f"Exact matches: {all_results['exact_matches']}")
        print(f"Evaluatable instances: {all_results['evaluatable_instances']}")
        print(f"Agreement: {all_results['exact_matches'] / all_results['evaluatable_instances']}")


        # guidance vs. cappr_predictions
        cappr_predictions = []
        guidance_predictions = []
        references = []
        tasks = []
        for prediction_file in prediction_files:
            with open(prediction_file) as fin:
                for line in fin:
                    prediction = json.loads(line)
                    if "cappr" in prediction_file:
                        cappr_predictions.append([prediction["prediction"]])
                        tasks.append(prediction["id"].split("_")[0])
                        references.append([prediction["reference"]])
                    elif "guidance" in prediction_file:
                        guidance_predictions.append(prediction["prediction"])

        print(f"Loaded {len(cappr_predictions)} predictions")
        print(f"Loaded {len(guidance_predictions)} references")

        all_results = {}
        results = compute_metrics(guidance_predictions, cappr_predictions, predictions_for_matching)
        # print("======== Overall Metrics ========")
        for metric, value in results.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value

        results_per_category = compute_grouped_metrics(
            guidance_predictions, cappr_predictions, predictions_for_matching, tasks)
        # print("======== Metrics per Category ========")
        for metric, value in results_per_category.items():
            # print(f"{metric}: {value}")
            all_results[f"{metric}"] = value
        print(f"CAPPR vs. Guidance")
        print(f"Exact matches: {all_results['exact_matches']}")
        print(f"Evaluatable instances: {all_results['evaluatable_instances']}")
        print(f"Agreement: {all_results['exact_matches'] / all_results['evaluatable_instances']}")
