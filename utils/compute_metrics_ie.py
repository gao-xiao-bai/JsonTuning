import logging
from .ie_scorer import EntityScorer, RelationScorer, EventScorer


logger = logging.getLogger(__name__)

scorer_dict = {
    'NER': EntityScorer,
    'RE': RelationScorer,
    'EE': EventScorer,
}


def compute_metrics(predictions, references, texts, scorer, text_tuning=0):
    assert len(predictions) == len(references)
    assert type(predictions[0]) is list
    assert type(references[0]) is list

    if text_tuning == 0:
        predictions = [scorer.load(x, text) for x, text in zip(predictions, texts)]
    elif text_tuning == 1:
        predictions = [scorer.load_text(x, text) for x, text in zip(predictions, texts)]
    else:
        raise ValueError(f"Text tuning {text_tuning} is wrong!")

    references = [scorer.load(x, text) for x, text in zip(references, texts)] 

    metrics = {}
    metrics.update(scorer.eval(predictions, references, verbose=False))

    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, texts, text_tuning=0):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task = group.split("_")[0]
        scorer = scorer_dict[task]
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, texts, scorer, text_tuning)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results



