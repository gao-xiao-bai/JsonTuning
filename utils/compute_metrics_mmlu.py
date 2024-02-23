import logging
from .scorer import MMLUScorer


logger = logging.getLogger(__name__)


def compute_metrics(predictions, references, scorer):
    assert len(predictions) == len(references)
    assert type(predictions[0]) is str
    assert type(references[0]) is str

    metrics = {}
    metrics.update(scorer.eval(predictions, references, verbose=False))

    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))

    results = {}
    scorer = MMLUScorer
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, scorer)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results



