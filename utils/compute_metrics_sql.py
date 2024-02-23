import logging
from typing import Optional, Dict, Any
from utils.third_party.test_suite import evaluation as test_suite_evaluation

logger = logging.getLogger(__name__)


def compute_test_suite_metric(predictions, references, db_dir: Optional[str] = None) -> Dict[str, Any]:
    flag = 0
    if db_dir is None:
        flag = 1
        db_dir = references[0]["db_path"]

    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )

    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=foreign_key_maps,
        etype="all",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}

    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            _ = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
    evaluator.finalize()
    test_suite_evaluation.print_scores(evaluator.scores, "all", include_turn_acc=False)

    if flag:
        return {
            "exec": evaluator.scores["all"]["exec"] * 100,
            "exact_match": evaluator.scores["all"]["exact"] * 100,
            # "exec_scores": evaluator.scores,
        }
    else:
        return {
            "test_suite_exec": evaluator.scores["all"]["exec"],
            "exact_match": evaluator.scores["all"]["exact"],
            # "exec_scores": evaluator.scores,
        }


def compute_metrics(predictions, references):
    assert len(predictions) == len(references)
    assert type(predictions[0]) is str
    assert type(references[0]) is dict

    metrics = compute_test_suite_metric(predictions, references)

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
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results



