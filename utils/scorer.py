from typing import Dict, List, Tuple
import string


class Metric:
    def __init__(self, verbose=False):
        self.correct = 0.
        self.all = 0.
        self.verbose = verbose

    def __repr__(self) -> str:
        return f"correct: {self.correct}, all: {self.all}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_accuracy(self, prefix=''):
        accuracy = self.safe_div(self.correct, self.all)
        return {prefix + 'correct': self.correct,
                prefix + 'all': self.all,
                prefix + 'acc': accuracy * 100,
                }

    def count_instance(self, gold, pred):
        gold = normalize(gold)
        pred = normalize(pred)
        if self.verbose:
            print("Gold:", gold)
            print("Pred:", pred)
        self.all += 1
        if pred == gold:
            self.correct += 1

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold, pred in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold=gold, pred=pred)


def normalize(s):
    def white_space_fix(s):
        return ' '.join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation) - {"$", "€", "£", "¥"}
        return ''.join(ch for ch in s if ch not in exclude)

    def lower(s):
        return s.lower()

    return lower(white_space_fix(remove_punc(s)))


class MMLUScorer:
    @staticmethod
    def eval(pred_instance_list: List[str], gold_instance_list: List[str], verbose=False):
        metric = Metric(verbose=verbose)

        for pred, gold in zip(pred_instance_list, gold_instance_list):

            metric.count_instance(
                gold=gold,
                pred=pred,
            )

        results = dict()
        results.update(metric.compute_accuracy(prefix=''))

        return results





