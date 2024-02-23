from copy import deepcopy
from typing import Dict, List, Tuple
import string
import re


def fix_unk_from_text(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    从 text 中找到 span，修复span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "<unk> colo e Bengo"
    text = "Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "Arr<unk> s negre"
    text = "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region ."

    span = "colo <unk>"
    text = "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ"

    span = "Tarō As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "Tar<unk> As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "<unk>Tar As<unk>"
    text = "The leader of Japan is ōTar Asō ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("("+"|".join([f"\\{s}" for s in sp])+")", "\\\\\g<1>", x)

    match = r'\s*\S+\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])

    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()


class Metric:
    """ Tuple Metric """
    def __init__(self, verbose=False, match_mode='normal'):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.verbose = verbose
        self.match_mode = match_mode
        assert self.match_mode in {'set', 'normal', 'multimatch'}

    def __repr__(self) -> str:
        return f"tp: {self.tp}, gold: {self.gold_num}, pred: {self.pred_num}"

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list):
        if self.match_mode == 'set':
            gold_list = set(gold_list)
            pred_list = set(pred_list)
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)
            self.tp += len(gold_list & pred_list)

        else:
            if self.verbose:
                print("Gold:", gold_list)
                print("Pred:", pred_list)
            self.gold_num += len(gold_list)
            self.pred_num += len(pred_list)

            if len(gold_list) > 0 and len(pred_list) > 0:
                assert len(gold_list[0]) == len(pred_list[0])

            dup_gold_list = deepcopy(gold_list)
            for pred in pred_list:
                if pred in dup_gold_list:
                    self.tp += 1
                    if self.match_mode == 'normal':
                        # Each Gold Instance can be matched one time
                        dup_gold_list.remove(pred)

    def count_batch_instance(self, batch_gold_list, batch_pred_list):
        for gold_list, pred_list in zip(batch_gold_list, batch_pred_list):
            self.count_instance(gold_list=gold_list, pred_list=pred_list)


def normalize(s, text=None):
    def white_space_fix(s):
        return ' '.join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in s if ch not in exclude)

    # Fix unk using Text
    def fix_unk(s, text=None):
        if text is not None and '<unk>' in s:
            s = fix_unk_from_text(s, text, '<unk>')
        return s

    return white_space_fix(remove_punc(fix_unk(s, text)))


class EntityScorer:
    @staticmethod
    def load(entity_list: List[Dict], text: str):
        """
        Args:
            entity_list (List[Dict])

        Returns:
            List[Tuple]
        """
        loaded_entity_list = []
        for entity_dict in entity_list:
            try:
                entity_category = entity_dict["entity category"].strip()
                entity_span = normalize(entity_dict["entity span"], text)
                loaded_entity_list.append((entity_category, entity_span))
            except:
                print("Wrong entity: ", entity_dict)

        return loaded_entity_list

    @staticmethod
    def load_text(entity_list: List[List], text: str):
        """
        Args:
            entity_list (List[List])

        Returns:
            List[Tuple]
        """
        loaded_entity_list = []
        for e_list in entity_list:
            try:
                entity_category = e_list[0].strip()
                entity_span = normalize(e_list[1], text)
                loaded_entity_list.append((entity_category, entity_span))
            except:
                print("Wrong entity: ", e_list)

        return loaded_entity_list

    @staticmethod
    def eval(pred_instance_list: List[List[Tuple]], gold_instance_list: List[List[Tuple]], verbose=False, match_mode='normal'):
        """
        Args:
            pred_instance_list (List[List[Tuple]]):
            gold_instance_list (List[List[Tuple]])
            verbose (bool, optional): [description]. Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal` .

        Returns:
            Dict: Result of Evaluation
                (gold, pred, tp, P, R, F1)
        """
        metric = Metric(verbose=verbose, match_mode=match_mode)

        for pred, gold in zip(pred_instance_list, gold_instance_list):

            metric.count_instance(
                gold_list=gold,
                pred_list=pred,
            )

        results = dict()
        results.update(metric.compute_f1(prefix='ent-'))

        return results


class RelationScorer:
    @staticmethod
    def load(relation_list: List[Dict], text: str):
        """
        Args:
            relation_list (List[Dict])

        Returns:
            List[Tuple]
        """
        loaded_relation_list = []
        for relation_dict in relation_list:
            try:
                head_entity_category = relation_dict["head entity category"].strip() if "head entity category" in relation_dict else None
                head_entity_span = normalize(relation_dict["head entity span"], text)
                relation = relation_dict["relation"].strip()
                tail_entity_category = relation_dict["tail entity category"].strip() if "tail entity category" in relation_dict else None
                tail_entity_span = normalize(relation_dict["tail entity span"], text)
                loaded_relation_list.append((head_entity_category, head_entity_span, relation, tail_entity_category, tail_entity_span))
            except:
                print("Wrong relational triplet: ", relation_dict)

        return loaded_relation_list

    @staticmethod
    def load_text(relation_list: List[List], text: str):
        """
        Args:
            relation_list (List[List])

        Returns:
            List[Tuple]
        """
        loaded_relation_list = []
        for r_list in relation_list:
            try:
                if len(r_list) == 5:
                    head_entity_category = r_list[0].strip()
                    head_entity_span = normalize(r_list[1], text)
                    relation = r_list[2].strip()
                    tail_entity_category = r_list[3]
                    tail_entity_span = normalize(r_list[4], text)
                    loaded_relation_list.append((head_entity_category, head_entity_span, relation, tail_entity_category, tail_entity_span))
                elif len(r_list) == 3:
                    head_entity_span = normalize(r_list[0], text)
                    relation = r_list[1].strip()
                    tail_entity_span = normalize(r_list[2], text)
                    loaded_relation_list.append((None, head_entity_span, relation, None, tail_entity_span))
                else:
                    print("Wrong relational triplet: ", r_list)
            except:
                print("Wrong relational triplet: ", r_list)

        return loaded_relation_list


    @staticmethod
    def eval(pred_instance_list: List[List[Tuple]], gold_instance_list: List[List[Tuple]], verbose=False, match_mode='normal'):
        """
        Args:
            pred_instance_list (List[List[Tuple]])
            gold_instance_list (List[List[Tuple]])
            verbose (bool, optional): Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal` .

        Returns:
            Dict: Result of Evaluation
                (boundary, strict) X (gold, pred, tp, P, R, F1)
        """
        # Span Boundary and Type
        metric = Metric(verbose=verbose, match_mode=match_mode)
        # Span Boundary Only
        boundary_metric = Metric(verbose=verbose, match_mode=match_mode)

        for pred, gold in zip(pred_instance_list, gold_instance_list):

            metric.count_instance(
                gold_list=gold,
                pred_list=pred,
            )

            boundary_metric.count_instance(
                gold_list=[(x[1], x[2], x[4]) for x in gold],
                pred_list=[(x[1], x[2], x[4]) for x in pred],
            )

        results = dict()
        results.update(metric.compute_f1(prefix='rel-strict-'))
        results.update(boundary_metric.compute_f1(prefix='rel-boundary-'))
        return results


class EventScorer:
    @staticmethod
    def load(event_list: List[Dict], text: str):
        """
        Args:
            event_list (List[Dict])

        Returns:
            List[Tuple]
        """
        loaded_event_list = []
        for event_dict in event_list:
            try:
                event_category = event_dict["event category"].strip()
                event_trigger = normalize(event_dict["event trigger"], text)
                arguments = []
                if "arguments" in event_dict:
                    for argument_dict in event_dict["arguments"]:
                        argument_category = argument_dict["argument category"].strip()
                        argument_span = normalize(argument_dict["argument span"], text)
                        arguments.append((event_category, argument_category, argument_span))
                loaded_event_list.append(((event_category, event_trigger), arguments))
            except:
                print("Wrong event: ", event_dict)
        return loaded_event_list

    @staticmethod
    def load_text(event_list: List[Dict], text: str):
        """
        Args:
            event_list (List[List])

        Returns:
            List[Tuple]
        """
        loaded_event_list = []
        for e_list in event_list:
            try:
                event_category = e_list[0].strip()
                event_trigger = normalize(e_list[1], text)
                arguments = []
                try:
                    if len(e_list) > 2:
                        assert type(e_list[2]) is list
                        for argument_list in e_list[2]:
                            argument_category = argument_list[0].strip()
                            argument_span = normalize(argument_list[1], text)
                            arguments.append((event_category, argument_category, argument_span))
                    loaded_event_list.append(((event_category, event_trigger), arguments))
                except:
                    print("Wrong event: ", e_list)
                    loaded_event_list.append(((event_category, event_trigger), arguments))
            except:
                print("Wrong event: ", e_list)
        return loaded_event_list


    @staticmethod
    def eval(pred_instance_list: List[List[Tuple]], gold_instance_list: List[List[Tuple]], verbose=False, match_mode='normal'):
        """[summary]

        Args:
            pred_instance_list (List[List[Tuple]])
            gold_instance_list (List[List[Tuple]])
            verbose (bool, optional): [description]. Defaults to False.
            match_mode (string, optional): [description]. Defaults to `normal`.

        Returns:
            Dict: Result of Evaluation
                (trigger, role) X (gold, pred, tp, P, R, F1)
        """
        trigger_metric = Metric(verbose=verbose, match_mode=match_mode)
        role_metric = Metric(verbose=verbose, match_mode=match_mode)

        for pred, gold in zip(pred_instance_list, gold_instance_list):

            trigger_metric.count_instance(
                gold_list=[x[0] for x in gold],
                pred_list=[x[0] for x in pred],
            )

            role_metric.count_instance(
                gold_list=sum([x[1] for x in gold], []),
                pred_list=sum([x[1] for x in pred], []),
            )

        results = dict()
        results.update(trigger_metric.compute_f1(prefix='evt-trigger-'))
        results.update(role_metric.compute_f1(prefix='evt-role-'))

        return results
