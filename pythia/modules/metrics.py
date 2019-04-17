import torch
import collections

from pythia.common.registry import registry


class Metrics:
    def __init__(self, metric_list):
        if not isinstance(metric_list, list):
            metrics_list = [metric_list]

        self.writer = registry.get("writer")
        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        for metric in metric_list:
            params = {}
            if isinstance(metric, collections.Mapping):
                if not hasattr(metric, "type"):
                    raise ValueError("Metric {} needs to have 'type' "
                                     "attribute".format(metric))
                metric = metric.type
                params = getattr(metric, "params", {})
            else:
                if not isinstance(metric, str):
                    raise TypeError("Metric {} has inappropriate type"
                                    "'dict' or 'str' allowed".format(metric))

            metric_cls = registry.get_metric_class(metric)
            if metric_cls is None:
                raise ValueError("No metric named {} registered to registry"
                                  .format(metric))
            metrics[metric] = metric_cls(**params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}
        if not hasattr(sample_list, "targets"):
            return values

        dataset_type = sample_list.dataset_type

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                key = "{}/{}".format(dataset_type, metric_name)
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

        registry.register("{}.{}.{}".format(
            "metrics", sample_list.dataset_name, dataset_type
        ), values)

        return values


class BaseMetric:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        # Override in your child class
        return

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value


@registry.register_metric("accuracy")
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__("accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]
        output = torch.max(output, 1)[1]

        correct = (expected == output.squeeze()).sum()

        correct = correct
        total = len(expected)

        value = (correct / total)
        return value


@registry.register_metric("vqa_accuracy")
class VQAAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("vqa_accuracy")

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        output = self._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected.new_zeros(*expected.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = (one_hots * expected)
        accuracy = torch.sum(scores) / expected.size(0)

        return accuracy


class RecallAtK(BaseMetric):
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def score_to_ranks(self, scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    def get_gt_ranks(self, ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    def calculate(self, sample_list, model_output, k, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall


@registry.register_metric("r@1")
class RecallAt1(RecallAtK):
    def __init__(self):
        super().__init__("r@1")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        return self.calculate(sample_list, model_output, k=1)


@registry.register_metric("r@5")
class RecallAt5(RecallAtK):
    def __init__(self):
        super().__init__("r@5")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        return self.calculate(sample_list, model_output, k=5)


@registry.register_metric("r@10")
class RecallAt10(RecallAtK):
    def __init__(self):
        super().__init__("r@10")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        return self.calculate(sample_list, model_output, k=10)


@registry.register_metric("mean_r")
class MeanRank(RecallAtK):
    def __init__(self):
        super().__init__("mean_r")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks)


@registry.register_metric("mean_rr")
class MeanReciprocalRank(RecallAtK):
    def __init__(self):
        super().__init__("mean_rr")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks.reciprocal())
