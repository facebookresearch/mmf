import torch

from pythia.common.registry import registry


class Meter:
    METRICS_TO_FUNC_MAPPING = {
        'accuracy': 'accuracy',
        'vqa_accuracy': 'average_vqa_accuracy',
        'r@1': 'recall_at_1',
        'r@5': 'recall_at_5',
        'r@10': 'recall_at_10',
        'mean_r': 'mean_rank',
        'mean_rr': 'mean_reciprocal_rank'
    }

    ACCURACY_DECAY = 0.99

    def __init__(self, dataset_name, dataset_type, meter_types):
        self.dataset_type = dataset_type

        if not isinstance(meter_types, list):
            meter_types = [meter_types]

        self.config = registry.get('config')
        self.meter_types = meter_types
        self.reset()
        self.dataset_name = dataset_name
        self.top_level_key = "metrics.%s." % dataset_type

    def __call__(self, report, *args, **kwargs):
        values = []
        self.iteration_count += 1
        for i in range(len(self.meter_types)):
            meter_type = self.meter_types[i]
            # func = getattr(self, self.METRICS_TO_FUNC_MAPPING[meter_type])
            func = registry.get_metric_func(meter_type)
            # Maintain value in the function itself.
            # If you need to calculate average,
            # use 'iteration_count' to update the value
            value = func(self, self.meter_values[i], report, *args, **kwargs)
            values.append(value)
            key = self.top_level_key + "%s_%s" % (self.dataset_name,
                                                  meter_type)
            registry.register(key, value)
            self.meter_values[i] = value

        return values

    def get_dataset_type(self):
        return self.dataset_type

    def get_values(self, index=None):
        if index is None:
            return self.meter_values
        else:
            return self.meter_values[index]

    def reset(self):
        self.meter_values = []

        for _ in self.meter_types:
            self.meter_values.append(0)

        if self.dataset_type == "train":
            self.iteration_count = registry.get('current_iteration', 0)
        else:
            self.iteration_count = 0

    def get_log_string(self, loss):
        log_string = ["Average loss: %.4f" % loss]

        for i in range(len(self.meter_types)):
            meter_type = self.meter_types[i]
            value = self.meter_values[i]
            log_string.append("Average %s: %.4f" %
                              (self.dataset_name + "_" + meter_type, value))

        max_iterations = self.config['training_parameters']['max_iterations']
        iteration = "%s: %s/%s: " % (self.dataset_type,
                                     self.iteration_count,
                                     max_iterations)

        # If it is not train, then no sense of iteration count
        if self.dataset_type != 'train':
            iteration = self.dataset_type + ": "
        return iteration + ', '.join(log_string)

    def masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    @registry.register_metric('accuracy')
    def accuracy(self, current, report):
        output = report.scores
        expected = report.targets
        output = torch.max(output, 1)[1]

        if self.config['use_cuda']:
            correct = (expected == output.squeeze()).data.cpu().numpy().sum()
        else:
            correct = (expected == output.squeeze()).data.sum()

        total = len(expected)

        current = current * (self.iteration_count - 1)
        current += (correct / total)
        current /= self.iteration_count

        return current

    # TODO: Fix for incomplete batches
    @registry.register_metric('vqa_accuracy')
    def average_vqa_accuracy(self, current, report):
        output = report.scores
        expected_data = report.targets
        output = self.masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected_data.new_zeros(*expected_data.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = (one_hots * expected_data)
        accuracy = torch.sum(scores) / expected_data.size(0)

        if self.dataset_type == 'train':
            current += (1 - self.ACCURACY_DECAY) * (accuracy - current)
        else:
            current = current * (self.iteration_count - 1)
            current += accuracy
            current /= self.iteration_count
        return current

    def recall_at_k(self, current, report, k):
        ranks = self.get_ranks(output, expected)
        current = current * (self.iteration_count - 1)
        current += float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        current /= self.iteration_count
        return current

    @registry.register_metric('r@1')
    def recall_at_1(self, current, report):
        return self.recall_at_k(current, report, 1)

    @registry.register_metric('r@5')
    def recall_at_5(self, current, report):
        return self.recall_at_k(current, report, 5)

    @registry.register_metric('r@10')
    def recall_at_10(self, current, report):
        return self.recall_at_k(current, report, 10)

    @registry.register_metric('mean_r')
    def mean_rank(self, current, report, info={}):
        ranks = self.get_ranks(report)
        current = current * (self.iteration_count - 1)
        current += torch.mean(ranks)
        return current / self.iteration_count

    def get_ranks(self, report):
        output = report.scores
        expected = report.targets
        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    @registry.register_metric('mean_rr')
    def mean_reciprocal_rank(self, current, report, info={}):
        ranks = self.get_ranks(report)
        current = current * (self.iteration_count - 1)
        current += torch.mean(ranks.reciprocal())
        current /= self.iteration_count

        return current

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

    def process_ranks(self, ranks):
        num_opts = 100

        # none of the values should be 0, there is gt in options
        if torch.sum(ranks.le(0)) > 0:
            num_zero = torch.sum(ranks.le(0))
            print("Warning: some of ranks are zero: {}".format(num_zero))
            ranks = ranks[ranks.gt(0)]

        # rank should not exceed the number of options
        if torch.sum(ranks.ge(num_opts + 1)) > 0:
            num_ge = torch.sum(ranks.ge(num_opts + 1))
            print("Warning: some of ranks > 100: {}".format(num_ge))
            ranks = ranks[ranks.le(num_opts + 1)]
        return ranks
