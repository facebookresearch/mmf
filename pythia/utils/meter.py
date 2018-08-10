import torch


class Meter:
    METRICS_TO_FUNC_MAPPING = {
        'accuracy': 'accuracy',
        'vqa_accuracy': 'vqa_accuracy'
    }

    def __init__(self, dataset_type, config, meter_types):
        self.dataset_type = dataset_type

        if not isinstance(meter_types, list):
            meter_types = [meter_types]

        self.config = config
        self.meter_types = meter_types
        self.reset()

    def __call__(self, output, expected):
        values = []
        self.iteration_count += 1

        for i in range(len(self.meter_types)):
            meter_type = self.meter_types[i]
            func = getattr(self, self.METRICS_TO_FUNC_MAPPING[meter_type])
            value = func(output, expected)
            values.append(value)
            self.avg_meter_values[i] += value

        return values

    def get_dataset_type(self):
        return self.dataset_type

    def masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def accuracy(self, output, expected):
        if self.config['use_cuda']:
            correct = (expected == output.squeeze()).data.cpu().numpy().sum()
        else:
            correct = (expected == output.squeeze()).data.sum()

        total = len(expected)

        return correct / total

    def vqa_accuracy(self, output, expected):
        output = self.masked_unk_softmax(output, 1, 0)
        output = torch.max(output, 1)[1].data  # argmax
        one_hots = torch.zeros(*expected.size())
        one_hots = one_hots.cuda() if self.config['use_cuda'] else one_hots
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = (one_hots * expected)
        return torch.sum(scores)

    def get_averages(self):
        return self.avg_meter_values / self.iteration_count

    def reset(self):
        self.avg_meter_values = []

        for _ in self.meter_types:
            self.avg_meter_values.append(0)

        self.iteration_count = 0

    def get_log_string(self, loss):
        log_string = ["Average loss: %.4f" % loss]

        for i in range(len(self.meter_types)):
            meter_type = self.meter_types[i]
            avg_value = self.avg_meter_values[i] / self.iteration_count
            log_string.append("Average %s: %.4f" % (meter_type, avg_value))

        iteration = "%s: %s/%s: " % (self.dataset_type,
                                     self.iteration_count,
                                     self.config['max_iterations'])

        # If it is not train, then no sense of iteration count
        if self.dataset_type != 'train':
            iteration = self.dataset_type + ": "
        return iteration + ', '.join(log_string)
