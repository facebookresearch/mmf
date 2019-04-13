import collections

from collections import OrderedDict

from pythia.common.registry import registry


class Report(OrderedDict):
    def __init__(self, batch, prepared_batch, model_output, *args):
        all_args = [batch, prepared_batch, model_output] + [*args]
        for idx, arg in enumerate(all_args):
            if not isinstance(arg, collections.Mapping):
                raise TypeError("Argument {:d}, {} must be of instance of "
                                "collections.Mapping".format(idx, arg))

        super().__init__(batch)

        self.update(prepared_batch)
        self["batch_size"] = prepared_batch.get_batch_size()

        self.writer = registry.get("writer")

        self.warning_string = "Updating forward report with key {}" \
                         "{}, but it already exists " \
                         "in {}. " \
                         "Please consider using a different key, " \
                         "as this can cause issues during loss and " \
                         "metric calculations."

        for key, item in model_output.items():
            if key in self:
                log = self.warning_string.format(
                        key, " from model output",
                        "sample list returned from the dataset"
                )
                self.writer.single_write(log, "warning")
            self[key] = item

        for arg in args:
            for key, item in arg.items():
                if key in self:
                    log = self.warning_string.format(
                        key, "", "sample list and model output"
                    )
                    self.writer.single_write(log, "warning")
                self[key] = item

    def __setattr__(self, key, value):
        if key in self:
            log = self.warning_string.format(
                key, "", "sample list and model output"
            )
            self.writer.single_write(log, "warning")
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def fields(self):
        return list(self.keys())
