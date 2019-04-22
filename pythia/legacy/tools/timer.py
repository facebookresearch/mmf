# Copyright (c) Facebook, Inc. and its affiliates.
import timeit


class Timer:
    def __init__(self, unit="s"):
        self.s_time = timeit.default_timer()
        self.unit = unit
        if self.unit != "s" and self.unit != "m" and self.unit != "h":
            raise NotImplementedError("unkown time unit, using s, m, h")

    def start(self):
        self.s_time = timeit.default_timer()

    def end(self):
        self.e_time = timeit.default_timer()
        period = self.e_time - self.s_time
        if self.unit == "s":
            return "%.1f s" % period
        elif self.unit == "m":
            return "%.2f min" % (period / 60)
        else:
            return "%.2f h" % (period / 3600)
