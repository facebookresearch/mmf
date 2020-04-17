# Copyright (c) Facebook, Inc. and its affiliates.
import time


class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d %H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        self.start = time.time() * 1000

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def reset(self):
        self.start = time.time() * 1000

    def get_time_since_start(self, format=None):
        return self.get_time_hhmmss(self.start, format)

    def unix_time_since_start(self, in_seconds=True):
        gap = time.time() * 1000 - self.start

        if in_seconds:
            gap = gap // 1000

        # Prevent 0 division errors
        if gap == 0:
            gap = 1
        return gap

    def get_time_hhmmss(self, start=None, end=None, gap=None, format=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None and gap is None:

            if format is None:
                format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(format)

        if end is None:
            end = time.time() * 1000
        if gap is None:
            gap = end - start

        s, ms = divmod(gap, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        if format is None:
            format = self.DEFAULT_TIME_FORMAT

        items = [ms, s, m, h]
        assert len(items) == len(format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = format[idx] % item + " " + time_str

        # Means no more time is left.
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()
