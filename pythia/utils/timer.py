import time


class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d %H:%M:%S"
    DEFAULT_TIME_FORMAT = "%02d:%02d:%02d"

    def __init__(self):
        self.start = time.time()

    def get_current(self):
        return self.get_time_hhmmss(self.start)

    def get_time_hhmmss(self, start=None, format=None):
        """
        Calculates time since `start` and formats as a string.
        """
        if start is None:

            if format is None:
                format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(format)

        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)

        if format is None:
            format = self.DEFAULT_TIME_FORMAT

        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str
