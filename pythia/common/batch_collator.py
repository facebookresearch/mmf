from .sample import SampleList


class BatchCollator:
    def __call__(self, batch):
        return SampleList(batch)
