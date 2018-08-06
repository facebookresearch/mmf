class Meter:
    METRICS_TO_FUNC_MAPPING = {
        'accuracy': 'accuracy'
    }

    def __init__(self, meter_types):
        if not isinstance(meter_types, list):
            meter_types = [meter_types]

        self.meter_types = meter_types
        self.avg_meter_values = []

        for _ in self.meter_types:
            self.avg_meter_values.append(0)

        self.iteration_count = 0

    def __call__(self, expected, output):
        values = []
        self.iteration_count += 1

        for i in range(self.meter_types):
            meter_type = self.meter_types[i]
            func = getattr(self, self.METRICS_TO_FUNC_MAPPING[meter_type])
            value = func(expected, output)
            values.append(value)
            self.avg_meter_values[i] += value

        return values
