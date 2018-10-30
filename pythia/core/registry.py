class Registry:
    mapping = {

        # Mappings of task name to their respective classes
        # Use decorator "register_task" in pythia.core.decorators to regiter a
        # task class with a specific name
        # Further, use the name with the class is registered in the
        # command line or configuration to load that specific task
        'task_name_mapping': {},


        # Similar to the task_name_mapping above except that this
        # one is used to keep a mapping for dataset to its builder class.
        # Use "register_builder" decorator to mapping a builder
        'builder_name_mapping': {},
        'model_name_mapping': {},
        'metric_name_mapping': {},
        'state': {}
    }

    @classmethod
    def register_task(cls, name):
        def wrap(task_cls):
            cls.mapping['task_name_mapping'][name] = task_cls
            return task_cls
        return wrap

    @classmethod
    def register_builder(cls, name):
        def wrap(builder_cls):
            cls.mapping['builder_name_mapping'][name] = builder_cls
            return builder_cls
        return wrap

    @classmethod
    def register_metric(cls, name):
        def wrap(func):
            cls.mapping['metric_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(func):
            cls.mapping['model_name_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register(cls, name, obj):
        path = name.split('.')
        current = cls.mapping['state']

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping['task_name_mapping'].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping['builder_name_mapping'].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping['model_name_mapping'].get(name, None)

    @classmethod
    def get_metric_func(cls, name):
        return cls.mapping['metric_name_mapping'].get(name, None)

    @classmethod
    def get(cls, name, default=None):
        name = name.split('.')
        value = cls.mapping['state']
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break
        return value

    @classmethod
    def unregister(cls, name):
        cls.mapping['state'].pop(name, None)
