class Registry:
    register = {

        # Mappings of task name to their respective classes
        # Use decorator "register_task" in pythia.core.decorators to regiter a
        # task class with a specific name
        # Further, use the name with the class is registered in the
        # command line or configuration to load that specific task
        'task_name_mapping': {},


        # Similar to the task_name_mapping above except that this
        # one is used to keep a mapping for dataset to its builder class.
        # Use "register_builder" decorator to register a builder
        'builder_name_mapping': {},
        'metric_name_mapping': {},
        'state': {}
    }

    @classmethod
    def register_task(cls, task_cls, name):
        cls.register['task_name_mapping'][name] = task_cls

    @classmethod
    def register_builder(cls, builder_cls, name):
        cls.register['builder_name_mapping'][name] = builder_cls

    @classmethod
    def register_metric(cls, func, name):
        cls.register['metric_name_mapping'][name] = func

    @classmethod
    def register(cls, name, obj):
        path = name.split('.')
        current = cls.register['state']

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_task_class(cls, name):
        return cls.register['task_name_mapping'].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.register['builder_name_mapping'].get(name, None)

    @classmethod
    def get_metric_func(cls, name):
        return cls.register['metric_name_mapping'].get(name, None)

    @classmethod
    def get(cls, name):
        name = name.split('.')
        value = cls.register['state']
        for subname in name:
            value = value.get(subname, None)
            if value is None:
                break
        return value

    @classmethod
    def unregister(cls, name):
        cls.register['state'].pop(name, None)
