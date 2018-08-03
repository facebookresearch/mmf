from pythia.utils.flags import flags
from pythia.utils.configuration import Configuration
from pythia.task_loader import TaskLoader


class Trainer:
    def __init__(self):
        parser = flags.get_parser()
        self.args = parser.parse_args()

    def load(self):
        self.load_config()
        self.load_task()

        self.load_model()

    def load_config(self):
        self.configuration = Configuration(self.args.config)
        self.configuration.update_with_args(self.args)

        self.config = self.configuration.get_config()

    def load_task(self):
        self.task_loader = TaskLoader(self.config)

    def train(self):
        return
