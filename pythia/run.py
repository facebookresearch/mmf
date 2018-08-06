from pythia.trainer import Trainer


def run():
    trainer = Trainer()
    trainer.load()
    trainer.train()


if __name__ == '__main__':
    run()
