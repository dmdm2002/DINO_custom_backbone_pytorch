from RunModules.Train import Trainer
from Utils.Option import Param


class handler(Param):
    def __init__(self):
        super(handler, self).__init__()

    def starter(self):
        if self.run_type == 0:
            tr = Trainer()
            tr.run()


if __name__ == '__main__':
    hand = handler()
    hand.starter()
