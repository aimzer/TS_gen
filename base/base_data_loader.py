from utils.logger import get_logger

class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.train_counter = 0
        self.test_counter = 0
        self.log = get_logger()

    def next_train_batch(self, batch_size):
        raise NotImplementedError

    def next_test_batch(self, batch_size):
        raise NotImplementedError

    def reset_counter(self):
        self.train_counter = 0
        self.test_counter = 0