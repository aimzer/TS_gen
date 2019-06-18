from base.base_data_loader import BaseDataLoader
import numpy as np


class ExampleDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ExampleDataLoader, self).__init__(config)
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]