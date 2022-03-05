import numpy as np

class Dataset:
    def __init__(self, args, oracle):
        self.oracle = oracle
        self.args = args
        self.rng = np.random.RandomState(142857)

    def sample(self, num_samples, ratio=0.5):
        raise NotImplementedError()
    
    def validation_set(self, ratio=None):
        raise NotImplementedError()

    def add(self, batch):
        raise NotImplementedError()
    
    def top_k(self, k):
        raise NotImplementedError()