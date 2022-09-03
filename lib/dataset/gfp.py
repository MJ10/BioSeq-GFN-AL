import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from design_bench.datasets.discrete.gfp_dataset import GFPDataset
# dataset = GFPDataset()
AMINO_ACIDS = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

class GFPCLSDataset():
    def __init__(self):
        self.rng = np.random.RandomState(142857)
        self._load_dataset()
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self):
        dataset = GFPDataset()
        # import pdb;pdb.set_trace();
        dataset.map_normalize_y()
        import pdb;pdb.set_trace();
        x = self._process_x(dataset.x)

        y = dataset.y.reshape(-1)
        self.offset = y.min() 
        self.train, self.valid, self.train_scores, self.valid_scores  = train_test_split(x, y, test_size=0.2, random_state=self.rng)

    def _process_x(self, data):
        processed_data = []
        for i in range(len(data)):
            processed_data.append("".join([AMINO_ACIDS[c] for c in data[i]]))
        return processed_data

    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        train_seq, val_seq = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/10):
                val_seq.append(x)
                val.append(score)
            else:
                train_seq.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
        self.train = np.concatenate((self.train, train_seq), axis=0)
        self.valid = np.concatenate((self.valid, val_seq), axis=0)
    
    def _tostr(self, seqs):
        return ["".join([str(i) for i in x]) for x in seqs]

    def _top_k(self, data, k):
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = data[1][indices]
        topk_prots = np.array(data[0])[indices]
        return self._tostr(topk_prots), topk_scores

    def top_k(self, k):
        data = (np.concatenate((self.train, self.valid), axis=0), np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]), axis=0)
        data = (seqs, scores)
        return self._top_k(data, k)