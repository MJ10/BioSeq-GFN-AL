import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from clamp_common_eval.defaults import get_default_data_splits
import design_bench

import os.path as osp

from lib.dataset.base import Dataset


class AMPRegressionDataset(Dataset):
    def __init__(self, split, nfold, args, oracle):
        super().__init__(args, oracle)
        self._load_dataset(split, nfold)
        self._compute_scores(split)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self, split, nfold):
        source = get_default_data_splits(setting='Target')
        self.data = source.sample(split, -1)
        self.nfold = nfold
        if split == "D1": groups = np.array(source.d1_pos.group)
        if split == "D2": groups = np.array(source.d2_pos.group)
        if split == "D": groups = np.concatenate((np.array(source.d1_pos.group), np.array(source.d2_pos.group)))

        n_pos, n_neg = len(self.data['AMP']), len(self.data['nonAMP'])
        pos_train, pos_valid = next(GroupKFold(nfold).split(np.arange(n_pos), groups=groups))
        neg_train, neg_valid = next(GroupKFold(nfold).split(np.arange(n_neg),
                                                            groups=self.rng.randint(0, nfold, n_neg)))
        
        pos_train = [self.data['AMP'][i] for i in pos_train]
        neg_train = [self.data['nonAMP'][i] for i in neg_train]
        pos_valid = [self.data['AMP'][i] for i in pos_valid]
        neg_valid = [self.data['nonAMP'][i] for i in neg_valid]
        self.train = pos_train + neg_train
        self.valid = pos_valid + neg_valid
    
    def _compute_scores(self, split):
        loaded = self._load_precomputed_scores(split)
        if loaded:
            return
        self.train_scores = self.oracle(self.train)
        self.valid_scores = self.oracle(self.valid)
        if self.args.save_scores:
            np.save(osp.join(self.args.save_scores_path, "reg" + split+"train_scores.npy") , self.train_scores)
            np.save(osp.join(self.args.save_scores_path, "reg" + split+"val_scores.npy"), self.valid_scores)


    def _load_precomputed_scores(self, split):
        if osp.exists(osp.join(self.args.load_scores_path)):
            try: 
                self.train_scores = np.load(osp.join(self.args.load_scores_path, "reg" + split+"train_scores.npy"))
                self.valid_scores = np.load(osp.join(self.args.load_scores_path, "reg" + split+"val_scores.npy"))
            except:
                return False
            return True
        else:
            return False


    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return ([self.train[i] for i in indices],
                [self.train_scores[i] for i in indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        train, val = [], []
        for x, score in zip(samples, scores):
            if np.random.uniform() < (1/self.nfold):
                self.valid.append(x)
                val.append(score)
            else:
                self.train.append(x)
                train.append(score)
        self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
        self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
    
    def _top_k(self, data, k):
        topk_scores, topk_prots = [], []
        indices = np.argsort(data[1])[::-1][:k]
        topk_scores = np.concatenate((topk_scores, data[1][indices]))
        topk_prots = np.concatenate((topk_prots, np.array(data[0])[indices]))
        return topk_prots.tolist(), topk_scores

    def top_k(self, k):
        data = (self.train + self.valid, np.concatenate((self.train_scores, self.valid_scores), axis=0))
        return self._top_k(data, k)

    def top_k_collected(self, k):
        scores = np.concatenate((self.train_scores[self.train_added:], self.valid_scores[self.val_added:]))
        seqs = np.concatenate((self.train[self.train_added:], self.valid[self.val_added:]))
        data = (seqs, scores)
        return self._top_k(data, k)


class TFBind8Dataset(Dataset):
    def __init__(self, args, oracle):
        super().__init__(args, oracle)
        self._load_dataset()
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self):
        task = design_bench.make('TFBind8-Exact-v0')
        x = task.x
        y = task.y.reshape(-1)
        self.train, self.valid, self.train_scores, self.valid_scores  = train_test_split(x, y, test_size=0.1, random_state=self.rng)

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


class GFPDataset(Dataset):
    def __init__(self, args, oracle):
        super().__init__(args, oracle)
        self._load_dataset()
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self):
        task = design_bench.make('GFP-Transformer-v0')
        task.map_normalize_y()
        x = task.x
        y = task.y.reshape(-1) 
        self.train, self.valid, self.train_scores, self.valid_scores  = train_test_split(x, y, test_size=0.2, random_state=self.rng)

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