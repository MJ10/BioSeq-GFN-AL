import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from clamp_common_eval.defaults import get_default_data_splits
import pandas as pd
import os
import random
import pickle

def one_hot(loc, num):
    temp = np.zeros(num)
    temp[loc] = 1
    return temp

conc_mapping = [
    0.1, 0.5, 1, 3
]


class AMPClassificationDataset:
    def __init__(self, split, nfold):
        self.rng = np.random.RandomState(142857)
        self._load_dataset(split, nfold)
        # self._compute_scores(split)
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
        self.train_scores = [1] * len(pos_train) + [0] * len(neg_train)
        self.valid_scores = [1] * len(pos_valid) + [0] * len(neg_valid)
    
    # def _compute_scores(self, split):
    #     loaded = self._load_precomputed_scores(split)
    #     if loaded:
    #         return
    #     self.train_scores = self.oracle(self.train)
    #     self.valid_scores = self.oracle(self.valid)
    #     if self.args.save_scores:
    #         np.save(osp.join(self.args.save_scores_path, "reg" + split+"train_scores.npy") , self.train_scores)
    #         np.save(osp.join(self.args.save_scores_path, "reg" + split+"val_scores.npy"), self.valid_scores)

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



class AMPExperimentalDataset:
    def __init__(self, path, load_cache=False, save_cache=True, cache_path=None):
        self.rng = np.random.RandomState(142857)
        self.load_cache = load_cache
        self.save_cache=save_cache
        self.cache_path = cache_path
        self._load_dataset(path)
        # self._compute_scores(split)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self, base_path):
        if not self.load_cache or self.cache_path is None:
            print("loading data")
            
            rank_df_m1 = pd.read_csv(os.path.join(base_path, "logrank_liquid.csv"))
            rank_df_m2 = pd.read_csv(os.path.join(base_path, "logrank_plate.csv"))

            m1_good =  rank_df_m1.iloc[:self.num_bootsrap]
            m1_bad = rank_df_m1.iloc[-self.num_bootsrap:] 

            m2_good =  rank_df_m2.iloc[:self.num_bootsrap]
            m2_bad = rank_df_m2.iloc[-self.num_bootsrap:]

            df_m1 = pd.read_csv(os.path.join(base_path, "20220531-AMPs-lib-liquid-samples.csv"))
            df_m2 = pd.read_csv(os.path.join(base_path, "20220531-AMPs-lib-plate-samples.csv"))
            data = []
            for i in range(len(df_m1)):
                record = df_m1.iloc[i]
                for j in range(4,8):
                    data.append((record[1], [1,0], conc_mapping[j-4], np.log2(1e-7 + record[j]) - np.log2(1e-7 + record[3])))

            for i in range(len(df_m2)):
                record = df_m2.iloc[i]
                for j in range(3,8):
                    data.append((record[1], [0,1], conc_mapping[j-4], np.log2(1e-7 + record[j]) - np.log2(1e-7 + record[3])))
            # import pdb;pdb.set_trace();
            
            if self.save_cache:
                with open(self.cache_path, "wb") as f:
                    pickle.dump(data, f)
        else:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
        random.shuffle(data)
        num_examples = len(data)
        self.train = data[:int(0.85 * num_examples)]
        self.valid = data[int(0.85 * num_examples):] 


    def sample(self, n):
        indices = np.random.randint(0, len(self.train), n)
        return list(zip(*[self.train[i] for i in indices]))

    def validation_set(self):
        return list(zip(*self.valid))

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



class AMPLogRankDataset:
    def __init__(self, path, medium):
        self.rng = np.random.RandomState(142857)
        self._load_dataset(path, medium)
        # self._compute_scores(split)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self, base_path, medium):
        df = pd.read_csv(os.path.join(base_path, "logrank_{}.csv".format(medium)))
        data = df["seq"].to_numpy()
        scores = df["logrank"].to_numpy()
        scores = scores - scores.mean() / (scores.std() + 1e-4)
        self.offset = scores.min()
        train_X, val_X, train_Y, val_Y = train_test_split(data, scores, test_size=0.2)
        self.train = train_X.tolist()
        self.train_scores = train_Y.tolist()
        self.valid = val_X.tolist()
        self.valid_scores = val_Y.tolist()
        self.training_data = (self.train, self.train_scores)
        # import pdb; pdb.set_trace();

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