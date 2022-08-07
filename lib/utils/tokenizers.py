import torch
import numpy as np
from pathlib import Path

from cachetools import cached, LRUCache

AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
RESIDUE_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + AMINO_ACIDS + ["0"]

TOY_VOCAB = [
    "A",
    "R"
]
TOY_ALPHABET = ["[PAD]", "[CLS]", "[UNK]", "[MASK]", "[SEP]"] + TOY_VOCAB + ["0"]

def padding_collate_fn(batch, padding_value=0.0):
    with torch.no_grad():
        if isinstance(batch[0], tuple):
            k = len(batch[0])
            x = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True, padding_value=padding_value
            )
            rest = [torch.stack([b[i] for b in batch]) for i in range(1, k)]
            return (x,) + tuple(rest)
        else:
            x = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=padding_value
            )
            return x

class IntTokenizer:
    def __init__(self, non_special_vocab, full_vocab, padding_token="[PAD]",
                 masking_token="[MASK]", bos_token="[CLS]", eos_token="[SEP]"):
        self.non_special_vocab = non_special_vocab
        self.full_vocab = full_vocab
        self.special_vocab = set(full_vocab) - set(non_special_vocab)
        self.lookup = {a: i for (i, a) in enumerate(full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(full_vocab)}
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]

        self.sampling_vocab = non_special_vocab
        self.non_special_idxs = [self.convert_token_to_id(t) for t in non_special_vocab]
        self.special_idxs = [self.convert_token_to_id(t) for t in self.special_vocab]

    @cached(cache=LRUCache(maxsize=int(1e4)))
    def encode(self, seq, use_sep=True):
        if seq.endswith("%"):
            seq = ["[CLS]"] + list(seq[:-1])
            seq += ["[SEP]"] if use_sep else []
            return [self.convert_token_to_id(c) for c in seq] + [4]
        else:
            seq = ["[CLS]"] + list(seq)
            seq += ["[SEP]"] if use_sep else []
            return [self.convert_token_to_id(c) for c in seq]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return self.convert_id_to_token(token_ids)

        tokens = []
        for t_id in token_ids:
            token = self.convert_id_to_token(t_id)
            if token in self.special_vocab and token not in ["[MASK]", "[UNK]"]:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    def convert_id_to_token(self, token_id):
        if torch.is_tensor(token_id):
            token_id = token_id.item()
        assert isinstance(token_id, int)
        return self.inverse_lookup.get(token_id, '[UNK]')

    def convert_token_to_id(self, token):
        unk_idx = self.lookup["[UNK]"]
        return self.lookup.get(token, unk_idx)

    def set_sampling_vocab(self, sampling_vocab=None, max_ngram_size=1):
        if sampling_vocab is None:
            sampling_vocab = []
            for i in range(1, max_ngram_size + 1):
                prod_space = [self.non_special_vocab] * i
                for comb in itertools.product(*prod_space):
                    sampling_vocab.append("".join(comb))
        else:
            new_tokens = set(sampling_vocab) - set(self.full_vocab)
            self.full_vocab.extend(list(new_tokens))
            self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
            self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}

        self.sampling_vocab = sampling_vocab


class ResidueTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(AMINO_ACIDS, RESIDUE_ALPHABET)

class ToyTokenizer(IntTokenizer):
    def __init__(self):
        super().__init__(TOY_VOCAB, TOY_ALPHABET)


def random_strings(num, min_len=200, max_len=250, alphabet=AMINO_ACIDS):
    strs = []
    for _ in range(num):
        length = np.random.randint(min_len, max_len + 1)
        idx = np.random.choice(len(alphabet), size=length, replace=True)
        strs.append("".join([alphabet[i] for i in idx]))
    return np.array(strs)

def str_to_tokens(str_array, tokenizer, use_sep=True):
    tokens = [
        torch.tensor(tokenizer.encode(x, use_sep)) for x in str_array
    ]
    batch = padding_collate_fn(tokens, tokenizer.padding_idx)
    return batch


def tokens_to_str(tok_idx_array, tokenizer):
    str_array = np.array([
        tokenizer.decode(token_ids).replace(' ', '') for token_ids in tok_idx_array
    ])
    return str_array