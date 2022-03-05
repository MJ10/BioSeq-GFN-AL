import numpy as np

from clamp_common_eval.defaults import get_test_oracle
import design_bench


def get_oracle(args):
    if args.task == "amp":
        return AMPOracleWrapper(args)
    elif args.task == "gfp":
        return GFPWrapper(args)
    elif args.task == "tfbind":
        return TFBind8Wrapper(args)


class AMPOracleWrapper:
    def __init__(self, args):
        self.oracle = get_test_oracle(args.oracle_split, 
                                        model=args.oracle_type, 
                                        feature=args.oracle_features, 
                                        dist_fn="edit", 
                                        norm_constant=args.medoid_oracle_norm)
        self.oracle.to(args.device)

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.oracle.evaluate_many(x[i*batch_size:(i+1)*batch_size])
            if type(s) == dict:
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()
        return np.float32(scores)


class GFPWrapper:
    def __init__(self, args):
        self.task = design_bench.make('GFP-Transformer-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size])).reshape(-1)
            scores += s.tolist()
        return np.float32(scores)

class TFBind8Wrapper:
    def __init__(self, args):
        self.task = design_bench.make('TFBind8-Exact-v0')

    def __call__(self, x, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(x) / batch_size))):
            s = self.task.predict(np.array(x[i*batch_size:(i+1)*batch_size]))
            scores += s.tolist()
        return np.array(scores)