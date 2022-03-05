import pickle
import gzip
from torch.utils.tensorboard import SummaryWriter
import copy

def get_logger(args):
    if args.enable_tensorboard:
        return TensorboardLogger(args)
    else:
        return Logger(args)

class Logger:
    def __init__(self, args):
        self.data = {}
        self.args = copy.deepcopy(vars(args))
        self.context = ""

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))


class TensorboardLogger(Logger):
    def __init__(self, args):
        self.data = {}
        self.context = ""
        self.args = copy.deepcopy(vars(args))
        self.writer = SummaryWriter(log_dir=args.tb_log_dir, comment=f"{args.name}")
        print(self.args)
        self.writer.add_hparams(self.args, {})

    def set_context(self, context):
        self.context = context

    def add_scalar(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]
        self.writer.add_scalar(key, value, len(self.data[key]))

    def add_object(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({'logged_data': self.data, 'args': self.args}, gzip.open(save_path, 'wb'))
        self.writer.flush()
