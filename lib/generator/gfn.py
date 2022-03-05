import torch
import torch.nn.functional as F

from lib.generator.base import GeneratorBase
from lib.model.mlp import MLP

LOGINF = 1000


class FMGFlowNetGenerator(GeneratorBase):    
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.leaf_coef = args.gen_leaf_coef
        self.out_coef = args.gen_output_coef
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.pad_tok = 2
        self.num_tokens = args.vocab_size
        self.max_len = args.gen_max_len
        self.balanced_loss = args.gen_balanced_loss == 1
        if args.gen_model_type == "mlp":
            self.model = MLP(num_tokens=self.num_tokens, 
                            num_outputs=self.num_tokens, 
                            num_hid=args.gen_num_hidden,
                            num_layers=2,
                            max_len=self.max_len,
                            dropout=0,
                            partition_init=args.gen_partition_init,
                            causal=args.gen_do_explicit_Z)
        self.model.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), args.gen_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))
        self.device = args.device
        self.tokenizer=tokenizer

    @property
    def Z(self):
        return self.model.Z

    def train_step(self, input_batch):
        batch = self.preprocess_state(input_batch)
        loss, info = self.get_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss, info
    
    def preprocess_state(self, input_batch):
        s = self.tokenizer.process(sum(input_batch["traj_states"], [])).to(self.device)
        if self.args.gen_model_type == "mlp":
            inp_x = F.one_hot(s, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            s = inp.reshape(s.shape[0], -1).to(self.device).detach()
        a = torch.tensor(sum(input_batch["traj_actions"], [])).to(self.device)
        r = torch.tensor(sum(input_batch["traj_rewards"], [])).to(self.device).clamp(min=0)
        d = torch.tensor(sum(input_batch["traj_dones"], [])).to(self.device)
        tidx = [[-2]]
        # The index of s in the concatenated trajectories
        for i in input_batch["traj_states"]:
            tidx.append(torch.arange(len(i) - 1) + tidx[-1][-1] + 2)
        tidx = torch.cat(tidx[1:]).to(self.device)
        return s, a, r, d, tidx

    def get_loss(self, batch):
        s, a, r, d, tidx = batch
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        if self.args.gen_model_type == "mlp":
            if self.args.task == "tfbind":
                Q = self.model(s, s.gt(3))
            elif self.args.task == "gfp":
                Q = self.model(s, s.gt(19))
            else:
                Q = self.model(s, s.gt(20))
        qsa = torch.logaddexp(Q[tidx, a], torch.log(self.loss_eps))
        qsp = torch.logsumexp(Q[tidx+1], 1)
        qsp = qsp * (1-d) - LOGINF * d
        outflow = torch.logaddexp(torch.log(r + self.loss_eps), qsp)

        loss = (qsa - outflow).pow(2)
        leaf_loss = (loss * d).sum() / d.sum()
        flow_loss = (loss * (1-d)).sum() / (1-d).sum()

        if self.balanced_loss:
            loss = leaf_loss * self.leaf_coef + flow_loss
        else:
            loss = loss.mean()
        if loss.isnan():
            print(s)
            print(Q)
            print(r)
            print(qsa)
            print(qsp)
            import pdb; pdb.set_trace();
        return loss, {"leaf_loss": leaf_loss, "flow_loss": flow_loss}

    def forward(self, x, lens, return_all=False, coef=1, pad=2):
        if self.args.gen_model_type == "mlp":
            inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            inp = inp.reshape(x.shape[0], -1).to(self.device)
            out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef
            return out    
        out = self.model(x.swapaxes(0,1), x.eq(pad), lens=lens, return_all=return_all) * self.out_coef
        return out


class TBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.leaf_coef = args.gen_leaf_coef
        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.pad_tok = 1
        self.num_tokens = args.vocab_size
        self.max_len = args.gen_max_len
        self.tokenizer=tokenizer
        self.model = MLP(num_tokens=self.num_tokens, 
                                num_outputs=self.num_tokens, 
                                num_hid=1024,
                                num_layers=2,
                                max_len=self.max_len,
                                dropout=0,
                                partition_init=args.gen_partition_init,
                                causal=args.gen_do_explicit_Z)
        self.model.to(args.device)
        self.opt = torch.optim.Adam(self.model.model_params(), args.gen_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), args.gen_Z_learning_rate, weight_decay=args.gen_L2,
                            betas=(0.9, 0.999))
        self.device = args.device
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def train_step(self, input_batch):
        loss, info = self.get_loss(input_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        return loss, info

    @property
    def Z(self):
        return self.model.Z
    
    def get_loss(self, batch):
        strs, r = zip(*batch["bulk_trajs"])
        
        s = self.tokenizer.process(strs).to(self.device)
        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        if self.args.gen_model_type == 'mlp':
            inp_x = F.one_hot(s, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(s.shape[0], -1).to(self.device).detach()
            if self.args.task == "amp":
                lens = [self.max_len for i in s]
            else:
                lens = [len(i) for i in strs]
            pol_logits = self.logsoftmax2(self.model(x, None, return_all=True, lens=lens))[:-1]
            
            if self.args.task == "amp" and s.shape[1] != self.max_len:
                s = F.pad(s, (0, self.max_len - s.shape[1]), "constant", 21)
                mask = s.eq(21)
            else:
                mask = s.eq(self.num_tokens)
            s = s.swapaxes(0, 1)
            n = (s.shape[0] - 1) * s.shape[1]
        seq_logits = (pol_logits
                        .reshape((n, self.num_tokens))[torch.arange(n, device=self.device),(s[1:,].reshape((-1,))).clamp(0, self.num_tokens-1)]
                        .reshape(s[1:].shape)
                        * mask[:,1:].swapaxes(0,1).logical_not().float()).sum(0)
        # p(x) = R/Z <=> log p(x) = log(R) - log(Z) <=> log p(x) - log(Z)
        loss = (self.model.Z + seq_logits - r.clamp(min=self.reward_exp_min).log()).pow(2).mean()

        return loss, {}

    def forward(self, x, lens, return_all=False, coef=1, pad=2):
        if self.args.gen_model_type == "mlp":
            inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            inp = inp.reshape(x.shape[0], -1).to(self.device)
            out = self.model(inp, None, lens=lens, return_all=return_all) * self.out_coef
            return out    