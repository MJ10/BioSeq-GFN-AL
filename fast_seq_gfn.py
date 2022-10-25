import hydra
import wandb
import math
import time
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch.nn as nn
import itertools

from lib.utils.tokenizers import str_to_tokens, tokens_to_str
from lib.utils.distance import is_similar, edit_dist
from omegaconf import OmegaConf, DictConfig
from collections.abc import MutableMapping

from torch.distributions import Categorical
from tqdm import tqdm


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_run(cfg):
    trial_id = cfg.trial_id
    if cfg.job_name is None:
        cfg.job_name = '_'.join(randomname.get_name().lower().split('-') + [str(trial_id)])
    cfg.seed = random.randint(0, 100000) if cfg.seed is None else cfg.seed
    set_seed(cfg.seed)
    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve config interpolations
    cfg = DictConfig(cfg)
    # logger.write_hydra_yaml(cfg)

    print(OmegaConf.to_yaml(cfg))
    with open('hydra_config.txt', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    return cfg

def flatten_config(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class GFN:
    def __init__(self, cfg, tokenizer, **kwargs):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.setup_vars(kwargs)
        self.init_policy()

    def setup_vars(self, kwargs):
        cfg = self.cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Task stuff
        self.max_len = cfg.max_len
        self.min_len = cfg.min_len
        # GFN stuff
        self.train_steps = cfg.train_steps
        self.random_action_prob = cfg.random_action_prob
        self.batch_size = cfg.batch_size
        self.reward_min = cfg.reward_min
        self.gen_clip = cfg.gen_clip
        self.sampling_temp = cfg.sampling_temp
        self.sample_beta = cfg.sample_beta
        self.eval_batch_size = cfg.eval_batch_size
        # Eval Stuff
        # self.eval_freq = cfg.eval_freq
        # self.k = cfg.k
        # self.num_samples = cfg.num_samples
        self.eos_char = "[SEP]"
        self.pad_tok = self.tokenizer.convert_token_to_id("[PAD]")
        

    def init_policy(self):
        cfg = self.cfg
        self.model = hydra.utils.instantiate(cfg.model)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.model_params(), cfg.pi_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))
        self.opt_Z = torch.optim.Adam(self.model.Z_param(), cfg.z_lr, weight_decay=cfg.wd,
                            betas=(0.9, 0.999))

    def optimize(self, task, init_data=None):
        """
        optimize the task involving multiple objectives (all to be maximized) with 
        optional data to start with
        """
        losses, rewards = [], []
        pb = tqdm(range(self.train_steps))
        desc_str = "Evaluation := Reward: {:.3f} | Train := Loss: {:.3f} Rewards: {:.3f}"
        pb.set_description(desc_str.format(0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))

        for i in pb:
            loss, r = self.train_step(task, self.batch_size)
            losses.append(loss)
            rewards.append(r)
            
            # if i % self.eval_freq == 0:
            #     with torch.no_grad():
            #         self.evaluation(task)
                
            pb.set_description(desc_str.format(0, sum(losses[-10:]) / 10, sum(rewards[-10:]) / 10))
        
        return {
            'losses': losses,
            'train_rs': rewards
        }
    
    def train_step(self, task, batch_size):
        states, logprobs = self.sample(batch_size)

        r = self.process_reward(states, task).to(self.device)
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        # import pdb; pdb.set_trace();
        # TB Loss
        loss = (logprobs - self.sample_beta * r.clamp(self.reward_min).log()).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        return loss.item(), r.mean()


    def sample(self, episodes, train=True):
        states = [''] * episodes
        traj_logprob = torch.zeros(episodes).to(self.device)

        active_mask = torch.ones(episodes).bool().to(self.device)
        x = str_to_tokens(states, self.tokenizer, use_sep=False).to(self.device).t()[:1].long()
        lens = torch.zeros(episodes).long().to(self.device)
        uniform_pol = torch.empty(episodes).fill_(self.random_action_prob).to(self.device)

        for t in (range(self.max_len) if episodes > 0 else []):
            # import pdb;pdb.set_trace();
            logits = self.model(x, lens=lens, mask=None)
            
            if t <= self.min_len:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
                if t == 0:
                    traj_logprob += self.model.Z()

            cat = Categorical(logits=logits / self.sampling_temp)
            actions = cat.sample()
            if train and self.random_action_prob > 0:
                uniform_mix = torch.bernoulli(uniform_pol).bool()
                actions = torch.where(uniform_mix, torch.randint(int(t <= self.min_len), logits.shape[1], (episodes, )).to(self.device), actions)
            
            log_prob = cat.log_prob(actions) * active_mask
            lens += torch.where(active_mask, torch.ones_like(lens), torch.zeros_like(lens))
            traj_logprob += log_prob

            actions_apply = torch.where(torch.logical_not(active_mask), torch.zeros(episodes).to(self.device).long(), actions + 4)
            active_mask = torch.where(active_mask, actions != 0, active_mask)

            x = torch.cat((x, actions_apply.unsqueeze(0)), axis=0)
            if active_mask.sum() == 0:
                break
        states = tokens_to_str(x.t(), self.tokenizer)
        return states, traj_logprob
    
    def generate(self, num_samples, task):
        generated_seqs = []
        rewards = []
        while len(generated_seqs) < num_samples:
            with torch.no_grad():
                samples, _ = self.sample(self.eval_batch_size, train=False)
                r = self.process_reward(samples, task).cpu().numpy().tolist()
            generated_seqs.extend(samples.tolist())
            rewards.extend(r)
        return generated_seqs, np.array(rewards)

    def process_reward(self, seqs, task):
        return task(seqs).squeeze()

    def val_step(self, batch_size, task):
        overall_loss = 0.
        num_batches = len(self.val_split.inputs) // self.batch_size
        losses = 0
        for state, r in self.val_loader:
            state, r = x.to(self.device), y.to(self.device)
            logprobs = self._get_log_prob(states)

            r = self.process_reward(state, task).to(seq_logits.device)
            loss = (seq_logits - self.sample_beta * r.clamp(min=self.reward_min).log()).pow(2).mean()

            losses += loss.item()
        overall_loss += (losses)
        return overall_loss / len(self.simplex)

    def evaluation(self, task):
        val_loss = self.val_step(self.batch_size)
        samples = self.generate(self.eval_samples)
        rewards = self.process_reward(samples, task)
        return val_loss, rewards

    def _get_log_prob(self, states):
        lens = torch.tensor([len(z) + 2 for z in states]).long().to(self.device)
        x = str_to_tokens(states, self.tokenizer).to(self.device).t()
        mask = x.eq(self.tokenizer.padding_idx)
        logits = self.model(x, mask=mask.transpose(1,0), return_all=True, lens=lens, logsoftmax=True)
        seq_logits = (logits.reshape(-1, 21)[torch.arange(x.shape[0] * x.shape[1], device=self.device), (x.reshape(-1)-4).clamp(0)].reshape(x.shape) * mask.logical_not().float()).sum(0)
        seq_logits += self.model.Z()
        return seq_logits

class OffsetTask:
    def __init__(self, task, offset):
        self.task = task
        self.offset = offset
    
    def __call__(self, x):
        return self.offset + self.task(x)

class ClassificationTask:
    def __init__(self, task):
        self.task = task
        self.sigmoid = nn.Sigmoid()
    
    def __call__(self, x):
        return self.sigmoid(self.task(x))

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def evaluate_samples(seqs, scores, k):
    indices = np.argsort(scores)[::-1][:k]
    topk_scores = scores[indices]
    topk_prots = np.array(seqs)[indices]
    print(topk_prots)
    diversity_score = mean_pairwise_distances(topk_prots)
    score = topk_scores.mean()
    return score, diversity_score

@hydra.main(config_path='./config', config_name='amp_logrank')
def main(config):
    random.seed(None)  # make sure random seed resets between multirun jobs for random job-name generation

    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
        
    # init_run()

    wandb.init(project='bioseqgfn', config=log_config, mode=config.wandb_mode,
               group=config.group_name, name=config.exp_name, tags=config.exp_tags)
    config['job_name'] = wandb.run.name
    config = init_run(config)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    dataset = hydra.utils.instantiate(config.dataset)
    tokenizer = hydra.utils.instantiate(config.tokenizer)
    proxy = hydra.utils.instantiate(config.proxy, config.proxy, device=device, tokenizer=tokenizer)
    if config.load_proxy_path is not None and os.path.exists(config.load_proxy_path):
        # import pdb; pdb.set_trace();
        proxy.load(config.load_proxy_path)
    else:
        proxy.fit(dataset)
    if config.use_offset:
        task = OffsetTask(proxy, -dataset.offset)
    else:
        task = ClassificationTask(proxy)
    generator = GFN(config.gfn, tokenizer)
    generator.optimize(task)

    samples, scores = generator.generate(config.num_samples, task)
    score, diversity_score = evaluate_samples(samples, scores, config.k)
    print("Score: {}, Diversity: {}".format(score, diversity_score))
    # scores = proxy.score(samples)
    wandb.finish()

if __name__ == "__main__":
    main()