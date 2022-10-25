import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import hydra
from lib.utils.tokenizers import str_to_tokens, tokens_to_str

from lib.model.mlp import MLP


class LVMN(nn.Module):
    def __init__(self, args, tokenizer, device, **kwargs):
        super().__init__()
        self.args = args
        self.device = device
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        # self.sigmoid = nn.Sigmoid()
        self.proxy_num_iterations = args.train_steps
        # P(o|x) 
        self.p_o_x = hydra.utils.instantiate(args.model)
        self.p_o_x.to(self.device)
        
        # P(c|o,m,conc) 
        self.p_c = hydra.utils.instantiate(args.model)
        self.p_c.to(self.device)
        
        self.opt = torch.optim.Adam(list(self.p_o_x.parameters()) + list(self.p_c.parameters()), 
            self.args.learning_rate, weight_decay=self.args.L2)

    def fit(self, data, reset=False):
        losses = []
        test_losses = []
        best_params = None
        best_loss = 1e6
        early_stop_tol = self.args.early_stop_tol
        early_stop_count = 0
        epoch_length = 100
        if reset:
            self.init_model()
    
        for it in tqdm(range(self.proxy_num_iterations)):
            x, y = data.sample(self.args.batch_size)

            x = str_to_tokens(x, self.tokenizer).to(self.device).t()
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            # try: 
            output = self.model(x, None).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            
            losses.append(loss.item())
            # self.args.logger.add_scalar("proxy_train_loss", loss.item())
            
            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                acc = []
                for j in range(len(vx) // 256):
                    x = str_to_tokens(vx[j*256:(j+1)*256], self.tokenizer).to(self.device).t()
                    y = torch.tensor(vy[j*256:(j+1)*256], device=self.device, dtype=torch.float).reshape(-1)
                    output = self.model(x, None).squeeze(1)
                    if self.classification:
                        probs = self.sigmoid(output)
                        nll = -torch.log(probs) * y - torch.log(1-probs) * (1-y)
                        loss = nll
                        acc.append(((probs > 0.5) == y).sum())
                    else:
                        loss = (output -  y).pow(2)
                    vlosses.append(loss.sum().item())

                test_loss = np.sum(vlosses) / len(vx)
                test_losses.append(test_loss)
                print(test_loss, sum(acc) / len(vx))
                # self.args.logger.add_scalar("proxy_test_loss", test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [i.data.cpu().numpy() for i in self.model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(best_loss)
                    print('early stopping')
                    break

        if self.args.early_stop_to_best_params:
            # Put best parameters back in
            for i, besti in zip(self.model.parameters(), best_params):
                i.data = torch.tensor(besti).to(self.device)
        if self.args.save_path is not None:
            self.save(self.args.save_path)
        # self.args.logger.save(self.args.save_path, self.args)
        return {}

    def forward(self, curr_x, tok=None, uncertainty_call=False):
        if curr_x is not None and tok is None:
            x = str_to_tokens(curr_x, self.tokenizer).to(self.device).t()
        else:
            x = tok
        if uncertainty_call:
            ys = self.model(x, None).unsqueeze(0)
        else:
            self.model.eval()
            ys = self.model(x, None)
            self.model.train()
        return ys
    
    def forward_with_uncertainty(self, x):
        x = str_to_tokens(x, self.tokenizer).to(self.device).t()
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat([self.forward(None, x, True) for _ in range(self.args.proxy_num_dropout_samples)])
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))