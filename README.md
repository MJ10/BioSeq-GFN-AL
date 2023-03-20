# GFlowNets for Biological Sequence Design

This repo contains code for the paper [Biological Sequence Design with GFlowNets](http://arxiv.org/abs/2203.04115). 

The code has been extracted from an internal repository of the Mila Molecule Discovery project with some changes, so the hyperparameters might vary. Original commits are lost here, but the credit goes to [@MJ10](https://github.com/MJ10) and [@bengioe](https://github.com/bengioe). **There are some stability issues in training for the GFP task with this repo.**

## Setup
The code has been tested with Python 3.7 with CUDA 10.2 and CUDNN 8.0.

1. Install design-bench from our fork [`MJ10/design-bench`](https://github.com/MJ10/design-bench). This fork only changes some dependencies and resolves some minor changes to make it compatible with our code. To install clone the repo and run `pip install -e .` in the directory where the repo is cloned.
2. Instal the clamp-common-eval library from [MJ10/clamp-gen-data](https://github.com/MJ10/clamp-gen-data). This library handles the loading of the AMP data as well as oracles. To install clone the repo and run `pip install -r requirements.txt && pip install -e .` in the directory where the repo is cloned.
3. Run `pip install -r requirements.txt` in this directory to install the remaining packages.

## Running the code
`run_amp.py`, `run_gfp.py`, and `run_tfbind.py` are the entry points for the experiments.

Example:
```bash
python run_tfbind.py --gen_do_explicit_Z 1 --acq_fn ucb --gen_num_iterations 2500 --gen_reward_exp 8 --gen_data_sample_per_step 8 --proxy_num_iterations 10000 --gen_Z_learning_rate 1e-1 --gen_learning_rate 1e-3
```

Please reach out to Moksh Jain, [mokshjn00@gmail.com](mokshjn00@gmail.com) for any issues, comments, questions or suggestions.
