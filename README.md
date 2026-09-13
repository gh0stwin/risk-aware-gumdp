# Risk-Aware General-Utility Markov Decision Processes

## Publication

This repository contains the official implementation of the paper "Risk-Aware General-Utility Markov Decision Processes" by Pedro P. Santos, Fábio Vital, Alberto Sardinha, and Francisco S. Melo.

[![arXiv](https://img.shields.io/badge/arXiv-2607.09298-b31b1b.svg)](https://arxiv.org/abs/2607.09298)
![Static Badge](https://img.shields.io/badge/conference-RLC_2026-blue?logo=test&label=Conference&color=%231b3a9e&link=https%3A%2F%2Frlj.cs.umass.edu%2F2026%2Fpapers%2FPaper17.html)
[![Static Badge](https://img.shields.io/badge/Blog_Post-brightgreen?logo=test&label=Digest)](https://ppsantos.github.io/posts/risk-aware-gumdps/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Installation
1. Assuming [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) is already installed.

2. Install python environment

```bash
conda env create -f environment.yml -y
conda activate risk_aware_gumdp
export CONDA_ENVS_PATH="<miniconda3-path>"  # normally at `<home-dir>/miniconda3/envs`
poetry config virtualenvs.path $CONDA_ENVS_PATH
poetry config virtualenvs.create false
poetry install
```

## Run Experiments

### Illustrative Environments

#### Standard MDP

##### ERM-BI - Baseline

```bash
python src/risk_aware_gumdp/simulations/erm_backward_induction.py --env=linear_mdp --H=20 --N=100 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_backward_induction.py --env=linear_mdp --H=20 --N=100 --erm_beta=5.0
python src/risk_aware_gumdp/simulations/erm_backward_induction.py --env=linear_mdp --H=20 --N=100 --erm_beta=10.0
```

##### ERM-MCTS - Ours

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=linear_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=linear_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=5.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=linear_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=10.0
```

#### MSEE

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=entropy_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=entropy_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=25.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=entropy_mdp --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=50.0
```

#### IL

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=imitation_learning_mdp --H=20 --N=100 --n_iter_per_timestep=2000 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=imitation_learning_mdp --H=20 --N=100 --n_iter_per_timestep=2000 --erm_beta=40.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=imitation_learning_mdp --H=20 --N=100 --n_iter_per_timestep=2000 --erm_beta=80.0
```

#### MO

##### Weighted Cost

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_weighted --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_weighted --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=20.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_weighted --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=40.0
```

##### Max Cost

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_max --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_max --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=10.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_max --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=20.0
```

##### Min Cost

```bash
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_min --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=0.1
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_min --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=20.0
python src/risk_aware_gumdp/simulations/erm_mcts.py --env=multi_objective_mdp_min --H=20 --N=100 --n_iter_per_timestep=500 --erm_beta=50.0
```

### Grid Environments

#### MSEE

```bash
./scripts/erm_mcts.sh --env=msee --h=200 --gamma=0.99 --steps=1024 --beta=0.001 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=msee --h=200 --gamma=0.99 --steps=1024 --beta=1.0 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=msee --h=200 --gamma=0.99 --steps=1024 --beta=1000.0 --bs=128 --micro_bs=4
```

#### IL

```bash
./scripts/erm_mcts.sh --env=il --h=99 --gamma=0.99 --steps=1024 --beta=0.002 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=il --h=99 --gamma=0.99 --steps=1024 --beta=1.0 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=il --h=99 --gamma=0.99 --steps=1024 --beta=500.0 --bs=128 --micro_bs=4
```

#### MO

```bash
./scripts/erm_mcts.sh --env=mo --h=40 --gamma=0.99 --steps=1024 --beta=0.125 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=mo --h=40 --gamma=0.99 --steps=1024 --beta=1.0 --bs=128 --micro_bs=4
./scripts/erm_mcts.sh --env=mo --h=40 --gamma=0.99 --steps=1024 --beta=8.0 --bs=128 --micro_bs=4
```

### Generate Plots

After running the desired experiments, check the notebooks inside `/notebooks` to produce the respective plots.

