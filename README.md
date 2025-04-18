# Hydraulis

Codes for our submission entitled *Hydraulis: Balancing Large Transformer Model Training via Co-designing Parallel Strategies and Data Assignment*. This repository includes our system implementation and scripts for analyzing data imbalance, profiling parallel schemes, proposing data distribution-aware strategies, and conducting end-to-end training with dynamic heterogeneous strategies and two-stage sequence assignment. Additionally, we provide a standalone script for running the algorithms to evaluate its time cost and scalability.

## 1. Build & Compile Our System

To facilitate training with dynamic heterogeneous strategies, we developed a prototype deep learning system. It supports optimization-propagation disaggregation and utilizes subgraph abstraction. We use `cmake >= 3.24` to compile it. Essential third-party packages such as `flash-attn`, `onednn`, and `cutlass` have been prepared and will be compiled automatically. You can also configure paths to pre-built modules by modifying the `cmake/config_refactor.cmake` file. 

For GPU communication, we utilize `NCCL` as the backend library, included as a submodule. To download the latest version of `NCCL`, you can just execute `git submodule update --init --recursive`. If you prefer to use a different version of `NCCL`, you may also set the environment variable `NCCL_ROOT` to your desired path: `export NCCL_ROOT=/your/path/`. Furthermore, we utilize `grpc` for efficient CPU communication, please install it in advance and ensure the `set(CMAKE_PREFIX_PATH /your/path/to/grpc)` in `hydraulis/CMakeLists.txt` is correct.

The commands to compile our system are as follows:

```bash
mkdir -p build && cd build
cmake ..
make -j 32
cd ..
source hydraulis.exp
```

As for ILP problems solving, we use the `PuLP` library. You can simply download it by:

```bash
pip install pulp
```

## 2. Dataset Preparation

In the `Hydraulis` paper, we use the `CommonCrawl` (available at [Hugging Face](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)) and `GitHub` (available at [Hugging Face](https://huggingface.co/datasets/codeparrot/github-code)) datasets. You can also pick any dataset you want to run our scripts. For the `CommonCrawl` dataset, you can download it via the link provided above or use our provided script:

```bash
cd examples/data_utils
bash data/create_web_dataset.sh
```

## 3. Imbalance Analysis

You can analyze the imbalance issues described in our paper by running:

```
python examples/analyze_dataset.py
```

This will calculate the sequence length distribution of the dataset. You can also use the `draw_sample_simulation` function in it to recreate the figure in our paper.

## 4. Parallel Schemes Profiling

We provide a one-click profiling script to profile the memory and latency of different parallel schemes:

```
cd examples
bash scripts/profile_all_parallel_schemes.py
```

You can modify the parallel methods and their range in the script to narrow down or expand the parallel scheme space. By default, we use:

```bash
CP_VALUES=(1 2 4 8)
TP_VALUES=(1 2 4 8)
PP_VALUES=(1 2 4 8)
```

Based on the profiling statistics, you can utilize the following `.py` file to construct the parallel scheme space. This will generate a `parallel_scheme_space.json`, which will be employed in the data distribution-aware strategy proposal and the two-stage sequence assignment during end-to-end training:

```bash
python build_parallel_scheme_space.py
```

Additionally, we provide the specific parallel scheme statistics: `parallel_scheme_space_7b.json`, `parallel_scheme_space_13b.json`, `parallel_scheme_space_32b.json`. They are used in our paper's experiments with the `7b`, `13b`, and `32b` LLaMA models, respectively. You can also re-generate them directly using:

```python
python build_parallel_scheme_space_7b.py
python build_parallel_scheme_space_13b.py
python build_parallel_scheme_space_32b.py
```

## 5. Data Distribution-aware Strategy Proposal

Based on the statistics, you can use the following `.py` file to propose heterogeneous strategies with one click:

```bash
cd examples/strategy
python strategy_proposal.py
```

## 6. End-to-end Training

Eventually, you can conduct end-to-end training with dynamic heterogeneous strategies and two-stage sequence assignment using the following script:

```bash
cd examples/strategy
bash scripts/e2e_train.sh
```

You can manually adjust certain values in the script. By default, we provide the settings used in **Figure 14 of the case study**:

```bash
# Example
# 64 GPUs 32B CommonCrawl
# The "case study (Figure 14)" exp setup in our paper
# Model, devices and workload
NUM_GPUS=64
MODEL_SIZE=32b
GLOBAL_TOKEN_NUM=200000
MAX_SEQ_LEN=32768
# Dataset (CommonCrawl)
ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/data.json 
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# Dynamic Heterogeneous Strategies (proposed by our algorithm)
# A two-dimensional list, where each sublist represents a heterogeneous strategy and each element in the list indicates a pipeline using a <CP, TP, PP> parallel scheme
MULTI_CP_TP_PP_LIST="[[(1, 16, 1), (1, 16, 1), (1, 16, 1), (1, 16, 1)], [(1, 16, 1), (1, 8, 3), (1, 8, 3)], [(1, 16, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1), (1, 8, 1)], [(1, 16, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 8, 3), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 8, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1), (1, 4, 1)], [(1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2), (1, 4, 2)]]"

# Two-stage sequence assignment
# 0: padding
# 1: fixed-length packing
# 3: dispatching & fixed-length packing
# 4: dispatching & fine-grained packing
BATCHING_METHOD=4
```

## 7. Time Cost and Scalability of Algorithms 

We isolated the algorithm part from the end-to-end training code to analyze its runtime and simulate its performance on a large-scale cluster (1024 GPUs). You can directly run the following script:

```bash
cd examples
bash scripts/test_planner.sh
```

## 8. Experiments in the Paper

You can reproduce all the results in our paper using:

```bash
cd examples
bash paper_exp/train_CommonCrawl.sh
bash paper_exp/train_GitHub.sh
```

Note that the settings of these scripts will need to be slightly modified according to the experiments you wish to reproduce.

