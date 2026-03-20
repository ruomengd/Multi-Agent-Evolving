
## Installation

```bash
bash setup.bash
```

## Quick Start

### 1) Dataset preparation

```bash
# Code tasks (APPS, CodeContests, LiveCodeBench)
python scripts/dataprocess/load_code.py

# Math tasks (AIME24/25, OlympiadBench)
python scripts/dataprocess/load_math.py

```

Datasets are saved to `datasets/code/`, and `datasets/math/`.

### 2) Training

Example: train a multi-agent system on math tasks.

```bash
bash scripts/train/math/math_L1_prompt.sh
```


### 3) Evaluation

Edit `scripts/evaluate/evaluate.sh` to set your model path and config:

```bash
MODEL_PATHS=("/path/to/your/model")
CONFIG_NAME="math_single_policy"
```

Then run:

```bash
bash scripts/evaluate/evaluate.sh
```
