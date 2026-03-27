
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

### 4) MAS Graph Evaluation Sweep

For the math graph workflow, use:

```bash
bash scripts/evaluate/mas_graph/evaluate_L1.sh
```

Key knobs in `scripts/evaluate/mas_graph/evaluate_L1.sh`:

- `BASE_MODEL`: base model path, for example `Qwen/Qwen3-7B`
- `TOTAL_ROLE_TOKEN_BUDGET`: total generation budget shared across roles
- `VERIFIER_TOKEN_NUM`: fixed verifier budget
- `ROLE_TOKEN_PLAN`: planner / solver / revise / critique allocation plan
- `FORCE_GENERATE`: whether to keep appending `<wait>` and continue generation toward the role token cap

For sweep experiments across models, budgets, and benchmarks with a single fixed role plan, use:

```bash
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_single_plan.sh
```

This script sweeps:

- `BASE_MODELS`
- `TOTAL_ROLE_TOKEN_BUDGETS`
- `BENCHMARKS`

and expects these environment variables to be set by the caller:

- `ROLE_TOKEN_PLAN`
- `PLAN_NAME`

Example:

```bash
ROLE_TOKEN_PLAN="0.25/0.25/0.25/0.25" \
PLAN_NAME="uniform" \
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_single_plan.sh
```

For convenience, there are also one-plan wrapper scripts:

- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_planner.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_critique.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_revise.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_cr.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_only_solver.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_uniform.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_planner_dominant.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_solver_dominant.sh`
- `scripts/evaluate/mas_graph/run_model_budget_plan_sweep_cr_dominant.sh`

```bash
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_planner.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_critique.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_revise.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_cr.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_only_solver.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_uniform.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_planner_dominant.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_solver_dominant.sh
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_cr_dominant.sh
```


Demo commands to quickly test each wrapper with `Qwen/Qwen3-1.7B` on `AIME24`:

```bash
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_planner.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_critique.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_revise.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_cr.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_only_solver.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_uniform.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_planner_dominant.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_solver_dominant.sh
BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_cr_dominant.sh
```

Each wrapper fixes one `ROLE_TOKEN_PLAN` and still sweeps:

- model size
- total role token budget
- benchmark

Example:

```bash
bash scripts/evaluate/mas_graph/run_model_budget_plan_sweep_uniform.sh
```

For example, sweep logs are written to:

```bash
logs/mas_graph_sweeps/
logs/mas_graph_sweep_AIME24_Qwen_Qwen3-1_7B_b8192_p0_0_1_0_0_0_0_0
```

The saved sweep log keeps:

- run header metadata
- rollout progress bar lines
