#!/bin/bash
#SBATCH --job-name=mas_graph_sweep_aime24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/storage/home/bsunmiracle/repos/Multi-Agent-Evolving/logs/%x_%j.out
#SBATCH --error=/storage/home/bsunmiracle/repos/Multi-Agent-Evolving/logs/%x_%j.err
#SBATCH --account=mrs_2
#SBATCH --qos=h200_dev

set -euo pipefail

cd /storage/home/bsunmiracle/repos/Multi-Agent-Evolving

TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_planner.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_critique.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_revise.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_no_cr.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_only_solver.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_uniform.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_planner_dominant.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_solver_dominant.sh
TP_SIZE=1 BASE_MODELS_CSV="Qwen/Qwen3-1.7B" BENCHMARKS_CSV="AIME24" bash /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/scripts/evaluate/mas_graph/run_model_budget_plan_sweep_cr_dominant.sh
