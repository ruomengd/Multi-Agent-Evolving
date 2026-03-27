#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_L1.sh"


################################################################################
# Edit these lists for your sweep.
BASE_MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-7B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

TOTAL_ROLE_TOKEN_BUDGETS=(
    "4096"
    "8192"
    "16384"
)

# planner / solver / revise / critique
ROLE_TOKEN_PLANS=(
    "0/0.4/0.3/0.3" # No-Planner
    "0.3/0.4/0/0.3" # No-Critique
    "0.3/0.4/0.3/0" # No-Revise
    "0.5/0.5/0/0" # No-CR
    "0/1.0/0/0" # Only-Solver
    "0.25/0.25/0.25/0.25" # Uniform
    "0.4/0.2/0.2/0.2" # Planner-Dominant
    "0.2/0.4/0.2/0.2" # Solver-Dominant
    "0.1/0.1/0.4/0.4" # CR-Dominant
)

BENCHMARKS=(
    "AIME24"
    "AIME25"
    "OlympiadBench"
)
################################################################################

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-mas_graph_sweep}"
FORCE_GENERATE="${FORCE_GENERATE:-false}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-/home/ruomeng/PettingLLMs/logs/mas_graph_sweeps}"
TIMESTAMP="$(date +%m-%d_%H-%M-%S)"
SWEEP_LOG_FILE="${SWEEP_LOG_DIR}/${EXPERIMENT_PREFIX}_${TIMESTAMP}.log"

mkdir -p "${SWEEP_LOG_DIR}"

slugify() {
    echo "$1" | tr '/: .<>' '_' | tr -s '_'
}

run_idx=0
total_runs=$((${#BASE_MODELS[@]} * ${#TOTAL_ROLE_TOKEN_BUDGETS[@]} * ${#ROLE_TOKEN_PLANS[@]} * ${#BENCHMARKS[@]}))

log_msg() {
    echo "$@" | tee -a "${SWEEP_LOG_FILE}"
}

log_msg "Sweep script: ${EVAL_SCRIPT}"
log_msg "Sweep log: ${SWEEP_LOG_FILE}"
log_msg "Total runs: ${total_runs}"
log_msg

for base_model in "${BASE_MODELS[@]}"; do
    for total_budget in "${TOTAL_ROLE_TOKEN_BUDGETS[@]}"; do
        for role_plan in "${ROLE_TOKEN_PLANS[@]}"; do
            for benchmark in "${BENCHMARKS[@]}"; do
                run_idx=$((run_idx + 1))

                model_slug="$(slugify "$base_model")"
                budget_slug="$(slugify "$total_budget")"
                plan_slug="$(slugify "$role_plan")"
                benchmark_slug="$(slugify "$benchmark")"
                experiment_name="${EXPERIMENT_PREFIX}_${benchmark_slug}_${model_slug}_b${budget_slug}_p${plan_slug}"

                log_msg "=================================================="
                log_msg "Run ${run_idx}/${total_runs}"
                log_msg "  BENCHMARK=${benchmark}"
                log_msg "  BASE_MODEL=${base_model}"
                log_msg "  TOTAL_ROLE_TOKEN_BUDGET=${total_budget}"
                log_msg "  ROLE_TOKEN_PLAN=${role_plan}"
                log_msg "  EXPERIMENT_NAME=${experiment_name}"
                log_msg "  FORCE_GENERATE=${FORCE_GENERATE}"
                log_msg "=================================================="

                (
                    cd /home/ruomeng/PettingLLMs
                    BASE_MODEL="${base_model}" \
                    TOTAL_ROLE_TOKEN_BUDGET="${total_budget}" \
                    ROLE_TOKEN_PLAN="${role_plan}" \
                    EXPERIMENT_NAME="${experiment_name}" \
                    FORCE_GENERATE="${FORCE_GENERATE}" \
                    BENCHMARK="${benchmark}" \
                    bash "${EVAL_SCRIPT}"
                ) 2>&1 | tee >(
                    tr '\r' '\n' | grep -E '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] rollouts:' >> "${SWEEP_LOG_FILE}" || true
                )

                log_msg
            done
        done
    done
done

log_msg "Sweep completed: ${total_runs} runs"
