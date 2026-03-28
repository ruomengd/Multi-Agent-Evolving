#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_L1.sh"

BASE_MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
)

TOTAL_ROLE_TOKEN_BUDGETS=(
    "4096"
    "8192"
    "16384"
)

BENCHMARKS=(
    "AIME24"
    "AIME25"
    "OlympiadBench"
)

if [ -n "${BASE_MODELS_CSV:-}" ]; then
    IFS=',' read -r -a BASE_MODELS <<< "${BASE_MODELS_CSV}"
fi

if [ -n "${TOTAL_ROLE_TOKEN_BUDGETS_CSV:-}" ]; then
    IFS=',' read -r -a TOTAL_ROLE_TOKEN_BUDGETS <<< "${TOTAL_ROLE_TOKEN_BUDGETS_CSV}"
fi

if [ -n "${BENCHMARKS_CSV:-}" ]; then
    IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARKS_CSV}"
fi

ROLE_TOKEN_PLAN="${ROLE_TOKEN_PLAN:?ROLE_TOKEN_PLAN must be set}"
PLAN_NAME="${PLAN_NAME:?PLAN_NAME must be set}"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-mas_graph_sweep_${PLAN_NAME}}"
FORCE_GENERATE="${FORCE_GENERATE:-false}"
TP_SIZE="${TP_SIZE:-1}"
SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${SCRIPT_DIR}/../../../logs/mas_graph_sweeps}"
TIMESTAMP="$(date +%m-%d_%H-%M-%S)"
SWEEP_LOG_FILE="${SWEEP_LOG_DIR}/${EXPERIMENT_PREFIX}_${TIMESTAMP}.log"

mkdir -p "${SWEEP_LOG_DIR}"

slugify() {
    echo "$1" | tr '/: .<>' '_' | tr -s '_'
}

run_idx=0
total_runs=$((${#BASE_MODELS[@]} * ${#TOTAL_ROLE_TOKEN_BUDGETS[@]} * ${#BENCHMARKS[@]}))

log_msg() {
    echo "$@" | tee -a "${SWEEP_LOG_FILE}"
}

log_msg "Sweep script: ${EVAL_SCRIPT}"
log_msg "Sweep log: ${SWEEP_LOG_FILE}"
log_msg "Plan name: ${PLAN_NAME}"
log_msg "ROLE_TOKEN_PLAN=${ROLE_TOKEN_PLAN}"
log_msg "TP_SIZE=${TP_SIZE}"
log_msg "Total runs: ${total_runs}"
log_msg

for base_model in "${BASE_MODELS[@]}"; do
    for total_budget in "${TOTAL_ROLE_TOKEN_BUDGETS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do
            run_idx=$((run_idx + 1))

            model_slug="$(slugify "$base_model")"
            budget_slug="$(slugify "$total_budget")"
            plan_slug="$(slugify "$ROLE_TOKEN_PLAN")"
            benchmark_slug="$(slugify "$benchmark")"
            experiment_name="${EXPERIMENT_PREFIX}_${benchmark_slug}_${model_slug}_b${budget_slug}_p${plan_slug}"

            log_msg "=================================================="
            log_msg "Run ${run_idx}/${total_runs}"
            log_msg "  BENCHMARK=${benchmark}"
            log_msg "  BASE_MODEL=${base_model}"
            log_msg "  TOTAL_ROLE_TOKEN_BUDGET=${total_budget}"
            log_msg "  ROLE_TOKEN_PLAN=${ROLE_TOKEN_PLAN}"
            log_msg "  PLAN_NAME=${PLAN_NAME}"
            log_msg "  EXPERIMENT_NAME=${experiment_name}"
            log_msg "  FORCE_GENERATE=${FORCE_GENERATE}"
            log_msg "  TP_SIZE=${TP_SIZE}"
            log_msg "=================================================="

            (
                BASE_MODEL="${base_model}" \
                TOTAL_ROLE_TOKEN_BUDGET="${total_budget}" \
                ROLE_TOKEN_PLAN="${ROLE_TOKEN_PLAN}" \
                EXPERIMENT_NAME="${experiment_name}" \
                FORCE_GENERATE="${FORCE_GENERATE}" \
                TP_SIZE="${TP_SIZE}" \
                BENCHMARK="${benchmark}" \
                bash "${EVAL_SCRIPT}"
            ) 2>&1 | tee >(
                tr '\r' '\n' | grep -E '^\[[0-9]{2}:[0-9]{2}:[0-9]{2}\] rollouts:' >> "${SWEEP_LOG_FILE}" || true
            )

            log_msg
        done
    done
done

log_msg "Sweep completed: ${total_runs} runs"
