#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE_TOKEN_PLAN="0.3/0.4/0/0.3"
PLAN_NAME="no_critique"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-mas_graph_sweep_no_critique}"
ROLE_TOKEN_PLAN="${ROLE_TOKEN_PLAN}" PLAN_NAME="${PLAN_NAME}" EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}" bash "${SCRIPT_DIR}/run_model_budget_plan_sweep_single_plan.sh"
