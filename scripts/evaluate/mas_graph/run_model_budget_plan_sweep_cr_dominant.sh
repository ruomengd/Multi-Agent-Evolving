#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE_TOKEN_PLAN="0.1/0.1/0.4/0.4"
PLAN_NAME="cr_dominant"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-mas_graph_sweep_cr_dominant}"
ROLE_TOKEN_PLAN="${ROLE_TOKEN_PLAN}" PLAN_NAME="${PLAN_NAME}" EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX}" bash "${SCRIPT_DIR}/run_model_budget_plan_sweep_single_plan.sh"
