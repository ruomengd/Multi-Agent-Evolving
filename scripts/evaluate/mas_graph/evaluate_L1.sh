#!/bin/bash
set -e

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ============================================
# Configuration - Edit these parameters
# Environment variable overrides are supported for sweep scripts.
# ============================================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
MODEL_PATHS=(
    "$BASE_MODEL"
    # "Qwen/Qwen3-4B"
)
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mas_graph_inference}"
# Assuming execution from repository root
REPO_ROOT="$(pwd)"
CONFIG_PATH="${REPO_ROOT}/pettingllms/config/mas_graph"
CONFIG_NAME="${CONFIG_NAME:-math_graph_multi_L1_prompt}"
BENCHMARK="${BENCHMARK:-AIME25}"
BASE_VLLM_PORT="${BASE_VLLM_PORT:-8601}"
BASE_PROXY_PORT="${BASE_PROXY_PORT:-8620}"
GPU_START_ID="${GPU_START_ID:-0}"
HOST="${HOST:-127.0.0.1}"
GPU_MEM="${GPU_MEM:-0.85}"  # Reduced from 0.8 to fit in available memory (33.38 GiB available)
VLLM_SHUTDOWN="${VLLM_SHUTDOWN:-true}"  # If true, vLLM will be shut down when script exits; if false, vLLM will remain running
TP_SIZE="${TP_SIZE:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
TOTAL_ROLE_TOKEN_BUDGET="${TOTAL_ROLE_TOKEN_BUDGET:-4096}"
VERIFIER_TOKEN_NUM="${VERIFIER_TOKEN_NUM:-128}"
ROLE_TOKEN_PLAN="${ROLE_TOKEN_PLAN:-0.25/0.25/0.25/0.25}"  # planner / solver / revise / critique
FORCE_GENERATE="${FORCE_GENERATE:-false}"
CONTINUATION_WAIT_TOKEN="${CONTINUATION_WAIT_TOKEN:-<wait>}"
MAX_CONTINUATION_ROUNDS="${MAX_CONTINUATION_ROUNDS:-8}"
MAX_LEN="${MAX_LEN:-32768}"
MAX_WAIT="${MAX_WAIT:-180}"  # Maximum wait time in seconds
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"  # Check interval in seconds

# Multi-GPU configuration
# If TP_SIZE > 1, each model will use TP_SIZE consecutive GPUs
# Example: TP_SIZE=2, GPU_START_ID=0 -> model0 uses GPU 0,1; model1 uses GPU 2,3
# ============================================

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    REQUIRED_GPUS=$((GPU_START_ID + ${#MODEL_PATHS[@]} * TP_SIZE))
    if [ $REQUIRED_GPUS -gt $AVAILABLE_GPUS ]; then
        echo "ERROR: Not enough GPUs available!"
        echo "  Required: ${REQUIRED_GPUS}, Available: ${AVAILABLE_GPUS}"
        exit 1
    fi
fi

declare -a VLLM_PIDS PROXY_PIDS
CLEANUP_DONE=0

cleanup() {
    if [ $CLEANUP_DONE -eq 1 ]; then
        exit 1
    fi
    CLEANUP_DONE=1

    # Always cleanup proxy processes
    for pid in "${PROXY_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done

    # Only cleanup vLLM if VLLM_SHUTDOWN is true
    if [ "$VLLM_SHUTDOWN" = true ]; then
        echo "Shutting down vLLM..."
        for pid in "${VLLM_PIDS[@]}"; do
            kill $pid 2>/dev/null || true
        done
        for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
            timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT INT TERM

# Function to wait for HTTP endpoint
wait_for_endpoint() {
    local host=$1
    local port=$2
    local name=$3
    local max_wait=$4

    echo -n "Waiting for $name at $host:$port"
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://$host:$port/v1/models" >/dev/null 2>&1; then
            echo " ✓ Ready (${elapsed}s)"
            return 0
        fi
        echo -n "."
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
    done
    echo " ✗ Timeout after ${max_wait}s"
    return 1
}

# Check whether a vLLM endpoint serves the expected model name
endpoint_has_model() {
    local host=$1
    local port=$2
    local expected_model=$3

    local models_json
    models_json=$(curl -s "http://$host:$port/v1/models" 2>/dev/null || true)

    if [ -z "$models_json" ]; then
        return 1
    fi

    if echo "$models_json" | grep -F "\"id\":\"$expected_model\"" >/dev/null 2>&1; then
        return 0
    fi

    if echo "$models_json" | grep -F "\"id\": \"$expected_model\"" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

# Kill existing proxy processes
echo "Cleaning existing proxy processes..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    timeout 2 lsof -ti:$((BASE_PROXY_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done
sleep 1

# Try to connect to existing vLLM first (5 seconds timeout)
echo "Checking for existing vLLM services..."
VLLM_ALREADY_RUNNING=false
VLLM_CHECK_TIMEOUT=5
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
        SERVED_MODEL_NAME="$MODEL_PATH"
    else
        SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
    fi

    echo -n "Trying to connect to vLLM at port $((BASE_VLLM_PORT + i))..."
    if timeout $VLLM_CHECK_TIMEOUT curl -s "http://$HOST:$((BASE_VLLM_PORT + i))/v1/models" >/dev/null 2>&1 && \
       endpoint_has_model "$HOST" "$((BASE_VLLM_PORT + i))" "$SERVED_MODEL_NAME"; then
        echo " ✓ Connected with expected model ${SERVED_MODEL_NAME}"
        VLLM_ALREADY_RUNNING=true
    else
        echo " ✗ Not available or missing expected model ${SERVED_MODEL_NAME}"
        VLLM_ALREADY_RUNNING=false
        break
    fi
done

# If vLLM not running or connection failed, clean up ports and prepare for new launch
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    echo "vLLM not detected, will start new instances..."
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        timeout 2 lsof -ti:$((BASE_VLLM_PORT + i)) 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    sleep 1
else
    echo "✓ Existing vLLM instances detected, will reuse them"
fi

# Launch vLLM services (only if not already running)
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    echo "Starting vLLM services..."
else
    echo "Skipping vLLM launch (already running)..."
fi
if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        MODEL_PATH="${MODEL_PATHS[$i]}"
        if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
            SERVED_MODEL_NAME="$MODEL_PATH"
        else
            SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
        fi

        # Calculate GPU IDs
        FIRST_GPU=$((GPU_START_ID + i * TP_SIZE))
        if [ $TP_SIZE -eq 1 ]; then
            GPU_IDS="$FIRST_GPU"
        else
            GPU_IDS="$FIRST_GPU"
            for ((g=1; g<$TP_SIZE; g++)); do
                GPU_IDS="$GPU_IDS,$((FIRST_GPU + g))"
            done
        fi

        echo "vLLM model$((i+1)): ${SERVED_MODEL_NAME} | GPU: $GPU_IDS | Port: $((BASE_VLLM_PORT + i))"
        CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_PATH}" \
            --served-model-name "${SERVED_MODEL_NAME}" \
            --host $HOST \
            --port $((BASE_VLLM_PORT + i)) \
            --gpu-memory-utilization $GPU_MEM \
            --tensor-parallel-size $TP_SIZE \
            --max-model-len $MAX_LEN > /tmp/vllm_model${i}.log 2>&1 &
        VLLM_PIDS[$i]=$!

        sleep 3
        if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
            echo "ERROR: vLLM died. Log: /tmp/vllm_model${i}.log"
            tail -n 30 /tmp/vllm_model${i}.log
            exit 1
        fi
    done
fi

if [ "$VLLM_ALREADY_RUNNING" = false ]; then
    echo "Waiting for vLLM services to initialize..."
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        # Check if process is still running
        if ! kill -0 ${VLLM_PIDS[$i]} 2>/dev/null; then
            echo "Error: model$((i+1)) vLLM process died"
            echo "====== Full vLLM log ======"
            cat /tmp/vllm_model${i}.log
            echo "=========================="
            exit 1
        fi

        # Wait for HTTP endpoint
        if ! wait_for_endpoint "$HOST" "$((BASE_VLLM_PORT + i))" "model$((i+1)) vLLM" "$MAX_WAIT"; then
            echo "Error: model$((i+1)) vLLM failed to start"
            echo "====== Last 50 lines of vLLM log ======"
            tail -n 50 /tmp/vllm_model${i}.log
            echo "======================================="
            echo ""
            echo "To view full log: cat /tmp/vllm_model${i}.log"
            exit 1
        fi
    done
else
    echo "Verifying existing vLLM services..."
    for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
        MODEL_PATH="${MODEL_PATHS[$i]}"
        if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
            SERVED_MODEL_NAME="$MODEL_PATH"
        else
            SERVED_MODEL_NAME="$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-2 | rev)"
        fi

        if ! curl -s "http://$HOST:$((BASE_VLLM_PORT + i))/v1/models" >/dev/null 2>&1; then
            echo "Error: Expected vLLM at port $((BASE_VLLM_PORT + i)) but it's not responding"
            echo "Please check if vLLM is running or set VLLM_SHUTDOWN=true to restart"
            exit 1
        fi

        if ! endpoint_has_model "$HOST" "$((BASE_VLLM_PORT + i))" "$SERVED_MODEL_NAME"; then
            echo "Error: Existing vLLM at port $((BASE_VLLM_PORT + i)) does not serve expected model ${SERVED_MODEL_NAME}"
            echo "Please restart the server or let this script launch a fresh instance"
            exit 1
        fi

        echo "✓ vLLM model$((i+1)) at port $((BASE_VLLM_PORT + i)) serves ${SERVED_MODEL_NAME}"
    done
fi

echo "✓ All vLLM services ready"
echo

# Launch proxy services
echo "Launching proxy services..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    VLLM_BACKEND_ADDRESS="${HOST}:$((BASE_VLLM_PORT + i))" \
    PROXY_PORT=$((BASE_PROXY_PORT + i)) \
    python pettingllms/evaluate/vllm_id_token_proxy.py > /tmp/proxy_model${i}.log 2>&1 &
    PROXY_PIDS[$i]=$!
done

# Wait for proxy ready
echo "Waiting for proxy services to initialize..."
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    if ! kill -0 ${PROXY_PIDS[$i]} 2>/dev/null; then
        echo "ERROR: Proxy model$((i+1)) died. Log: /tmp/proxy_model${i}.log"
        tail -n 20 /tmp/proxy_model${i}.log
        exit 1
    fi

    # Wait for HTTP endpoint
    if ! wait_for_endpoint "$HOST" "$((BASE_PROXY_PORT + i))" "model$((i+1)) proxy" "60"; then
        echo "Error: model$((i+1)) proxy failed to start"
        echo "Last 20 lines of log:"
        tail -n 20 /tmp/proxy_model${i}.log
        exit 1
    fi
done

echo "✓ All proxy services ready"
echo

# Derive per-role token budgets from one total budget and an allocation plan.
# Verifier is fixed, and the remaining budget is split across the other roles.
# Supported formats:
#   3 values: planner / solver+revise / critique
#   4 values: planner / solver / revise / critique
IFS='/' read -r -a ROLE_PLAN_PARTS <<< "$ROLE_TOKEN_PLAN"
if [ "${#ROLE_PLAN_PARTS[@]}" -eq 3 ]; then
    PLANNER_RATIO="${ROLE_PLAN_PARTS[0]}"
    SOLVER_RATIO="${ROLE_PLAN_PARTS[1]}"
    REVISE_RATIO="${ROLE_PLAN_PARTS[1]}"
    CRITIQUE_RATIO="${ROLE_PLAN_PARTS[2]}"
elif [ "${#ROLE_PLAN_PARTS[@]}" -eq 4 ]; then
    PLANNER_RATIO="${ROLE_PLAN_PARTS[0]}"
    SOLVER_RATIO="${ROLE_PLAN_PARTS[1]}"
    REVISE_RATIO="${ROLE_PLAN_PARTS[2]}"
    CRITIQUE_RATIO="${ROLE_PLAN_PARTS[3]}"
else
    echo "ERROR: ROLE_TOKEN_PLAN must have 3 or 4 '/'-separated values"
    echo "  3-value format: planner / solver+revise / critique"
    echo "  4-value format: planner / solver / revise / critique"
    exit 1
fi

REMAINING_ROLE_TOKEN_BUDGET=$((TOTAL_ROLE_TOKEN_BUDGET - VERIFIER_TOKEN_NUM))
if [ "$REMAINING_ROLE_TOKEN_BUDGET" -le 0 ]; then
    echo "ERROR: TOTAL_ROLE_TOKEN_BUDGET must be greater than VERIFIER_TOKEN_NUM"
    exit 1
fi

PLAN_SUM=$(awk -v a="$PLANNER_RATIO" -v b="$SOLVER_RATIO" -v c="$REVISE_RATIO" -v d="$CRITIQUE_RATIO" 'BEGIN { printf "%.8f", a + b + c + d }')
if ! awk -v sum="$PLAN_SUM" 'BEGIN { exit !(sum > 0) }'; then
    echo "ERROR: ROLE_TOKEN_PLAN must sum to a positive value"
    exit 1
fi

calc_role_tokens() {
    local ratio=$1
    awk -v total="$REMAINING_ROLE_TOKEN_BUDGET" -v ratio="$ratio" -v sum="$PLAN_SUM" 'BEGIN {
        tokens = int((total * ratio / sum) + 0.5)
        if (tokens < 1) tokens = 1
        print tokens
    }'
}

PLANNER_TOKEN_NUM=$(calc_role_tokens "$PLANNER_RATIO")
SOLVER_TOKEN_NUM=$(calc_role_tokens "$SOLVER_RATIO")
CRITIQUE_TOKEN_NUM=$(calc_role_tokens "$CRITIQUE_RATIO")
REVISE_TOKEN_NUM=$(calc_role_tokens "$REVISE_RATIO")
MAX_RESPONSE_LENGTH=$(printf "%s\n%s\n%s\n%s\n%s\n" \
    "$PLANNER_TOKEN_NUM" \
    "$SOLVER_TOKEN_NUM" \
    "$CRITIQUE_TOKEN_NUM" \
    "$REVISE_TOKEN_NUM" \
    "$VERIFIER_TOKEN_NUM" | sort -nr | head -n 1)

echo "Role token budget: total=${TOTAL_ROLE_TOKEN_BUDGET}, verifier_fixed=${VERIFIER_TOKEN_NUM}, remaining=${REMAINING_ROLE_TOKEN_BUDGET}, plan=${ROLE_TOKEN_PLAN}"
echo "  planner=${PLANNER_TOKEN_NUM}, solver=${SOLVER_TOKEN_NUM}, critique=${CRITIQUE_TOKEN_NUM}, revise=${REVISE_TOKEN_NUM}, verifier=${VERIFIER_TOKEN_NUM}"
echo "Derived MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "Force generate: enabled=${FORCE_GENERATE}, wait_token=${CONTINUATION_WAIT_TOKEN}, max_rounds=${MAX_CONTINUATION_ROUNDS}"
CONTINUATION_WAIT_TOKEN_OVERRIDE="'${CONTINUATION_WAIT_TOKEN}'"

# Build model args for both base_models and models
MODEL_ARGS=""
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    MODEL_ARGS="$MODEL_ARGS models.model_${i}.path=\"${MODEL_PATHS[$i]}\""
    MODEL_ARGS="$MODEL_ARGS base_models.policy_${i}.path=\"${MODEL_PATHS[$i]}\""
done

# Run evaluation
echo "Running evaluation..."
VLLM_ADDRESS="${HOST}:${BASE_PROXY_PORT}"

python3 -m pettingllms.evaluate.evaluate \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    +parallel=true \
    +vllm_address="$VLLM_ADDRESS" \
    env.benchmark="$BENCHMARK" \
    $MODEL_ARGS \
    training.max_prompt_length=$MAX_PROMPT_LENGTH \
    training.max_response_length=$MAX_RESPONSE_LENGTH \
    agent_policy_configs.agent_configs.agent_0.train_llm_config.token_num=$PLANNER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_0.val_llm_config.token_num=$PLANNER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_0.train_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    agent_policy_configs.agent_configs.agent_0.val_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    "agent_policy_configs.agent_configs.agent_0.train_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    "agent_policy_configs.agent_configs.agent_0.val_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    agent_policy_configs.agent_configs.agent_0.train_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_0.val_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_1.train_llm_config.token_num=$SOLVER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_1.val_llm_config.token_num=$SOLVER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_1.train_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    agent_policy_configs.agent_configs.agent_1.val_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    "agent_policy_configs.agent_configs.agent_1.train_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    "agent_policy_configs.agent_configs.agent_1.val_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    agent_policy_configs.agent_configs.agent_1.train_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_1.val_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_2.train_llm_config.token_num=$CRITIQUE_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_2.val_llm_config.token_num=$CRITIQUE_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_2.train_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    agent_policy_configs.agent_configs.agent_2.val_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    "agent_policy_configs.agent_configs.agent_2.train_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    "agent_policy_configs.agent_configs.agent_2.val_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    agent_policy_configs.agent_configs.agent_2.train_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_2.val_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_3.train_llm_config.token_num=$REVISE_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_3.val_llm_config.token_num=$REVISE_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_3.train_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    agent_policy_configs.agent_configs.agent_3.val_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    "agent_policy_configs.agent_configs.agent_3.train_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    "agent_policy_configs.agent_configs.agent_3.val_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    agent_policy_configs.agent_configs.agent_3.train_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_3.val_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_4.train_llm_config.token_num=$VERIFIER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_4.val_llm_config.token_num=$VERIFIER_TOKEN_NUM \
    agent_policy_configs.agent_configs.agent_4.train_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    agent_policy_configs.agent_configs.agent_4.val_llm_config.continue_until_max_tokens=$FORCE_GENERATE \
    "agent_policy_configs.agent_configs.agent_4.train_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    "agent_policy_configs.agent_configs.agent_4.val_llm_config.continuation_wait_token=${CONTINUATION_WAIT_TOKEN_OVERRIDE}" \
    agent_policy_configs.agent_configs.agent_4.train_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    agent_policy_configs.agent_configs.agent_4.val_llm_config.max_continuation_rounds=$MAX_CONTINUATION_ROUNDS \
    training.experiment_name="$EXPERIMENT_NAME" \
    resource.n_gpus_per_node=$TP_SIZE \
    resource.nnodes=1 \
    models.model_0.ppo_trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_gpus_per_node=$TP_SIZE \
    models.model_0.ppo_trainer_config.actor_rollout_ref.trainer.n_training_gpus_per_node=$TP_SIZE

echo "Evaluation completed"
