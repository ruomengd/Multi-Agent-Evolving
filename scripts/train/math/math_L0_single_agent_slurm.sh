#!/bin/bash
#SBATCH --job-name=math_l0_single_agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/bsunmiracle/repos/Multi-Agent-Evolving/logs/%x_%j.out
#SBATCH --error=/storage/home/bsunmiracle/repos/Multi-Agent-Evolving/logs/%x_%j.err
#SBATCH --account=mrs_2
#SBATCH --qos=h200_dev


cd /storage/home/bsunmiracle/repos/Multi-Agent-Evolving
conda init
conda activate mas

mkdir -p /storage/home/bsunmiracle/repos/Multi-Agent-Evolving/logs
mkdir -p /tmp/r

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
export WANDB_MODE=offline

export RAY_TMPDIR=/tmp/r
export TMPDIR=/tmp/r

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

ray stop --force || true

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_num=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  GPU_num=${SLURM_GPUS_ON_NODE:-1}
fi

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_num"

python -m pettingllms.trainer.train \
  --config-path ../config/math \
  --config-name math_L0_single_agent \
  $model_0_resource \
  base_models.policy_0.path="Qwen/Qwen3-1.7B" \
  training.experiment_name=math_single_agent \
  training.total_training_steps=10 \
  training.train_batch_size=32 \
  training.train_sample_num=8 \
  training.validate_sample_num=1 \
  training.max_prompt_length=2048 \
  training.max_response_length=4096 \
  training.val_freq=10 \
  env.dataset=polaris \
  env.benchmark=AIME24
