set -x

# Use GPUs 4,5,6,7 (last 4 GPUs)
export CUDA_VISIBLE_DEVICES=4,5,6,7
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
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}


# GPU_num should be 4 since we're using 4 GPUs (after CUDA_VISIBLE_DEVICES, they appear as 0,1,2,3)
GPU_num=4

# Verify CUDA_VISIBLE_DEVICES is set correctly
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU_num: $GPU_num"

# Since we have 3 models and only 4 GPUs, each model uses 1 GPU
# tensor_model_parallel_size should be 1 for each model
model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="resource.n_gpus_per_node=$GPU_num $model_0_config_path.trainer.n_gpus_per_node=1 $model_0_config_path.trainer.nnodes=1 $model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=1 $model_1_config_path.trainer.nnodes=1 $model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"

model_2_config_path="models.model_2.ppo_trainer_config"
model_2_resource="$model_2_config_path.trainer.n_gpus_per_node=1 $model_2_config_path.trainer.nnodes=1 $model_2_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=1"


python3 -m pettingllms.trainer.train --config-path ../config/code --config-name code_L3_model \
    $model_0_resource \
    $model_1_resource \
    $model_2_resource \
    base_models.policy_0.path="Qwen/Qwen3-0.6B"\
    base_models.policy_1.path="Qwen/Qwen3-0.6B"\
    base_models.policy_2.path="Qwen/Qwen3-0.6B"\
    training.experiment_name=code_multi_model\
    training.total_training_steps=200\
    training.train_batch_size=32\
    training.train_sample_num=8\
    training.validate_sample_num=1\
    training.max_prompt_length=4096\
    training.max_response_length=2048\
    training.val_freq=10\
    training.resample_freq=3\
    env.dataset=code_contests\
    env.benchmark=code_contests\
