#!/bin/bash
export RAY_DISABLE_DASHBOARD=1
export RAY_START_ARGS="--include-dashboard=false"
export SWANLAB_API_KEY="gYbI1egFijpdmf4K0uYoS"
export WANDB_API_KEY="a552f7169e3e6889c5c9bb37306e265b0168ae02"
export CUDA_VISIBLE_DEVICES="6,7"
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AIME-TTT"
BACKBONE="Qwen2.5-Math-1.5B"
ADVANTAGE="grpo"
PROCESS_REWARD="true"

K=3
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16
fi

EPISODE=80
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=64
N_SAMPLES_PER_PROMPT=32
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=1
PROCESS_REWARD_WEIGHT=0.2
PROCESS_REWARD_STRATEGY="avg"
PROCESS_LCS_NORM="avg"
PROCESS_LCS_MAX_TOKENS=1000
USE_CONTRASTIVE_PROCESS_REWARD=False
CONTRASTIVE_TEMPERATURE=0.1

DATA_LOCAL_DIR="/caobing/biomedical/TTRL/verl/data"
BACKBONE_PATH="/caobing/biomedical/llm_checkpoint/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-Len@${K}k-process_reward_strategy:${PROCESS_REWARD_STRATEGY}-reward_weight:${PROCESS_REWARD_WEIGHT}-lcs_norm:${PROCESS_LCS_NORM}-lcs_max_tokens:${PROCESS_LCS_MAX_TOKENS}-use_contrastive:${USE_CONTRASTIVE_PROCESS_REWARD}-contrastive_temperature:${CONTRASTIVE_TEMPERATURE}"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}-${PROCESS_REWARD}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${PROCESS_REWARD}-${TIME_TAG}"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
--config-name='ppo_trainer_ttrl.yaml'\
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func_batch \
  reward_model.reward_manager=batch \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  ttrl.process_reward_weight=$PROCESS_REWARD_WEIGHT \
  ttrl.process_reward_strategy=$PROCESS_REWARD_STRATEGY \
  ttrl.process_lcs_norm=$PROCESS_LCS_NORM \
  ttrl.process_lcs_max_tokens=$PROCESS_LCS_MAX_TOKENS \
  ttrl.use_contrastive_process_reward=$USE_CONTRASTIVE_PROCESS_REWARD \
  ttrl.contrastive_temperature=$CONTRASTIVE_TEMPERATURE \
  trainer.logger=['console','swanlab'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=2000000 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"