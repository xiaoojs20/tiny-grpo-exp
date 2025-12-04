# TRW-GRPO: Trust-Region Weighted Group Relative Policy Optimization

> A trust-region weighting view of GRPO for stable RLHF / reasoning RL with LLMs  
> 基于“信任域加权视角”的 GRPO 改造，用于更稳定的大模型 RLHF / 推理强化学习

---

## 🔍 Overview / 项目简介

This repository implements **Trust-Region Weighted GRPO (TRW-GRPO)**, a family of GRPO-style algorithms where
the **gradient is explicitly reweighted by a trust-region weight** `w(r, Â)` instead of relying on hard PPO-like clipping.

- We **re-interpret GRPO** as a policy-gradient method with an *implicit* importance-ratio–dependent weight.
- We show how **PPO clipping, DAPO Clip-Higher, TRPA, MRPO** can all be seen as special cases of **trust-region weighting**.
- We provide **drop-in implementations** of:
  - vanilla GRPO,
  - DAPO-style Clip-Higher GRPO, and
  - several TRW-GRPO variants: triangular, logistic, entropy-aware.

> 本仓库实现了 **TRW-GRPO（Trust-Region Weighted GRPO）**，将 GRPO 视为“带信任域权重的策略梯度方法”，
> 用显式的 `w(r, Â)` 取代传统的硬剪切（piecewise 0/1 clipping），从而在稳定性、探索性和样本利用率之间取得更好的平衡。

当前状态：  
- ✅ 理论 & 方法（NeurIPS 风格论文草稿）  
- ✅ PyTorch 参考实现（可插入现有 GRPO / RLHF pipeline）  
- 🚧 实验脚本 & 完整 benchmark 将在后续补充  

---

## ✨ Key Ideas / 核心思想

### Gradient-weighting view of GRPO

Standard GRPO gradient can be written as:

\[
\nabla_\theta L_{\text{GRPO}}
= \mathbb{E}_{x,y \sim \pi_{\text{old}}}
\big[ w_{\text{GRPO}}(r, \hat{A})\, r\, \hat{A}\, \nabla_\theta \log \pi_\theta(y \mid x) \big]
\]

where `r = πθ / πold` and `Â` is group-normalized advantage.  
In vanilla GRPO, `w_GRPO` is an **implicit piecewise 0/1 function** induced by clipping.

> 标准 GRPO 的梯度可以重写为“带权重的策略梯度”，其中 `w_GRPO(r, Â)` 本质上是由 clip 产生的
> **分段 0/1 权重函数**，在 `1±ε` 之外直接把梯度截断。

### Why DAPO uses Clip-Higher?

DAPO observes that symmetric clipping `[1−ε, 1+ε]` causes **entropy collapse**:
- high-probability tokens can still increase a lot,
- low-probability “exploration” tokens are over-constrained → exploration dies.

So DAPO **loosens the upper clip** (`ε_high > ε_low`), effectively giving larger weight to promising low-probability tokens.

> DAPO 的 Clip-Higher 实际上是在做“不对称信任域”：  
> - 下界更严格（防止好 token 被过度削弱）  
> - 上界更宽松（让稀有但高优势的 token 有机会被放大），从而缓解熵坍缩。

### TRW-GRPO: make the weight explicit

TRW-GRPO keeps the same GRPO structure but **replaces hard clipping** with an **explicit trust-region weight**:

\[
L_{\text{TRW}}(\theta) =
\mathbb{E}[\, w(r, \hat{A})\, r\, \hat{A} \,]
\]

We provide several designs of `w(r, Â)`:

- **Triangular / piecewise-linear** trust region  
  - `r ∈ [1−ε_low, 1]`: weight from `w0 → 1` (线性上升)  
  - `r ∈ [1, 1+ε_high]`: keep `1` or decay to `w2` (平缓下降)  
- **Smooth logistic** weight  
  - `w(r, Â) = σ(-α |log r|) · σ(γ Â) + η`  
- **Entropy-aware asymmetric** weight  
  - Extra boost for **rare but high-advantage** actions.

> 简单来说：  
> - 原来的 GRPO 是“信任域 = 硬边界 + 0/1 权重”  
> - TRW-GRPO 变成“信任域 = 连续 / 可调的权重函数 w(r, Â)”  
> - 更平滑、更可控、更方便结合 DAPO / TRPA / MRPO 这些工作。

---

## 🧱 Repository Structure / 代码结构



Installation

```
conda create -n tinygrpo python=3.10
conda activate tinygrpo
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
# verl
pip install -e .
# conda install -c nvidia cuda-nvcc=12.1
MAX_JOBS=4 pip3 install flash-attn --no-build-isolation
pip install wandb IPython matplotlib
```

test verl
data process
```
python3 examples/data_preprocess/gsm8k.py \
--local_dataset_path /mnt/sdb1/sdb1_xiaojinsong/datasets/openai/gsm8k \
--local_save_dir /mnt/sdb1/sdb1_xiaojinsong/datasets/openai/gsm8k
```

train
```
local_dataset_path=/mnt/sdb1/sdb1_xiaojinsong/datasets
local_model_path=/mnt/sdb1/sdb1_xiaojinsong/llms
HF_USE_FLASH_ATTENTION_2=0
FLASH_ATTENTION_SKIP=1


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$local_dataset_path/openai/gsm8k/train.parquet \
    data.val_files=$local_dataset_path/openai/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=128 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$local_model_path/Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=20 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True # new \
    actor_rollout_ref.model.attn_implementation=sdpa \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_0_5b_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@
```