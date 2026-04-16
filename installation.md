# Installation & Training Guide for verl

This guide provides a step-by-step walkthrough to set up your environment and start training with `verl`.

## 1. Prerequisites
- **Python**: 3.10 or higher (3.12 recommended).
- **CUDA**: 12.8 or higher.
- **Hardware**: NVIDIA GPUs (e.g., H100, A100). FSDP is recommended for prototyping.

## 2. Environment Setup
We recommend using `uv` for Python environment management as it is fast and reliable.

```bash
# Create and activate a virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install core development tools
uv pip install pre-commit hydra-core
```

## 3. verl Installation
Install `verl` in editable mode with the `vllm` extra for inference support.

```bash
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e ".[vllm]"
```

> [!TIP]
> If you plan to use **Megatron-LM** for large-scale training, you should also install the additional dependencies:
> `bash scripts/install_vllm_sglang_mcore.sh`

## 4. Data Preprocessing
Prepare your dataset using the provided preprocessing scripts. For example, to prepare the **GSM8K** dataset:

```bash
# Run the preprocessing script
python3 examples/data_preprocess/gsm8k.py
```
This will generate `.parquet` files in `data/gsm8k/` required for the training trainer.

## 5. Launch Training
You can start training using the PPO trainer. A sample script `sample_training.sh` is provided in the root directory for a 2-GPU setup.

```bash
# Enable VLLM V1 engine for optimal performance
export VLLM_USE_V1=1

# Run the training sample
bash sample_training.sh
```

### Key Parameters in `sample_training.sh`
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **GPUs**: Configured for 2 GPUs by default (`trainer.n_gpus_per_node=2`).
- **Algorithm**: PPO.

## 6. Verification
- **Monitor Logs**: Logs are piped to `verl_demo.log`. Use `tail -f verl_demo.log` to watch progress.
- **GPU Activity**: Use `nvidia-smi` to verify GPU utilization and memory consumption.
- **Checkpoints**: By default, checkpoints are saved according to the `trainer.save_freq` parameter.

---

For more detailed information on specific backends (Megatron-LM, SGLang) or advanced algorithms, please refer to the [docs/](docs/) directory.
