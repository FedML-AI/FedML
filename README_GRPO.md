# GRPO Setup Guide

This guide provides setup instructions for running GRPO (Group Relative Policy Optimization) experiments using FedML.

## Initial Setup

**Note:** Server and Client(s) use the same initial setup process.

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/bagel-org/FedML.git
cd FedML/python/spotlight_prj/fedllm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install "fedml>=0.8.4a7" "trl>=0.9.0" "accelerate>=0.27.0" 
```

### 3. Environment Configuration

Set up AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=<your_key>
export AWS_SECRET_ACCESS_KEY=<your_other_key>
```

Generate a unique run ID:

```bash
export RUN_ID=$(python -c "import uuid; print(uuid.uuid4().hex)") 
```

**Important:** Server and Client(s) should all use the same run ID for a given run to avoid data conflicts in the S3 bucket.

### 4. Weights & Biases Setup

Configure wandb for experiment logging:

```bash
wandb login
```

## Running Experiments

### 1-Client Test

#### Server

```bash
bash scripts/run_fedml_server_custom.sh 0 "$RUN_ID" localhost 29500 1 auto fedml_config/grpo_gsm8k_test_config.yaml
```

#### Client

```bash
bash scripts/run_fedml_client_custom.sh 1 "$RUN_ID" localhost 29500 1 auto fedml_config/grpo_gsm8k_test_config.yaml
```

## Notes

- The `RUN_ID` should be unique for each experimental run to prevent data conflicts across different experiments
- All participants (server and clients) must use the same `RUN_ID` for a given experimental run
- Make sure AWS credentials are properly configured before starting the experiments 