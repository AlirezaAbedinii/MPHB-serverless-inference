# Model Partitioner

Automatic DNN model partitioning for serverless inference. Extends [Gillis](https://github.com/MincYu/gillis-open-source) (ICDCS '21) with **cost-aware, ROI-based partition template selection**: instead of minimizing latency alone, the latency-optimal (DP) policy scores candidate partitions by speedup per unit of extra cost relative to a monolithic-CPU baseline, so the selected template balances latency feasibility against per-request cost.

Supports ONNX models with the MXNet runtime.

## Structure

```
partition/
  main.py            CLI entry point
  layer_graph.py     ONNX graph representation + per-op latency predictors
  parallel_util.py   Execution plans, attribute/parameter partitioning, ROI baseline registry
  partition.py       ONNX export of the selected partitions
  policy_dp.py       Latency-optimal DP algorithm (ROI-based selection)
  policy_rl.py       SLO-aware RL algorithm
  policy_bf.py       Brute-force search
  policy_bayesian.py Bayesian optimization
  layer_runtime.cfg  Measured per-op latency coefficients per backend
aws_lambda_deploy/   Generates and deploys one AWS Lambda function per partition (SAM)
tool/                MXNet Wide-ResNet builder used by the exporters
*_exporter.py        Scripts to export common architectures to ONNX
```

## Partition a model

```bash
cd partition
# place your ONNX model in models/
python main.py lo -n <model.onnx> -p true                 # latency-optimal (DP, cost-aware)
python main.py sa -n <model.onnx> -t <slo_ms> -p true -d <rl_model_dir>   # SLO-aware (RL)
python main.py bf -n <model.onnx> -t <slo_ms> -p true     # brute-force
python main.py bayesian -n <model.onnx> -t <slo_ms>       # Bayesian search
```

`-p true` exports the actual partitions: a `<model>_workspace/` directory is written containing per-partition ONNX and MXNet (`.json` + `.params`) files plus `input/structure.json` with the stage boundaries.

`structure.json` is the handoff artifact to the provisioning framework: its stage boundaries and profiled per-stage coefficients populate `../provisioning/conf/config2.json`.

> The partitioner targets the MXNet 1.5 runtime (see `requirements.txt`); use Python 3.7 for full compatibility with `mxnet==1.5.0`.

## Deploy partitions on AWS Lambda

`aws_lambda_deploy/` turns a partition workspace into a deployable AWS SAM application: `deploy.py` reads `structure.json`, generates one Lambda function per partition from the handler templates in `templates/`, and assembles a SAM `template.yaml`.

```bash
cd aws_lambda_deploy
mv ../<model>_workspace .          # bring in the exported workspace
bash deploy.sh -j <model>_workspace
# then follow the AWS SAM prompts: sam build && sam deploy --guided
```

Prerequisites: AWS CLI + SAM CLI configured with your credentials (see `aws_lambda_deploy/README.md`). The generated functions attach a `libgomp` Lambda layer — build/publish it in your account and set its ARN in `templates/template_yaml_prefix` (placeholder `<YOUR_ACCOUNT_ID>`).

After deployment, SAM prints an inference API endpoint:

```bash
curl [API]
```
