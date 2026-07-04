import policy_dp
import policy_rl
import policy_bayesian
import policy_bf
import partition
import click
from utils import *
from parallel_util import *
import os
import shutil
from pathlib import Path
# from onnx.external_data_helper import uses_external_data, ExternalDataInfo


all_benchmarks = {
    'vgg11.onnx': [500, 600],
}

algos = {
    'lo': policy_dp.test_model,
    'sa': policy_rl.test_benchmark,
    'bf': policy_bf.test_benchmark,
    'bayesian': policy_bayesian.test_benchmark
}

def export_model_partitions(name, plan, workspace_name=None):
    work_path =  workspace_name + '_workspace/input' if workspace_name else name.split('.')[0] + '_workspace/input'
    work_dir = f'{partition_results_dir}/{work_path}'
    if create_dir(work_dir):
        os.makedirs('/mnt/data/gillis-open-source/net0_workspace/input/models', exist_ok=True)  # Added by A: create parents safely
        
        
        shutil.copy2('/mnt/data/gillis-open-source/partition/models/net0.onnx',
                     '/mnt/data/gillis-open-source/net0_workspace/input/models/net0.onnx')         # Added by A
        shutil.copy2('/mnt/data/gillis-open-source/partition/models/net0.onnx.data',
                     '/mnt/data/gillis-open-source/net0_workspace/input/models/net0.onnx.data')  # Added by A
        partition.model_partition(name, plan, work_dir)
        plan_json_path = f'{work_dir}/structure.json'
        plan.to_json(plan_json_path)
# # def export_model_partitions(name, plan, workspace_name=None):
#     # ORIGINAL logic, but use Path to avoid path bugs
#     # work_path =  workspace_name + '_workspace/input' if workspace_name else name.split('.')[0] + '_workspace/input'
#     # work_dir = f'{partition_results_dir}/{work_path}'
#     work_path = f"{workspace_name}_workspace/input" if workspace_name else f"{Path(name).stem}_workspace/input"  # keep your semantics
#     work_dir = Path(partition_results_dir) / work_path

#     # NEW: create <workspace>/input/models with parents (idempotent)
#     models_dir = work_dir / "models"
#     models_dir.mkdir(parents=True, exist_ok=True)

#     # Stage the original .onnx into the workspace (safer if downstream code expects it there)
#     src_onnx = Path(name).resolve()
#     dst_onnx = models_dir / src_onnx.name
#     # (If the upstream code already copies .onnx, this overwrites with same content; harmless)
#     shutil.copy2(src_onnx, dst_onnx)

#     # Determine the sidecar filename embedded in the ONNX (don’t assume “.onnx.data”)
#     sidecar_name = None
#     try:
#         m = onnx.load(src_onnx.as_posix(), load_external_data=False)  # lightweight load
#         for t in m.graph.initializer:
#             if uses_external_data(t):
#                 sidecar_name = ExternalDataInfo(t).location
#                 break
#     except Exception:
#         # If anything goes wrong reading metadata, fall back to the common filename
#         sidecar_name = src_onnx.name + ".data"

#     # Compute src and dst sidecar paths
#     src_sidecar = (src_onnx.parent / sidecar_name) if not os.path.isabs(sidecar_name) else Path(sidecar_name)
#     dst_sidecar = models_dir / sidecar_name  # keep the same relative name inside workspace

#     # Ensure the destination parent exists (covers rare case where 'location' contains subdirs)
#     dst_sidecar.parent.mkdir(parents=True, exist_ok=True)

#     # Copy the sidecar if it exists (this is the piece that fixes your FileNotFoundError)
#     if src_sidecar.exists():
#         shutil.copy2(src_sidecar, dst_sidecar)
#     else:
#         # Clear message so you’ll know exactly what went wrong if it’s missing again
#         raise FileNotFoundError(f"Sidecar file not found: {src_sidecar} (needed by ONNX->MX import)")

#     # Proceed with partition export using the workspace path
#     partition.model_partition(name, plan, work_dir.as_posix())

#     plan_json_path = work_dir / "structure.json"
#     plan.to_json(plan_json_path.as_posix())

@click.command()
@click.argument('algo', type=str)
@click.option('-p', '--require-partition', type=bool, default=False)
@click.option('-n', '--name', type=str)
@click.option('-t', '--threshold', type=int)
@click.option('-d', '--rl-model-dir', type=str)
def main(algo, require_partition, name, threshold, rl_model_dir):
    if not algo in ['lo', 'sa', 'bf', 'bayesian']:
        print('Unknown algorithm.') 
        return
    
    if algo == 'lo':
        # latency-optimal
        ali_single_latency = {0.5: 5/4*5840.333333, 0.75: 5/4*3623.666667, 1.0: 5/4*2580.0, 1.5: 5/4*1728.75, 2.0: 5/4*1291.2}
        ali_single_cost    = {0.5: 5/4*7.6e-05,     0.75: 5/4*6.3e-05,     1.0: 5/4*5.6e-05,   1.5: 5/4*5.3e-05,  2.0: 5/4*5.1e-05}
        mem_to_vcpu = {
            2048: 0.5,
            3072: 0.75,
            4096: 1.0,
            6144: 1.5,
            8192: 2.0,
        }
        set_baseline(
            mode="memory",
            latency_map=ali_single_latency,   # <-- vCPU-keyed table is OK; the helper will translate
            cost_map=ali_single_cost,         # (recommended so ROI uses cost-loss, not just speedup)
            mem_to_vcpu=mem_to_vcpu
        )
        plan = algos[algo](name)
        predictor.debug_plan(plan)
        if require_partition:
            
            export_model_partitions(name, plan)

    elif algo == 'sa':
        # slo-aware
        benchmarks = {name : [threshold]}
        
        if require_partition:
            plan = policy_rl.gen_plan_with_model(name, rl_model_dir)
            workspace_name = 'rl_' + name.split('.')[0] + f'_{threshold}'
            export_model_partitions(name, plan, workspace_name=workspace_name)
        else:
            algos[algo](benchmarks)
    else:
        benchmarks = {name : [threshold]}

        benchmark_res = algos[algo](benchmarks)
        action = benchmark_res[(name, threshold)][2]
        if require_partition:
            plan = gen_plan_with_action(name, action)
            workspace_name = f'{algo}_' + name.split('.')[0] + f'_{threshold}'
            export_model_partitions(name, plan, workspace_name=workspace_name)

if __name__ == "__main__":
    main()