"""Export a partition plan's artifacts (structure.json + per-stage MXNet sym/params)
into aws_lambda_deploy/<workspace>/input/ for deployment.

Bypasses main.py's hardcoded net0 copy hack. One config per process.

Usage:
  python3 export_plan.py <model.onnx> <roi:0|1> <max_stages> <max_workers> <workspace_name>
"""
import os
import sys
import parallel_util
import policy_dp
import partition
from parallel_util import set_baseline, get_info_graph
from policy_dp import gen_plan

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_ROOT = os.path.join(REPO, "aws_lambda_deploy")


def configure(roi, max_stages, max_workers):
    policy_dp.OPTIMIZE_BY_ROI = bool(roi)
    policy_dp.MAX_STAGES = max_stages
    parallel_util.MAX_STAGES = max_stages
    parallel_util.MAX_WORKER_PER_STAGE = max_workers
    ali_single_latency = {0.5: 5/4*5840.333333, 0.75: 5/4*3623.666667, 1.0: 5/4*2580.0, 1.5: 5/4*1728.75, 2.0: 5/4*1291.2}
    ali_single_cost    = {0.5: 5/4*7.6e-05,     0.75: 5/4*6.3e-05,     1.0: 5/4*5.6e-05,   1.5: 5/4*5.3e-05,  2.0: 5/4*5.1e-05}
    mem_to_vcpu = {2048: 0.5, 3072: 0.75, 4096: 1.0, 6144: 1.5, 8192: 2.0}
    set_baseline(mode="memory", latency_map=ali_single_latency,
                 cost_map=ali_single_cost, mem_to_vcpu=mem_to_vcpu)


def main(model, roi, max_stages, max_workers, workspace_name):
    configure(roi, max_stages, max_workers)
    info_graph = get_info_graph(model)
    plan = gen_plan(info_graph)
    if not getattr(plan, "stage_list", None):
        print("ERROR: empty plan (DP bailed) — raise max_stages")
        sys.exit(2)

    work_dir = os.path.join(DEPLOY_ROOT, f"{workspace_name}_workspace", "input")
    os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)
    partition.model_partition(model, plan, work_dir)
    plan.to_json(os.path.join(work_dir, "structure.json"))

    print(f"EXPORTED {workspace_name} -> {work_dir}")
    print(f"  latency={plan.cal_latency():.1f}ms cost={plan.cal_cost():.3e} "
          f"nstages={len(plan.stage_list)}")
    for s in plan.stage_list:
        print(f"    {s.name[0]}_{s.name[1]} shape={s.partition_shape} "
              f"workers={len(s.models)} fids={list(s.func_ids)}")


if __name__ == "__main__":
    model = sys.argv[1]
    roi = int(sys.argv[2])
    max_stages = int(sys.argv[3])
    max_workers = int(sys.argv[4])
    workspace_name = sys.argv[5]
    main(model, roi, max_stages, max_workers, workspace_name)
