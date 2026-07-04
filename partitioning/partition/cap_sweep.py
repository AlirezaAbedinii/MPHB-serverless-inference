"""Phase-2 cap sweep (temp tool).

Runs the partition optimizer for one (model, roi, max_stages, max_workers) config
and prints a one-line summary: latency / cost / #stages / workers-per-stage.

Run ONE config per process (the optimizer's memo notebooks are keyed by layer/mem,
not by caps, so cap changes within a process would return stale cached plans).

Usage:
  python3 cap_sweep.py <model.onnx> <roi:0|1> <max_stages> <max_workers>
"""
import sys
import parallel_util
import policy_dp
from parallel_util import set_baseline, hyper_params, get_info_graph
from policy_dp import gen_plan


def configure(roi, max_stages, max_workers):
    policy_dp.OPTIMIZE_BY_ROI = bool(roi)
    policy_dp.MAX_STAGES = max_stages
    parallel_util.MAX_STAGES = max_stages
    parallel_util.MAX_WORKER_PER_STAGE = max_workers
    # ROI baseline (mirror main.py lo path) so ROI mode has a single-CPU reference
    ali_single_latency = {0.5: 5/4*5840.333333, 0.75: 5/4*3623.666667, 1.0: 5/4*2580.0, 1.5: 5/4*1728.75, 2.0: 5/4*1291.2}
    ali_single_cost    = {0.5: 5/4*7.6e-05,     0.75: 5/4*6.3e-05,     1.0: 5/4*5.6e-05,   1.5: 5/4*5.3e-05,  2.0: 5/4*5.1e-05}
    mem_to_vcpu = {2048: 0.5, 3072: 0.75, 4096: 1.0, 6144: 1.5, 8192: 2.0}
    set_baseline(mode="memory", latency_map=ali_single_latency,
                 cost_map=ali_single_cost, mem_to_vcpu=mem_to_vcpu)


def summarize(model, roi, max_stages, max_workers):
    info_graph = get_info_graph(model)
    try:
        plan = gen_plan(info_graph)
    except AttributeError:
        # ExecutionPlan([]) has no stage_list -> DP bailed (too many physical stages)
        print(f"RESULT,{model},roi={roi},maxS={max_stages},maxW={max_workers},EMPTY_PLAN(DP_bail)")
        return
    if not getattr(plan, 'stage_list', None):
        print(f"RESULT,{model},roi={roi},maxS={max_stages},maxW={max_workers},EMPTY_PLAN")
        return
    lat = plan.cal_latency()
    cost = plan.cal_cost()
    stages = []
    total_workers = 0
    for s in plan.stage_list:
        nworkers = sum(1 for f in s.func_ids if f >= 1)
        total_workers += nworkers
        stages.append(f"{s.name[0]}_{s.name[1]}|shape={s.partition_shape}|w={len(s.models)}|fids={list(s.func_ids)}")
    print(f"RESULT,{model},roi={roi},maxS={max_stages},maxW={max_workers},"
          f"lat={lat:.1f},cost={cost:.3e},nstages={len(plan.stage_list)},workers={total_workers}")
    for st in stages:
        print("   ", st)


if __name__ == "__main__":
    model = sys.argv[1]
    roi = int(sys.argv[2])
    max_stages = int(sys.argv[3])
    max_workers = int(sys.argv[4])
    configure(roi, max_stages, max_workers)
    summarize(model, roi, max_stages, max_workers)
