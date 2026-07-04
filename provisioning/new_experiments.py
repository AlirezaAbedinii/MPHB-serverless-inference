import json
import os
import time
from harmony.algorithm.algorithm import Harmony
import harmony
import argparse
from harmony.core.util import Cfg
import harmony.serverless.request as csreqest
from typing import List, Union
from harmony.config import get_config, set_config


results_base_path = "/mnt/data/HarmonyBatch/Experiments/WRN50-4/relaxed_hb_vs_mine/"
# 87 for 2.0
experiment_offset = 1.0
"""
for WRN50-5
mine = 2.3
with_idle = 2.6

"""
def process_single_result(result: List[csreqest.Result]):
    """
    Aggregate SLO violations and cost over a list of Result objects.

    NOTE: This function is now defensive against empty result sets and
    missing fields to avoid crashes when the underlying request/response
    format changes (e.g., during partitioned vs single migrations).
    """
    slo_count = 0
    cost = 0.0
    total_count = 0
    bad_req = 0
    for r in result:
        # Guard against Result with no 'requests' field
        try:
            reqs = r.requests
        except AttributeError:
            continue
        total_count += len(reqs)
        for req in reqs:
            try:
                # Defensive: some malformed requests may not have latency
                print(req)
                latency = getattr(req, "latency", 0)
                wait_time = getattr(req, "wait_time", 0)
                slo = getattr(req, "slo", float("inf"))
                req_cost = getattr(req, "cost", 0.0)
            except Exception:
                
                # If anything unexpected happens, treat as bad request
                bad_req += 1
                continue

            if latency == 0:
                bad_req += 1
                continue
            
                
            if latency + wait_time > slo:
                print("latency:", latency, "wait_time:", wait_time, "slo:", slo)
                slo_count += 1
            cost += req_cost

    if total_count == 0:
        # Avoid division by zero in edge cases (e.g., no successful requests)
        slo_violation = 0.0
        avg_cost = 0.0
    else:
        slo_violation = slo_count / total_count
        avg_cost = cost / total_count
    print("bad_req:", bad_req)
    return slo_violation, avg_cost, total_count


def process_per_app_results(result_list):
    """
    Returns per-app stats from a list of Result objects:
    {
        app_name: {
            "count": N,
            "violations": V,
            "total_cost": C,
            "latencies": [...],
        }
    }
    """
    per_app = {}

    for r in result_list:
        for req in getattr(r, "requests", []):
            app = getattr(req, "app_name", None)
            if app is None:
                continue

            if app not in per_app:
                per_app[app] = {
                    "count": 0,
                    "violations": 0,
                    "total_cost": 0.0,
                    "latencies": []
                }

            lat = getattr(req, "latency", 0)
            wait = getattr(req, "wait_time", 0)
            slo = getattr(req, "slo", float("inf"))
            cost = getattr(req, "cost", 0.0)

            per_app[app]["count"] += 1
            per_app[app]["total_cost"] += cost
            per_app[app]["latencies"].append(lat)

            if lat + wait > slo:
                per_app[app]["violations"] += 1

    return per_app



def result_to_metric(results):
    """
    Convert the list of per-minute results into summary metrics.

    IMPORTANT CHANGE:
    - Now supports both 2-tuple (cpu_results, gpu_results) and
      3-tuple (cpu_results, gpu_results, part_results) formats to
      remain backward-compatible while adding partitioned support.
    """
    slo_violations = []
    costs = []
    count_counts = []
    total_cost = 0.0
    for result in results:
        # Backward-compatible: allow old (cpu, gpu) tuples
        if len(result) == 2:
            cpu_results, gpu_results = result
            part_results = []
        else:
            cpu_results, gpu_results, part_results = result
        aggregate = cpu_results + gpu_results + part_results
        slo_violation, cost, total_count = process_single_result(aggregate)
        slo_violations.append(slo_violation)
        costs.append(cost)
        count_counts.append(total_count)
        total_cost += cost * total_count
    print("slo_violations: ", slo_violations)
    print("costs: ", costs)
    print("count_counts: ", count_counts)
    print("total_cost: ", total_cost)


def configure_partition_workers2(cfg, worker_name_list):
    """
    OPTIONAL helper to reconfigure worker resources for partitioned runs
    using the topology in cfg.topology.

    - We only use worker *names* here; the master will actually invoke
      them internally. The URL mapping is not needed for resource changes.
    - This relies on csreqest.Sample.main(function_name, cpu, mem, gpu)
      from request.py.

    The function is wrapped in try/except so that any mismatch in topology
    shape or Sample API does not crash the experiment script.
    """
    try:
        topo = getattr(cfg, "topology", None)
        if not topo:
            return

        workers_cfg = topo.get("workers", [])
        if not workers_cfg:
            return

        # Map each worker cfg entry to a worker function name by index.
        for i, w_cfg in enumerate(workers_cfg):
            if i >= len(worker_name_list):
                break
            name = worker_name_list[i]
            cpu = w_cfg.get("cpu", None)
            mem = w_cfg.get("mem", None)
            # Partitioned workers are CPU-only in this setup (gpu=0)
            if cpu is None or mem is None:
                continue
            try:
                # Use Sample from request.py to update worker configuration
                sampler = csreqest.Sample()
                sampler.main(name, cpu, mem, 0)
                print(f"[partitioned] Reconfigured worker {name} to cpu={cpu}, mem={mem}")
            except Exception as e:
                # We log a warning but keep going so one bad worker doesn't kill the run
                print(f"[partitioned][WARN] Failed to reconfigure worker {name}: {e}")
    except Exception as e:
        print(f"[partitioned][WARN] configure_partition_workers encountered error: {e}")


def configure_partition_workers(cfg, worker_name_list):
    """
    Reconfigure worker resources for partitioned runs using cfg.topology.

    IMPORTANT:
    - This maps cfg.topology["workers"] to worker_name_list **by index**.
      We are assuming that:
        workers[0], workers[1] -> stage 0
        workers[2], workers[3] -> stage 1
        workers[4], workers[5] -> stage 2
      and that worker_name_list is ordered consistently with that.
    - Python 3.7+ preserves dict insertion order, so worker_name_list
      will follow the order defined in worker_urls literal.
    - If topology size does not match the number of workers, we warn
      but still do a best-effort mapping on the common prefix.
    """
    try:
        topo = getattr(cfg, "topology", None)
        if not topo:
            return

        workers_cfg = topo.get("workers", [])
        if not workers_cfg:
            return

        if len(workers_cfg) != len(worker_name_list):
            print("[partitioned][WARN] workers_cfg len != worker_name_list len:",
                  len(workers_cfg), "vs", len(worker_name_list))

        for i, w_cfg in enumerate(workers_cfg):
            if i >= len(worker_name_list):
                break
            name = worker_name_list[i]
            cpu = w_cfg.get("cpu", None)
            mem = w_cfg.get("mem", None)
            if cpu is None or mem is None:
                continue
            try:
                mem_mb = int(round(float(mem) * 1024))
            except Exception:
                mem_mb = 128
            if mem_mb < 128:
                mem_mb = 128
            try:
                sampler = csreqest.Sample()
                sampler.main(name, cpu, mem_mb, None)  # workers are CPU-only (gpu=0)
                print(f"[partitioned] Reconfigured worker {name} to cpu={cpu}, mem={mem}")
            except Exception as e:
                print(f"[partitioned][WARN] Failed to reconfigure worker {name}: {e}")
    except Exception as e:
        print(f"[partitioned][WARN] configure_partition_workers encountered error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='harmony')
    parser.add_argument('--config', type=str,
                        default='conf', help='config path')

    args = parser.parse_args()
    with open(os.path.join(args.config, "config.json"), 'r') as f:
        config = json.load(f)
    with open(os.path.join(config['cfg_path'], 'model.json'), 'r') as f:
        config["model_config"] = json.load(f)
    h = Harmony(get_config())

    duration_min: int = config["duration_min"]
    app_num: int = config["app_num"]
    slos: List[int] = config["slos"]
    start = 0
    app_names = ["app" + str(i) for i in range(1, app_num + 1)]
    trace_path = os.path.join(args.config, config["app_path"])
    traces = [csreqest.Trace(os.path.join(trace_path, app + ".csv"), duration_min, 1.0, start_time=start) for app in app_names]
    applications = [csreqest.Application(traces[i], slos[i], app_names[i]) for i in range(app_num)]
    if False:
        for app in applications:
            print(app.get_app(0))
        for trace in traces:
            print(trace)
    else:
        #### for WRN50-5
        # Base URLs for the three deployment types
            # MASTER_URL = "https://wmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
            # cpu_url = "https://wrn-single-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
            # gpu_url = "https://gwrn-single-xxxxxxxxxx.us-east-1.fcapp.run/invoke"

            # # Master + worker identifiers for the partitioned deployment
            # # NOTE: worker_urls mapping is used only for naming/order; the master
            # # internally invokes these workers for partitioned runs.
            # master_info = {"w5master": "https://wmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"}
            # worker_urls = {
            #     'from0to17Worker1': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',
            #     'from0to17Worker2': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',

            #     'from18to29Worker1': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',
            #     'from18to29Worker2': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',

            #     'from30to33Worker1': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',
            #     'from30to33Worker2': 'https://fromtoworker-xxxxxxxxxx.us-east-1-vpc.fcapp.run/invoke',
            # }
        #### for WRN50-4
        MASTER_URL = "https://wmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
        cpu_url = "https://singlemx-w-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
        gpu_url = "https://gwrn-single-xxxxxxxxxx.us-east-1.fcapp.run/invoke"

        # Master + worker identifiers for the partitioned deployment
        # NOTE: worker_urls mapping is used only for naming/order; the master
        # internally invokes these workers for partitioned runs.
        master_info = {"wmaster": "https://wmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"}
        worker_urls = {
            'wcfrom0to17Worker1': 'https://wcfromtoworker-xxxxxxxxxx.us-east-1.fcapp.run/invoke',
            'wcfrom0to17Worker2': 'https://wcfromtoworker-xxxxxxxxxx.us-east-1.fcapp.run/invoke',

            'wcfrom18to29Worker1': 'https://wcfromtoworker-xxxxxxxxxx.us-east-1.fcapp.run/invoke',
            'wcfrom18to29Worker2': 'https://wcfromtoworker-xxxxxxxxxx.us-east-1.fcapp.run/invoke',
        }
        # init function client
        # NOTE: default_partition_cfg is used as a template for partitioned masters.
        default_cpu_cfg = Cfg(harmony.Instance(16, 32, None), 1, 1, 1, 1, 1)  # edited by A
        default_gpu_cfg = Cfg(harmony.Instance(16, 32, 24), 1, 1, 1, 1, 1)  # edited by A
        default_partition_cfg = Cfg(
            harmony.Instance(0.5, 1, None),
            1, 1, 1, 1, 1,
            function_type='partitioned'
        )

        cpu_functions: List[Union[csreqest.Function, None]] = []
        gpu_functions: List[Union[csreqest.Function, None]] = []
        partition_functions: List[Union[csreqest.Function, None]] = []  # NEW: active partitioned master clients

        # Put the function name and function url here for real serverless inference
        # For CPU/GPU we point directly at the single-function endpoints.
        idle_cpu_functions: List[Union[csreqest.Function, None]] = [
            csreqest.Function("singlemx_w4", cpu_url, default_cpu_cfg, config)
            for _ in range(3)
        ]
        idle_gpu_functions: List[Union[csreqest.Function, None]] = [
            
            csreqest.Function("gwrn50-4_single", gpu_url, default_gpu_cfg, config)
            
        ]
        # NEW: idle pool for partitioned masters (using the master_info mapping)
        idle_partition_functions: List[Union[csreqest.Function, None]] = [
            csreqest.Function(list(master_info.keys())[0], MASTER_URL, default_partition_cfg, config)
            # for _ in range(2)
        ]

        results = []
        predicted_costs = []
        ts = []
        print("experiments duration is: ", duration_min)
        for iter in range(duration_min):
            print("time: ", iter)
            apps = [app.get_app(iter) for app in applications]
            # INTENDED_RPS = {
            #     "app1": 0.1,
            #     "app2": 0.2
            # }
            # for a in apps:
                # r = INTENDED_RPS.get(a.name)
            #     if r is not None:
            #         a.rps = r
            print("apps: ", apps)
            apps = [app for app in apps if app.rps > 0]
            total_rps = sum(app.rps for app in apps)

            t1 = time.time()
            groups, cost = harmony.Algorithm(apps)
            t2 = time.time()
            t = int((t2 - t1) * 1000)
            print("time is: ", t)
            ts.append(t)
            predicted_costs.append(cost)

            for group in groups:
                if group.config is not None:
                    group.config.set_apps(group.apps)

            # NEW: split configs into CPU(single), GPU(single), and PARTITIONED
            cpu_cfg_list = []
            gpu_cfg_list = []
            partition_cfg_list = []
            for group in groups:
                # print(group)
                cfg = group.config
                if cfg is None:
                    continue
                # Safely inspect function_type; default to 'single' if not present
                ftype = getattr(cfg, "function_type", "single")
                if ftype == "partitioned":
                    partition_cfg_list.append(cfg)
                elif cfg.instance.gpu is None:
                    cpu_cfg_list.append(cfg)
                else:
                    gpu_cfg_list.append(cfg)

            print("total function num: ", len(groups),
                "; cpu function num: ", len(cpu_cfg_list),
                "; gpu function num: ", len(gpu_cfg_list),
                "; partitioned function num: ", len(partition_cfg_list))

            new_cpu_functions = []
            new_gpu_functions = []
            new_partition_functions = []

            # cpu function which do not need change resource
            for i in range(len(cpu_cfg_list)):
                cpu_cfg = cpu_cfg_list[i]
                if cpu_cfg is None:
                    continue
                for j in range(len(cpu_functions)):
                    function = cpu_functions[j]
                    if function is not None and function.eq_cfg(cpu_cfg):
                        new_cpu_functions.append(function)
                        function.bind_cfg(cpu_cfg)
                        cpu_functions[j] = None
                        cpu_cfg_list[i] = None
                        break

            cpu_cfg_list = [cfg for cfg in cpu_cfg_list if cfg is not None]
            cpu_functions = [function for function in cpu_functions if function is not None]

            # gpu function which do not need change resource
            for i in range(len(gpu_cfg_list)):
                gpu_cfg = gpu_cfg_list[i]
                if gpu_cfg is None:
                    continue
                for j in range(len(gpu_functions)):
                    function = gpu_functions[j]
                    if function is not None and function.eq_cfg(gpu_cfg):
                        new_gpu_functions.append(function)
                        function.bind_cfg(gpu_cfg)
                        gpu_functions[j] = None
                        gpu_cfg_list[i] = None
                        break
            gpu_cfg_list = [cfg for cfg in gpu_cfg_list if cfg is not None]
            gpu_functions = [function for function in gpu_functions if function is not None]

            # NEW: partitioned functions which do not need resource change
            for i in range(len(partition_cfg_list)):
                part_cfg = partition_cfg_list[i]
                if part_cfg is None:
                    continue
                for j in range(len(partition_functions)):
                    function = partition_functions[j]
                    if function is not None and function.eq_cfg(part_cfg):
                        new_partition_functions.append(function)
                        function.bind_cfg(part_cfg)
                        partition_functions[j] = None
                        partition_cfg_list[i] = None
                        break
            partition_cfg_list = [cfg for cfg in partition_cfg_list if cfg is not None]
            partition_functions = [function for function in partition_functions if function is not None]

            # Return unused functions to idle pools
            idle_cpu_functions.extend(cpu_functions)
            idle_gpu_functions.extend(gpu_functions)
            idle_partition_functions.extend(partition_functions)
            cpu_functions = []
            gpu_functions = []
            partition_functions = []

            print("CPU functions need to be change resource: ", len(cpu_cfg_list))
            print("GPU functions need to be change resource: ", len(gpu_cfg_list))
            print("PARTITIONED functions need to be change resource: ", len(partition_cfg_list))

            # cpu function which need change resource
            for i in range(len(cpu_cfg_list)):
                cpu_cfg = cpu_cfg_list[i]
                if cpu_cfg is None:
                    continue

                assert len(idle_cpu_functions) > 0

                function = idle_cpu_functions.pop()
                assert function is not None
                function.set_cfg(cpu_cfg)
                new_cpu_functions.append(function)
            print('umad')
            # gpu function which need change resource
            for i in range(len(gpu_cfg_list)):
                gpu_cfg = gpu_cfg_list[i]
                if gpu_cfg is None:
                    continue

                assert len(idle_gpu_functions) > 0

                function = idle_gpu_functions.pop()
                assert function is not None
                function.set_cfg(gpu_cfg)
                new_gpu_functions.append(function)

            # NEW: partitioned functions which need change resource
            # We configure the master function and, optionally, reconfigure workers.
            for i in range(len(partition_cfg_list)):
                part_cfg = partition_cfg_list[i]
                if part_cfg is None:
                    continue

                assert len(idle_partition_functions) > 0

                function = idle_partition_functions.pop()
                assert function is not None
                # This will update the master FC function's vCPU/mem based on cfg.instance
                # function.set_cfg(part_cfg)
                # # Optionally reconfigure workers based on topology (wrapped in try/except)
                # print('umad p')
                # configure_partition_workers(part_cfg, list(worker_urls.keys()))

                # new_partition_functions.append(function)
                try:
                    topo = getattr(part_cfg, "topology", {}) or {}
                    master_topo = topo.get("master", {})
                    master_cpu = master_topo.get("cpu", None)
                    master_mem = master_topo.get("mem", None)

                    # Fallbacks: if topology is missing, fall back to instance values
                    if master_cpu is None:
                        master_cpu = part_cfg.instance.cpu
                    if master_mem is None:
                        master_mem = part_cfg.instance.mem
                    try:
                        mem_mb = int(round(float(master_mem) * 1024))
                    except Exception:
                        mem_mb = 128
                    if mem_mb < 128:
                        mem_mb = 128
                    # Reconfigure the master FC function by name (w5master)
                    sampler = csreqest.Sample()
                    master_name = list(master_info.keys())[0]
                    sampler.main(master_name, master_cpu, mem_mb, None)
                    print(f"[partitioned] Reconfigured master {master_name} "
                        f"to cpu={master_cpu}, mem={master_mem}")
                except Exception as e:
                    print(f"[partitioned][WARN] Failed to reconfigure master: {e}")

                # NOTE: we deliberately do NOT call function.set_cfg(part_cfg) here,
                # because set_cfg would use cfg.instance (sum of all vCPUs) to update
                # the master. Instead we just bind cfg to this client so request
                # generation uses the correct rps/batch/topology information.
                function.bind_cfg(part_cfg)

                # Optionally reconfigure workers based on topology (safe, best-effort)
                configure_partition_workers(part_cfg, list(worker_urls.keys()))

                new_partition_functions.append(function)

            cpu_functions = new_cpu_functions
            gpu_functions = new_gpu_functions
            partition_functions = new_partition_functions

            
            # ---- NEW: Warm-up active GPU + partitioned functions ----
            # for f in cpu_functions:
            #     try:
            #         f.start(1)
            #         f.finish()
            #         print('cpu warmup done')
            #     except:
            #         pass

            # for f in partition_functions:
            #     try:
            #         f.start(1)
            #         f.finish()
            #         print('partitioned warmup done')
            #     except:
            #         pass
            # for f in gpu_functions:
            #     try:
            #         f.start(2)
            #         f.finish()
            #         print('gpu warmup done')
            #     except:
            #         pass
            # for f in partition_functions:
            #     try:
            #         f.start(1)
            #         f.finish()
            #         print
            #     except:
            #         pass
            # for f in gpu_functions:
            #     try:
            #         f.start(2)
            #         f.finish()
            #         print('gpu warmup done 2')
            #     except:
                    # pass
            print('Warmup finished')
            # for f in cpu_functions:
            #     try:
            #         f.start(1)
            #         f.finish()
            #     except:
            #         pass
            # start serverless inference
            # NOTE: underlying Serverless code will handle 'batch' vs 'BATCH'
            # parameter name; here we just pass cfg.batch_size via the Cfg.
            for function in cpu_functions:
                assert function is not None
                print("CPU batch size is", function.cfg.batch_size)
                function.start(1)
            for function in gpu_functions:
                assert function is not None
                print("GPU batch size is", function.cfg.batch_size)
                function.start(1)
            for function in partition_functions:
                assert function is not None
                print("PARTITIONED batch size is", function.cfg.batch_size)
                function.start(1)

            cpu_results = []
            gpu_results = []
            partition_results = []

            for function in cpu_functions:
                assert function is not None
                cpu_results.append(function.finish())
            for function in gpu_functions:
                assert function is not None
                gpu_results.append(function.finish())
            for function in partition_functions:
                assert function is not None
                res = function.finish()

                # NEW: attempt to persist worker latencies + total latency
                # for partitioned runs, if the lower-level code exposes them.
                # We wrap in try/except so that any schema mismatch is non-fatal.
                try:
                    for req in getattr(res, "requests", []):
                        # Heuristics: check for worker latencies under several possible names
                        worker_lat = getattr(req, "worker_latencies", None)
                        if worker_lat is None:
                            worker_lat = getattr(req, "worker_latencies_ms", None)
                        if worker_lat is None:
                            worker_lat = getattr(req, "latencies", None)

                        total_lat = getattr(req, "latency", None)
                        if worker_lat is not None:
                            record = {
                                "total_latency": total_lat,
                                "worker_latencies": worker_lat,
                            }
                            with open("partitioned_worker_latencies_v2.0.jsonl", "a") as f:
                                f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"[partitioned][WARN] Failed to log worker latencies: {e}")

                partition_results.append(res)

            
            # New results storage
            # ===== NEW: Log per-request details for all function types =====
            with open(results_base_path+"detailed_requests_" + str(experiment_offset)+".jsonl", "a") as f:
                for (ftype, flist) in [
                    ("cpu", cpu_results),
                    ("gpu", gpu_results),
                    ("partitioned", partition_results)
                ]:
                    for r in flist:
                        for req in getattr(r, "requests", []):
                            # log_record = {
                            #     "minute": iter,
                            #     "app": getattr(req, "app_name", None),
                            #     "function_type": ftype,
                            #     "latency": getattr(req, "latency", None),
                            #     "wait_time": getattr(req, "wait_time", None),
                            #     "slo": getattr(req, "slo", None),
                            #     "cost": getattr(req, "cost", None)
                            # }
                            lat = getattr(req, "latency", None)
                            cold_flag = False
                            try:
                                # simple cold-start heuristic: >4 seconds or >4x median
                                cold_flag = (lat is not None and lat > 4.0)
                            except:
                                pass

                            log_record = {
                                "minute": iter,
                                "app": getattr(req, "app_name", None),
                                "function_type": ftype,
                                "latency": lat,
                                "wait_time": getattr(req, "wait_time", None),
                                "slo": getattr(req, "slo", None),
                                "cost": getattr(req, "cost", None),
                                "cold": cold_flag
                            }
                            f.write(json.dumps(log_record) + "\n")

            
            # ===== NEW: Log per-minute config summary =====
            with open(results_base_path+"configs_" + str(experiment_offset)+".jsonl","a") as f:
                for group in groups:
                    cfg = group.config
                    if cfg is None: continue
                    f.write(json.dumps({
                        "minute": iter,
                        "apps": [app.name for app in group.apps],
                        "function_type": getattr(cfg, "function_type", "single"),
                        "cpu": cfg.instance.cpu,
                        "mem": cfg.instance.mem,
                        "gpu": cfg.instance.gpu,
                        "batch": cfg.batch_size,
                        "topology": getattr(cfg, "topology", None)
                    }) + "\n")

            
            
            
            
            # Store all three types of results for this minute
            results.append((cpu_results, gpu_results, partition_results))

            # ---- NEW: Per-app minute summary ----
            per_app = process_per_app_results(cpu_results + gpu_results + partition_results)

            with open(results_base_path+"per_app_summary_" + str(experiment_offset)+".jsonl", "a") as f:
                for app_name, stats in per_app.items():
                    c = stats["count"]
                    viol = stats["violations"]
                    total_cost = stats["total_cost"]
                    avg_cost = total_cost / c if c > 0 else 0.0
                    slo_rate = viol / c if c > 0 else 0.0
                    avg_lat = sum(stats["latencies"]) / c if c > 0 else None

                    record = {
                        "minute": iter,
                        "app": app_name,
                        "slo_violation_rate": slo_rate,
                        "avg_cost": avg_cost,
                        "avg_lat": avg_lat,
                        "count": c
                    }
                    f.write(json.dumps(record) + "\n")

            
            # Aggregate SLO and cost across CPU/GPU/partitioned for the CSV
            slo_violation, cost, total_count = process_single_result(
                cpu_results + gpu_results + partition_results
            )
            file_name = "result.csv"
            with open(file_name, "a") as f:
                f.write(
                    
                    str(slo_violation)
                    + ","
                    + str(cost)
                    + ","
                    + str(total_count)
                    + ","
                    + str(predicted_costs[-1])
                    + "\n"
                )
        

            print("[cooldown] sleeping 30 seconds to allow GPU/CPU functions to shut down...")
            if iter == duration_min - 1:
                print("last round, do not need to cooldown")
                break
            time.sleep(30)
            # ==== WARMUP PHASE ====
            print("[warmup] sending two warm-up requests to GPU and partitioned…")
    
            time.sleep(45)
            print('Warmup phase done')
            

           
        
        
        result_to_metric(results)
        print("predicted_costs: ", predicted_costs)
        print(ts)
        print("avg ts: ", sum(ts) / len(ts))
