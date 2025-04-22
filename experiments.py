import json
import os
import time

import harmony
import argparse
from harmony.core.util import Cfg
import harmony.serverless.request as csreqest
from typing import List, Union


def process_single_result(result : List[csreqest.Result]):
    slo_count = 0
    cost = 0
    total_count = 0
    bad_req = 0
    for r in result:
        reqs = r.requests
        total_count += len(reqs)
        for req in reqs:
            if req.latency == 0:
                bad_req += 1
                continue
            if req.latency + req.wait_time > req.slo:
                slo_count += 1
            cost += req.cost
    slo_violation = slo_count / total_count
    cost = cost / total_count
    print("bad_req:", bad_req)
    return slo_violation, cost, total_count

def result_to_metric(results):
    slo_violations = []
    costs = []
    count_counts = []
    total_cost = 0
    for result in results:
        cpu_results, gpu_results = result
        result = cpu_results + gpu_results
        slo_violation, cost, total_count = process_single_result(result)
        slo_violations.append(slo_violation)
        costs.append(cost)
        count_counts.append(total_count)
        total_cost += cost * total_count
    print("slo_violations: ", slo_violations)
    print("costs: ", costs)
    print("count_counts: ", count_counts)
    print("total_cost: ", total_cost)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='harmony')
    parser.add_argument('--config', type=str,
                        default='conf', help='config path')

    args = parser.parse_args()

    with open(os.path.join(args.config, "config.json"), 'r') as f:
        config = json.load(f)
    with open(os.path.join(config['cfg_path'], 'model.json'), 'r') as f:
        config["model_config"] = json.load(f)

    duration_min : int = config["duration_min"]
    app_num : int = config["app_num"]
    slos : List[int] = config["slos"]
    start = 0
    app_names = ["app" + str(i) for i in range(1, app_num+1)]
    trace_path = os.path.join(args.config, config["app_path"])
    traces = [csreqest.Trace(os.path.join(trace_path, app + ".csv"), duration_min, 1.0, start_time=start) for app in app_names]
    applications = [csreqest.Application(traces[i], slos[i], app_names[i]) for i in range(app_num)]

    
    # init function client 
    default_cpu_cfg = Cfg(harmony.Instance(16, 32, None), 1, 1, 1, 1, 1) #edited by A
    default_gpu_cfg = Cfg(harmony.Instance(16, 32, 24), 1, 1, 1, 1, 1) # edited by A
    cpu_functions : List[Union[csreqest.Function, None]] = []
    # Put the function name and function url here for real serverless inference
    # This is for simulation, so we don't put the real function name and url here
    idle_cpu_functions : List[Union[csreqest.Function, None]] = [csreqest.Function("", "", default_cpu_cfg, config) for _ in range(app_num)]
    gpu_functions : List[Union[csreqest.Function, None]] = []
    idle_gpu_functions : List[Union[csreqest.Function, None]] = [csreqest.Function("", "", default_gpu_cfg, config) for _ in range(app_num)]


    results = []
    predicted_costs = []
    ts = []
    for iter in range(duration_min):
        print("time: ", iter)
        apps = [app.get_app(iter) for app in applications]
        print("apps: ", apps)
        apps = [app for app in apps if app.rps > 0]
        total_rps = sum(app.rps for app in apps)

        t1 = time.time()
        groups, cost = harmony.Algorithm(apps)
        t2 = time.time()
        t = int((t2-t1) * 1000)
        print(t)
        ts.append(t)
        predicted_costs.append(cost)
        # print(groups)
        # continue
        # break

        for group in groups:
            if group.config is not None:
                group.config.set_apps(group.apps)


        cpu_cfg_list  = [group.config for group in groups if group.config.instance.gpu is None]
        gpu_cfg_list  = [group.config for group in groups if group.config.instance.gpu is not None]

        print("total function num: ", len(groups), "; cpu function num: ", len(cpu_cfg_list), "; gpu function num: ", len(gpu_cfg_list))

        new_cpu_functions = []
        new_gpu_functions = []

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
        
        idle_cpu_functions.extend(cpu_functions)
        idle_gpu_functions.extend(gpu_functions)
        cpu_functions = []
        gpu_functions = []
        
        print("CPU functions need to be change resource: ", len(cpu_cfg_list))
        print("GPU functions need to be change resource: ", len(gpu_cfg_list))
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


        cpu_functions = new_cpu_functions
        gpu_functions = new_gpu_functions


        # # start serverless inference
        for function in cpu_functions:
            assert function is not None
            function.start(1)
        for function in gpu_functions:
            assert function is not None
            function.start(1)
        
        cpu_results = []
        gpu_results = []
        for function in cpu_functions:
            assert function is not None 
            cpu_results.append(function.finish())
        for function in gpu_functions:
            assert function is not None
            gpu_results.append(function.finish())
        
        results.append((cpu_results, gpu_results))
        slo_violation, cost, total_count = process_single_result(cpu_results + gpu_results)
        file_name = "result.csv"
        with open(file_name, "a") as f:
            f.write(str(slo_violation) + "," + str(cost) + "," + str(total_count) + "," + str(predicted_costs[-1]) + "\n")

    result_to_metric(results)
    print("predicted_costs: ", predicted_costs)
    print(ts)
    print("avg ts: ", sum(ts) / len(ts))