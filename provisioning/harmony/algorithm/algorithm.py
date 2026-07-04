import math
from typing import List, Union
import numpy as np
from abc import ABC, abstractmethod
from bayes_opt import BayesianOptimization

from harmony.core.cost import FunctionCost, Multi_Cost, equivalent_timeout, adjust_gpu_cost_for_idle
from harmony.core.util import Instance, Cfg, Mem, Apps
from harmony.core.latency import CPULatency, CPULatency_AVG, GPULatency, GPULatency_AVG, EndToEndCPULatency
from harmony.core.latency import UnifiedLatency  # ✅ NEW IMPORT
from harmony.core.util import Instance

config_path = "/mnt/data/HarmonyBatch/conf/config.json"
# Arrival-rate cutoff above which partitioned mode is disabled.
# Module-level so experiments can toggle it (set to math.inf to disable the cutoff)
# without changing default behaviour.
PARTITION_RPS_THRESHOLD = 1.4
FUNCTION_TYPE = ["CPU",  "GPU", 'partitioned']
# FUNCTION_TYPE = ["CPU",  "GPU"]

class FunctionCfg(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.model_name = self.config['model_name']
        self.model_config = config["model_config"]
        self.mem_cal = Mem(self.model_config, self.model_name)
        self.mini_cpu = self.mem_cal.get_mini_cpu_batch(self.config["B_CPU"][1])
        self.get_lat_cal()
        self.get_cost_cal()

    @abstractmethod
    def get_config(self, apps : Apps, function_type : List[str] = ["CPU", "GPU"], need_one = False) -> Union[Cfg, None]:
        pass

    @abstractmethod
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        pass

    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency_AVG(
            self.model_config[self.model_name]['CPU'], self.model_name)
        self.gpu_lat_cal = GPULatency_AVG(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
    
    def get_cost_cal(self) -> None:
        self.cost_cal = Multi_Cost()
        self.func_cost = self.cost_cal
        
    def get_max_timeout(self, instance: Instance, batch_size: int, slo : float) -> float:
        if instance.gpu is None:
            return slo - self.cpu_lat_cal.lat_max(instance, batch_size)
        else:
            return slo - self.gpu_lat_cal.lat_max(instance, batch_size)

class BATCH(FunctionCfg):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int], function_type : List[str] = ["CPU"]):
        # print('hereeeee')
        cpu_cfg = None
        if "CPU" in function_type:
            cpu_cfg = self.get_config_with_one_platform(Res_CPU, B_CPU, False)
        gpu_cfg = None
        if "GPU" in function_type:
            gpu_cfg = self.get_config_with_one_platform(Res_GPU, B_GPU, True)
        if cpu_cfg is None:

            return gpu_cfg
        else:

            return cpu_cfg.update(gpu_cfg)
    
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        rps = self.arrival_rate
        if is_gpu:
            lat_cal = self.gpu_lat_cal
        else:
            lat_cal = self.cpu_lat_cal

        cfg = None
        for res in Res:
            for b in B:
                if is_gpu:
                    gpu = self.mem_cal.get_gpu_gpu_mem(b)
                    if res < gpu:
                        continue
                    cpu = res / 2  # Tesla 16 GB instance: 16 GB GPU → 8 vCPU (ratio 2:1)
                    mem = self.mem_cal.get_gpu_mem(res, b)
                    if mem is None:
                        continue
                    ins = Instance(cpu, mem, res)
                else:
                    if self.mini_cpu[b-1] > res:
                        continue
                    ins = Instance(res, self.mem_cal.get_cpu_mem(res, b), None)

                time_out = self.get_max_timeout(ins, b, self.slo)
                if time_out < 0:
                    continue
                if b == 1:
                    time_out = 0
                cost = self.cost_cal.cost_with_multi_timeout_and_rps([time_out], [rps], b, lat_cal, ins)
                tmp = Cfg(ins, b, cost, rps, self.slo, float(time_out))
                if cfg is None:
                    cfg = tmp
                else:
                    cfg.update(tmp)
        if cfg is None and self.need_one:
            ins = Instance(Res[-1], self.mem_cal.get_cpu_mem(res, 1), None)
            cost = self.cost_cal.cost_with_multi_timeout_and_rps([0], [rps], 1, lat_cal, ins)
            tmp = Cfg(ins, 1, cost, rps, self.slo, float(0))
            cfg = tmp
        return cfg

    def get_config(self, apps : Apps, function_type : List[str] = ["CPU"], need_one = False) -> Union[Cfg, None]:
        self.need_one = need_one
        a, b = apps.get_rps_slo()
        self.arrival_rates = a
        self.slos = b
        return self.get_config_(sum(a), min(b), function_type)

    def get_config_(self, arrival_rate: float, slo: float, function_type : List[str] = ["CPU"]) -> Union[Cfg, None]:
        self.arrival_rate = arrival_rate
        self.slo = slo

        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        B_CPU = list(range(B_CPU_low, B_CPU_high+1, 1))
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+0.05, 0.05))
        Res_CPU = [round(res, 2) for res in Res_CPU]

        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        B_GPU = list(range(B_GPU_low, B_GPU_high+1,1))
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))
        # print('here4')
        return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU, function_type)

class MBS(BATCH):
    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name)
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
    def get_config_from_optimizer(self, res, batch, is_gpu):
        rps = self.arrival_rate
        if is_gpu:
            lat_cal = self.gpu_lat_cal
        else:
            lat_cal = self.cpu_lat_cal
        if is_gpu:
            res = int(res)
            cpu = res / 2  # Tesla 16 GB instance: 16 GB GPU → 8 vCPU (ratio 2:1)
            mem = self.mem_cal.get_gpu_mem(res, batch)
            ins = Instance(cpu, mem, res)
        else:
            res = int(res / 0.05) * 0.05
            ins = Instance(res, self.mem_cal.get_cpu_mem(res, batch), None)
        tau = (batch-1)/rps

        time_out = self.get_max_timeout(ins, batch, self.slo)
        # constraint check
        if batch == 1 or time_out < 0:
            time_out = 0
            batch = 1
        if batch == 1:
            cost = self.cost_cal.cost_with_distribution(time_out, rps, batch, lat_cal, ins)
            tmp = Cfg(ins, batch, cost, rps, self.arrival_rates, time_out)
        else:
            time_outs = [time_out + slo-self.slo for slo in self.slos]
            cost = self.cost_cal.cost_with_multi_timeout_and_rps(time_outs, self.arrival_rates, batch, lat_cal, ins)
            tmp = Cfg(ins, batch, cost, rps, self.arrival_rates, time_outs)
        return tmp

    def minimize_cost(self, is_gpu):
        def minimize_cost_helper(res, batch):
            batch = int(batch)
            rps = self.arrival_rate
            if is_gpu:
                lat_cal = self.gpu_lat_cal
            else:
                lat_cal = self.cpu_lat_cal
            min_v = -10**9
            if is_gpu:
                res = int(res)
                gpu = self.mem_cal.get_gpu_gpu_mem(batch)
                if res < gpu:
                    return min_v
                cpu = res / 2  # Tesla 16 GB instance: 16 GB GPU → 8 vCPU (ratio 2:1)
                mem = self.mem_cal.get_gpu_mem(res, batch)
                if mem is None:
                    return min_v
                ins = Instance(cpu, mem, res)
            else:
                res = round(int(res / 0.05) * 0.05, 2)
                mem = self.mem_cal.get_cpu_mem(res, batch)
                if mem is None:
                    return min_v
                ins = Instance(res, mem, None)
            tau = (batch-1)/rps

            time_out = self.get_max_timeout(ins, batch, self.slo)
            # constraint check
            if time_out < 0:
                return min_v
            if batch == 1:
                time_out = 0
                cost = self.cost_cal.cost_with_distribution(time_out, rps, batch, lat_cal, ins)
                return -cost
            time_outs = [time_out + slo-self.slo for slo in self.slos]
            cost = self.cost_cal.cost_with_multi_timeout_and_rps(time_outs, self.arrival_rates, batch, lat_cal, ins)
            return -cost
        return minimize_cost_helper

    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
            pbounds = {'batch': (B[0], B[-1]), 'res': (Res[0], Res[-1])}
            optimizer = BayesianOptimization(f=self.minimize_cost(is_gpu), pbounds=pbounds, random_state=1, verbose=0)
            optimizer.maximize(init_point=1, n_iter=10)
            params = optimizer.max['params']
            if params is not None:
                bs = int(params['batch'])
                res = params['res']
                if is_gpu:
                    res = int(res)
                else:
                    res = round(int(res / 0.05) * 0.05, 2)
                cfg = self.get_config_from_optimizer(res, bs, is_gpu)
                return cfg
            return None


class Harmony(BATCH):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.apps_obj = None

    
    def get_lat_cal(self) -> None:
        # ✅ Replacing old EndToEndCPULatency with new UnifiedLatency class
        # print('YO')
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name
        )
        self.partitioned_lat_cal = UnifiedLatency(
            batch_size=1, model_type="avg"
        )
        # TODO: Change This
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)

    

    def _ternary_search(self, objective_func, low, high, tolerance=0.05):
        """
        Finds the vCPU value 'c' in the range [low, high] that
        minimizes the objective_func(c).

        Stops when the search range is smaller than our vCPU step (0.05).
        """
        # We run for a fixed number of iterations, which is more
        # than enough to narrow the range.
        # (log_1.5( (high-low)/tolerance ) iterations)
        for _ in range(100):
            if (high - low) < tolerance:
                break

            # Calculate two midpoints
            m1 = low + (high - low) / 3
            m2 = high - (high - low) / 3

            # Compare the objective at the two midpoints
            if objective_func(m1) < objective_func(m2):
                high = m2  # The minimum is in the left 2/3
            else:
                low = m1   # The minimum is in the right 2/3

        # Return the midpoint of the final range, rounded to the
        # nearest 0.05 vCPU increment
        optimal_vcpu = (low + high) / 2
        return round(optimal_vcpu * 20) / 20
    
    def estimate_worker_latencies(self, cpu: float) -> List[float]:
        """
        ✅ Predict latency for each worker based on CPU using UnifiedLatency
        """
        return self.partitioned_lat_cal.worker_latencies(cpu)
    
    def get_config(self, apps : Apps, function_type : List[str] = FUNCTION_TYPE) -> Union[Cfg, None]:
        
        # print('here with func types:'   , function_type)
        self.apps_obj = apps
        a, b = apps.get_rps_slo()
        self.arrival_rates = a
        self.slos = b

        self.arrival_rate = sum(self.arrival_rates)
        self.slo = min(self.slos)
        # print(self.slo)

        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        B_CPU = list(range(B_CPU_low, B_CPU_high+1, 1))
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+1, 0.05))
        Res_CPU = [round(res, 2) for res in Res_CPU]
        # print('Res_CPU:', Res_CPU)
        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        B_GPU = list(range(B_GPU_low, B_GPU_high+1,1))
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))
        # print('here5')
        return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU, function_type, force_cpu_only=False)

    
    def get_config_with_one_platform_gpu(self, Res: List, B: List[int]):
        rpses = self.arrival_rates
        lat_cal = self.gpu_lat_cal
        total_rps = sum(rpses)
        cfg = None
        B = list(reversed(B))
        for res in Res:
            for b in B:
                gpu = self.mem_cal.get_gpu_gpu_mem(b)
                if res < gpu:
                    continue
                cpu = res / 2  # Tesla 16 GB instance: 16 GB GPU → 8 vCPU (ratio 2:1)
                mem = self.mem_cal.get_gpu_mem(res, b)
                if mem is None:
                    break
                ins = Instance(cpu, mem, res)
                time_out = self.get_max_timeout(ins, b, self.slo)
                # constraint check
                if time_out < 0:
                    continue
                if b == 1:
                    timeouts = [0] * len(rpses)
                timeouts = [self.get_max_timeout(ins, b, slo) for slo in self.slos]
                t_eq, r_eq = equivalent_timeout(timeouts, rpses)
                if b == 1:
                    cost = self.cost_cal.cost_with_multi_timeout_and_rps(timeouts, rpses, b, lat_cal, ins)
                    # cost = adjust_gpu_cost_for_idle(cost, ins, total_rps, self.func_cost)  # CHANGE: apply GPU idle billing
                    tmp = Cfg(ins, b, cost, rpses, self.slos, timeouts)
                    if cfg is None:
                        cfg = tmp
                    else:
                        cfg.update(tmp)
                    break
                # if 1 + r_eq * t_eq > b:
                if math.floor(r_eq*t_eq + 1) == b:
                    cost = self.cost_cal.cost_with_multi_timeout_and_rps(timeouts, rpses, b, lat_cal, ins)
                    # cost = adjust_gpu_cost_for_idle(cost, ins, total_rps, self.func_cost)  # CHANGE: apply GPU idle billing
                    tmp = Cfg(ins, b, cost, rpses, self.slos, timeouts)
                    if cfg is None:
                        cfg = tmp
                    else:
                        cfg.update(tmp)
                    break
        return cfg
    
    # def get_config_with_one_platform_cpu(self, Res: List):
        
        cfg = None
        b = 1
        params_avg = self.model_config[self.model_name]['CPU']['avg']['Exponential'][b-1]
        params_max = self.model_config[self.model_name]['CPU']['max']['Exponential'][b-1]

        def cost_func_1d(c):
            return params_avg[0] * (1-c/params_avg[1]) * np.exp(-c/params_avg[1]) + params_avg[2]
        
        def slo_lat_func():
            if self.slo - params_max[2] <= 0:
                return np.inf
            return np.log((self.slo - params_max[2]) / params_max[0]) * (-params_max[1])
        
        def slo_mem_func(b):
            return self.mini_cpu[b-1]

        res = None
        cost = None

        slo_lat_res = slo_lat_func()
        slo_mem_res = slo_mem_func(b)
        slo_res = max(slo_lat_res, slo_mem_res)

        if slo_res > Res[-1]:
            return None

        Res = [r for r in Res if r >= slo_res]
        if len(Res) == 0:
            return None
        low_index = 0
        high_index = len(Res) - 1
        cost = abs(cost_func_1d(Res[low_index]))
        res = Res[low_index]
        while low_index < high_index:
            index = (low_index + high_index) // 2
            c = cost_func_1d(Res[index])
            if abs(c) < abs(cost):
                cost = c
                res = Res[index]
            if c < 0:
                low_index = index + 1
            else :
                high_index = index
        
        tmp = super().get_config_with_one_platform([res], [b], False)
        if cfg is None:
            cfg = tmp
        else:
            cfg.update(tmp)
        return cfg
    
    
    def get_config_with_one_platform_cpu(self, Res: List):
        """
        HarmonyBatch-faithful CPU configuration selection.
        Uses SLO only for feasibility, and real billing cost for optimization.
        """

        cfg = None
        b = 1  # CPU batch fixed to 1 (as in your design)

        params_avg = self.model_config[self.model_name]['CPU']['avg']['Exponential']
        params_max = self.model_config[self.model_name]['CPU']['max']['Exponential']

        # --- SLO-based minimum CPU constraint (HB-style) ---
        def slo_lat_constraint():
            g = params_max[b-1]
            if self.slo <= g[2]:
                return None
            cpu = -np.log((self.slo - g[2]) / g[0]) * g[1]
            if cpu <= 0:
                return None
            return cpu

        # --- Memory-based minimum CPU ---
        def slo_mem_constraint():
            return self.mini_cpu[b-1]

        # Compute minimum feasible CPU
        cpu_lb_lat = slo_lat_constraint()
        cpu_lb_mem = slo_mem_constraint()
        cpu_lb = max(cpu_lb_mem, cpu_lb_lat) if cpu_lb_lat is not None else cpu_lb_mem

        # Filter feasible CPU candidates
        candidates = [c for c in Res if c >= cpu_lb]
        if len(candidates) == 0:
            return None

        best_cfg = None
        best_cost = float("inf")

        # --- Enumerate candidates and pick minimum COST (HB semantics) ---
        for c in candidates:
            ins = Instance(c, self.mem_cal.get_cpu_mem(c, b), None)
            time_out = self.get_max_timeout(ins, b, self.slo)
            duration = self.cpu_lat_cal.lat_max(ins, b)
            cost = self.cost_cal.cost(duration, b, ins)
            tmp = Cfg(ins, b, cost, self.arrival_rate, self.slo, time_out)
            # tmp = super().get_config_with_one_platform([c], [b], False)
            # if tmp is None:
            #     continue

            # cost = tmp.cost  # REAL monetary cost

            if cost < best_cost:
                best_cost = cost
                best_cfg = tmp

        return best_cfg

    
    
    # def get_config_with_one_platform_cpu(self, Res: List):
    #     cfg = None
    #     b = 1  # ✅ Fixed batch size for now

    #     # ✅ Cost calculation using new latency + cost logic
    #     for res in Res:
    #         cpu = res
    #         mem = self.mem_cal.get_cpu_mem(cpu, b)
    #         if mem is None:
    #             continue

    #         instance = Instance(cpu=cpu, mem=mem, gpu=None)

    #         # 🔍 Predict per-worker latencies
    #         worker_latencies = self.estimate_worker_latencies(cpu)
    #         master_latency = self.cpu_lat_cal.total_latency(cpu)

    #         # 🔢 Total cost using your function
    #         cost = self.cost_cal.calculate_total_cost_with_workers(
    #             master_latency=master_latency / 1000,  # convert ms → seconds
    #             worker_latencies=worker_latencies,
    #             batch_size=b,
    #             master_instance=instance,
    #             worker_instances=[instance] * 16,
    #             billed_second=True
    #         )

    #         cfg_tmp = Cfg(instance, b, cost, self.arrival_rate, self.slo, timeout=0.0)
    #         if cfg is None:
    #             cfg = cfg_tmp
    #         else:
    #             cfg.update(cfg_tmp)

    #     return 
    
    ### For 16 workers
    # def get_config_with_one_platform_cpu(self, Res: List):
        
    #     cfg = None
    #     b = 1  # Currently batch size fixed to 1

    #     Res = [round(r, 2) for r in Res]  # Just in case
    #     feasible = []

    #     for res in Res:
    #         instance = Instance(res, self.mem_cal.get_cpu_mem(res, b), None)

    #         # ➕ Predict total latency
    #         total_latency = self.cpu_lat_cal.lat(vcpu=res)
    #         # print(f"Testing vCPU={res}: total_latency={total_latency:.2f}ms vs slo={self.slo * 1000:.2f}ms")

    #         if total_latency > self.slo*1000:
    #             continue

    #         # ➕ Predict worker latencies for 16 workers
    #         worker_latencies = [self.cpu_lat_cal.lat(vcpu=res, worker_id=i+1) for i in range(16)]

    #         # ➕ Cost is calculated via cost.py, just gather input here
    #         cost = self.cost_cal.calculate_total_cost_with_workers(
    #             master_latency=total_latency / 1000,  # convert ms to sec
    #             worker_latencies=worker_latencies,
    #             batch_size=b,
    #             master_instance=instance,
    #             worker_instances=[
    #                 Instance(1, 1, 0) for _ in range(16)  # ⚠️ default instance, update later if needed
    #             ],
    #             billed_second=True
    #         )

    #         tmp = Cfg(instance, b, cost, self.arrival_rate, self.slo, 0.0)
    #         if cfg is None:
    #             cfg = tmp
    #             # print('cfg was none')
    #         else:
    #             cfg.update(tmp)
    #     # print("cfg:", cfg)
    #     return cfg
    
    # --- algorithm.py (inside Harmony class) ---


    # --- algorithm.py (Replace your old function with this) ---

    def get_config_with_one_platform_partitioned(self, apps=None, slo_ms=None, rps_total=None):
        # print("Starting partitioned-CPU search...", self.slos)
        ###
        # Partitioned-CPU provisioner for WRN50-5 with 3 stages (2 workers each).
        # Master is fixed at config.master_resources.
        # Returns (best_cfg, best_cost, best_latency_p99)
        ###
        slo_ms = min(self.slos)*1000
        rps_total = self.arrival_rate
        ul = UnifiedLatency()
        lm = ul.models
        master_fixed = lm.get("master_resources", {"vcpu": 0.5, "mem": 2.0})
        architecture = lm.get("architecture")
        stages = architecture['stages']
        worker_per_stage = architecture['worker_per_stage']
        total_workers = stages * worker_per_stage
        
        # ✅ CHANGED: Get the vCPU search bounds from your main config
        # We no longer need vcpu_grid for the search
        min_vcpus_per_stage = lm.get("min_cpu_per_stage", [0.5, 0.75, 0.75])
        
        max_vcpu_global = self.config["Res_CPU"][1] # e.g., 4.0 or 8.0
        # print(max_vcpu_global)
        # --- batching feasibility (HB-style) ---
        if apps is not None:
            timeouts = self.slos
            rps_list = self.arrival_rates
            T_eq, r_eq = equivalent_timeout(timeouts, rps_list)  
            print("equivalent timeout and rps are:", T_eq, r_eq)  
            b_max = int(math.floor(r_eq * T_eq) + 1)
            b_max = max(1, b_max)
        else:
            T_eq, r_eq, b_max = 1.0, 1.0, 1
            
        batch_candidates = list(range(b_max, 0, -1))
        # print('Partitioned search batch candidates:', batch_candidates)
        best = None  # (cost, cfg_dict, p99)

        
    # ===================== NEW (generalized workers) =====================
        stage_workers = []   # stage_idx → tuple(worker_keys)
        w = 1
        for _ in range(stages):
            stage_workers.append(
                tuple(f"worker_{i}" for i in range(w, w + worker_per_stage))
            )
            w += worker_per_stage
        # ====================================================================
        # memory helper
        def choose_stage_mem_old(cpu, stage_idx, model_cfg):
            """
            stage_idx ∈ {0,1,2}  for stages 1..3
            model_cfg points to latency_models[...] tree from your config
            """
            # stage_workers = [
            #     ("worker_1", "worker_2"),   # stage 1
            #     ("worker_3", "worker_4"),   # stage 2
            #     ("worker_5", "worker_6"),   # stage 3
            # ]
            
            
            
            wA, wB = stage_workers[stage_idx]
            minA = model_cfg[wA].get("min_mem", 1.0)
            minB = model_cfg[wB].get("min_mem", 1.0)
            stage_min = max(minA, minB)

            lower = max(stage_min, cpu)
            upper = 4.0 * cpu

            if lower > upper:
                return None  # infeasible pairing
            return lower
        
        
        def choose_stage_mem(cpu, stage_idx, model_cfg):
            # ===================== MODIFIED =====================
            workers = stage_workers[stage_idx]   # generalized
            stage_min = max(model_cfg[w].get("min_mem", 1.0) for w in workers)
            # ===================================================

            lower = max(stage_min, cpu)
            upper = 4.0 * cpu

            if lower > upper:
                return None
            return lower

        
        
        def p99_proxy_per_tuple_old(c1, c2, c3, b, with_print=False):
            bk = f"batch_size_{b}"
            def w_p99(worker_key, cpu):
                val = ul._curve(bk, worker_key, "avg", cpu) # Using avg as base
                if val is None: return 1e9 # Infeasible
                # infl = ul.models.get("p99_factor_worker", 1.15)
                return val

            s1 = max(w_p99("worker_1", c1), w_p99("worker_2", c1))
            s2 = max(w_p99("worker_3", c2), w_p99("worker_4", c2))
            s3 = max(w_p99("worker_5", c3), w_p99("worker_6", c3))

            comm = ul.models.get("comm_overhead", {})
            theta0 = comm.get("theta0_ms", 150.0)
            theta1 = comm.get("theta1_ms_per_batch", 8.0)
            if with_print:
                print("Total latency: ", (s1 + s2 + s3) + (theta0 * 3.0) + theta1 * float(b),"stage1: ",s1, "stage2: ", s2, "stage3: ", s3, "thetas and batches:", theta0, theta1, b)
            return (s1 + s2 + s3) + (theta0 * 3.0) + theta1 * float(b)
        
        
        def p99_proxy_per_tuple(stage_cpus, b, with_print=False):
            bk = f"batch_size_{b}"

            def w_p99(worker_key, cpu):
                val = ul._curve(bk, worker_key, "avg", cpu)
                return val if val is not None else 1e9

            # ===================== MODIFIED =====================
            stage_latencies = []
            for stage_idx, cpu in enumerate(stage_cpus):
                stage_latencies.append(
                    max(w_p99(w, cpu) for w in stage_workers[stage_idx])
                )
            # ===================================================

            comm = ul.models.get("comm_overhead", {})
            theta0 = comm.get("theta0_ms", 150.0)
            theta1 = comm.get("theta1_ms_per_batch", 8.0)

            # ===================== MODIFIED =====================
            return sum(stage_latencies) + (theta0 * stages) + theta1 * float(b)
            # ===================================================

        
        
        for b in batch_candidates:
            bk = f"batch_size_{b}"
            if bk not in ul.models:
                continue

            # --- coordinate descent over (c1, c2, c3) ---
            
            # ✅ CHANGED: Use a list for stage CPUs
            ### Make sure to double check this part for correctness
            stage_cpus = min_vcpus_per_stage[:stages] # Start from min ([:] makes a copy)

            improved = True
            
            # ✅ CHANGED: This helper now just evaluates. It does *not* check the SLO.
            def evaluate_tuple_old(c1, c2, c3):
                p99 = p99_proxy_per_tuple(c1, c2, c3, b)

                # For cost, we need per-worker average latencies (billing)
                w_avg_all = [None]*6
                w_avg_all[0:2] = [ul._curve(bk, "worker_1", "avg", c1),
                                ul._curve(bk, "worker_2", "avg", c1)]
                w_avg_all[2:4] = [ul._curve(bk, "worker_3", "avg", c2),
                                ul._curve(bk, "worker_4", "avg", c2)]
                w_avg_all[4:6] = [ul._curve(bk, "worker_5", "avg", c3),
                                ul._curve(bk, "worker_6", "avg", c3)]

                if None in w_avg_all: # Check if any vCPU was out of bounds
                    return None 

                master_avg = ul.pipeline_proxy_per_stage([c1, c2, c3], batch=b)

                inst_master = Instance(cpu=master_fixed["vcpu"], mem=master_fixed["mem"], gpu=None)
                mem1 = choose_stage_mem(c1, 0, ul.models[bk])
                mem2 = choose_stage_mem(c2, 1, ul.models[bk])
                mem3 = choose_stage_mem(c3, 2, ul.models[bk])
                
                if None in (mem1, mem2, mem3):
                    return None  # infeasible due to min_mem or coupling

                inst_workers = [
                    Instance(cpu=c1, mem=mem1, gpu=None), Instance(cpu=c1, mem=mem1, gpu=None),
                    Instance(cpu=c2, mem=mem2, gpu=None), Instance(cpu=c2, mem=mem2, gpu=None),
                    Instance(cpu=c3, mem=mem3, gpu=None), Instance(cpu=c3, mem=mem3, gpu=None),
                ]

                cost = FunctionCost().calculate_total_cost_with_workers(
                    master_latency=master_avg,
                    worker_latencies=w_avg_all,
                    batch_size=b,
                    master_instance=inst_master,
                    worker_instances=inst_workers
                )
                return cost, p99, inst_master, inst_workers
            
            def evaluate_tuple(stage_cpus):
                bk = f"batch_size_{b}"

                # ===================== MODIFIED =====================
                w_avg_all = []
                inst_workers = []

                for stage_idx, cpu in enumerate(stage_cpus):
                    mem = choose_stage_mem(cpu, stage_idx, ul.models[bk])
                    if mem is None:
                        return None

                    for w in stage_workers[stage_idx]:
                        lat = ul._curve(bk, w, "avg", cpu)
                        if lat is None:
                            return None
                        w_avg_all.append(lat)
                        inst_workers.append(Instance(cpu=cpu, mem=mem, gpu=None))
                # ===================================================

                master_avg = ul.pipeline_proxy_per_stage(stage_cpus, batch=b)
                inst_master = Instance(cpu=master_fixed["vcpu"], mem=master_fixed["mem"], gpu=None)

                cost = FunctionCost().calculate_total_cost_with_workers(
                    master_latency=master_avg,
                    worker_latencies=w_avg_all,
                    batch_size=b,
                    master_instance=inst_master,
                    worker_instances=inst_workers
                )

                p99 = p99_proxy_per_tuple(stage_cpus, b)
                return cost, p99, inst_master, inst_workers

            
            # ✅ CHANGED: Coordinate descent now uses ternary search
            for _ in range(20): # Run for a fixed number of iterations or until convergence
                if not improved:
                    break
                improved = False
                
                for stage_idx in range(stages):
                    
                    # 1. Define the 1D objective function for the search
                    def stage_objective(vcpu_candidate):
                        test_tuple = list(stage_cpus) # current config
                        test_tuple[stage_idx] = vcpu_candidate # vary this stage
                        
                        res = evaluate_tuple(test_tuple)
                        
                        if res is None:
                            # Hard infeasibility (e.g., memory or bad vCPU)
                            return 1e20 # A very large cost
                        
                        cost, p99, _, _ = res
                        
                        # Apply penalty function
                        penalty = max(0, p99 - slo_ms) * 1e9 # Large penalty factor
                        return cost + penalty

                    # 2. Get search bounds for this stage
                    min_c = min_vcpus_per_stage[stage_idx]
                    max_c = max_vcpu_global 
                    
                    # 3. Run the ternary search
                    optimal_c_stage = self._ternary_search(stage_objective, min_c, max_c)

                    # 4. Update the config and check for improvement
                    if abs(stage_cpus[stage_idx] - optimal_c_stage) > 0.01:
                        improved = True
                    stage_cpus[stage_idx] = optimal_c_stage

            # --- End of optimization loop ---
            
            # Now, evaluate the final optimized tuple (c1, c2, c3)
            final_res = evaluate_tuple(stage_cpus)

            if final_res is not None:
                cost, p99, inst_master, inst_workers = final_res
                
                # ✅ CHANGED: Final SLO check
                if p99 > slo_ms:
                    # print(f"  > b={b} config ({stage_cpus}) optimized but failed SLO (p99={p99:.0f}ms)")
                    # p99_proxy_per_tuple(*stage_cpus, b, with_print=True)
                    continue # This config is invalid

                # print(f"  > b={b} config ({stage_cpus}) is valid! (cost={cost:.6f}, p99={p99:.0f}ms)")
                # p99_proxy_per_tuple(1,1,1, b, with_print=True)
                # This is a valid candidate, see if it's the best
                if (best is None) or (cost < best[0]):
                    best = (cost, stage_cpus, p99, inst_master, inst_workers, b)
                    
        if best is None:
            # print("No feasible partitioned configuration found.")
            return None
        
        # Unpack the best-ever configuration
        best_cost, stage_cpus, best_p99, inst_master, inst_workers, best_batch = best

        # flatten total CPU/mem for bookkeeping (sum of all worker resources)
        total_cpu = sum(w.cpu for w in inst_workers) + inst_master.cpu
        total_mem = sum(w.mem for w in inst_workers) + inst_master.mem
        aggregate_instance = Instance(cpu=total_cpu, mem=total_mem, gpu=None)
        
        best_cfg = Cfg(
            instance=aggregate_instance,
            batch_size=best_batch,
            cost=best_cost,
            rps=rps_total,
            slo=self.slo,
            timeout=T_eq or 0.0,
            function_type="partitioned",           # <— important
            topology={
                "master": {"cpu": inst_master.cpu, "mem": inst_master.mem},
                "workers": [{"cpu": w.cpu, "mem": w.mem} for w in inst_workers],
                "per_stage_cpu": stage_cpus,
                "p99_ms": best_p99
            }
        )
        return best_cfg
    
    
    def get_config_with_one_platform_partitioned2(self, apps=None, slo_ms=None, rps_total=None):
        ###
        # Partitioned-CPU provisioner for WRN50-5 with 3 stages (2 workers each).
        # Master is fixed at config.master_resources.
        # Returns (best_cfg, best_cost, best_latency_p99)
        ###
        # apps = self.apps_obj
        slo_ms = min(self.slos)*1000
        rps_total = self.arrival_rate
        ul = UnifiedLatency()   # or however you instantiate it
        lm = ul.models.get("latency_models", {})
        master_fixed = lm.get("master_resources", {"vcpu": 0.5, "mem": 1.0})
        vcpu_grid = lm.get("vcpu_grid", [0.75, 1, 2, 3, 4])

        # --- batching feasibility (HB-style) ---
        if apps is not None:
            timeouts = self.slos          # or per-app SLO slack in seconds
            rps_list = self.arrival_rates
            # timeouts  = [t/1000.0 for t in timeouts]
            T_eq, r_eq = equivalent_timeout(timeouts, rps_list)  
            print("equivalent timeout and rps are:", T_eq, r_eq)  
            b_max = int(math.floor(r_eq * T_eq) + 1)
            b_max = max(1, b_max)
        else:
            T_eq, r_eq, b_max = 1.0, 1.0, 1
        # we’ll try batch sizes descending (bigger first)
        batch_candidates = list(range(b_max, 0, -1))
        print('gpaodsapsod')
        best = None  # (cost, cfg, p99)

        # memory helper
        def mem_from_cpu(cpu):
            # return Mem.get_mem_from_cpu(cpu)  # your util enforces 1x..4x coupling
            return cpu*4  # simple 4x coupling

        def choose_stage_mem(cpu, stage_idx, model_cfg):
            """
            stage_idx ∈ {0,1,2}  for stages 1..3
            model_cfg points to latency_models[...] tree from your config
            """
            # map stage -> worker keys in config
            stage_workers = [
                ("worker_1", "worker_2"),   # stage 1
                ("worker_3", "worker_4"),   # stage 2
                ("worker_5", "worker_6"),   # stage 3
            ]
            wA, wB = stage_workers[stage_idx]
            minA = model_cfg[wA].get("min_mem", 1.0)
            minB = model_cfg[wB].get("min_mem", 1.0)
            stage_min = max(minA, minB)

            # provider coupling: mem in [cpu, 4*cpu]
            lower = max(stage_min, cpu)          # must be ≥ both stage_min and cpu
            upper = 4.0 * cpu

            if lower > upper:
                return None  # infeasible pairing
            # simplest policy: pick lowest legal mem (cheapest)
            # (replace with a discrete mem grid if your provider only sells specific sizes)
            return lower
        
        def p99_proxy_per_tuple(c1, c2, c3, b, with_print=False):
            bk = f"batch_size_{b}"
            # per-worker P99 at THE RIGHT stage CPU
            def w_p99(worker_key, cpu):
                val = ul._curve(bk, worker_key, "avg", cpu)
                if with_print:
                    print('val1: ', val)
                if val is None:
                    avg = ul._curve(bk, worker_key, "avg", cpu)
                    infl = ul.models.get("p99_factor_worker", 1.15)
                    val = avg * infl
                    if with_print:
                        print("val2: ", val)
                return val

            # stage maxima at their own CPUs
            
            s1 = max(w_p99("worker_1", c1), w_p99("worker_2", c1))
            # print(w_p99("worker_1", c1), w_p99("worker_2", c1))
            s2 = max(w_p99("worker_3", c2), w_p99("worker_4", c2))
            s3 = max(w_p99("worker_5", c3), w_p99("worker_6", c3))
            # print(w_p99("worker_1", c1), w_p99("worker_2", c1),  w_p99("worker_3", c2), w_p99("worker_4", c2),
                #   w_p99("worker_5", c3), w_p99("worker_6", c3))
            comm = ul.models.get("comm_overhead", {})
            theta0 = comm.get("theta0_ms", 150.0)
            theta1 = comm.get("theta1_ms_per_batch", 8.0)
            # print("s1, s2, s3, c1, c2, c3, b:",s1, s2, s3, c1, c2, c3, b)
            # print('per stage cpus: ',c1, c2, c3, b, 'per stage latencies:',s1,s2,s3,'p99:', (s1 + s2 + s3) + (theta0 * 3.0) + theta1 * float(b))
            # print(theta0, theta1)
            return (s1 + s2 + s3) + (theta0 * 3.0) + theta1 * float(b)
        
        for b in batch_candidates:
            bk = f"batch_size_{b}"
            if bk not in ul.models:   # skip batch sizes we don't have fits for yet
                continue

            # --- coordinate descent over (c1, c2, c3) ---
            # start from a small cpu for each stage; you can also seed from last best
            c1 = c2 = c3 = vcpu_grid[0]
            
            min_vcpus = lm.get("min_cpu_per_stage", [0.5, 0.75, 0.75])
            stage_cpus = min_vcpus
            improved = True
            best_local = None  # (cost, (c1,c2,c3), p99)
        
            def evaluate_tuple(c1, c2, c3):
                # predict end-to-end P99 using TOTAL curve if present
                # NOTE: total curve was fit when all stages shared the same vCPU.
                # If total exists and you want to stay exact, use the MAX of (c1,c2,c3)
                # as the vcpu argument. Otherwise, we fall back to the proxy automatically.
                p99 = p99_proxy_per_tuple(c1, c2, c3, b)
                # if b == 1:
                #             print(f"TRY b={b} c=({c1},{c2},{c3}) p99={p99:.1f} slo={slo_ms}")
                if p99 > slo_ms:
                    print('p99 too large:', p99, 'slo is:', slo_ms)
                    return None

                # For cost, we need per-worker average latencies (billing)
                # Use the per-stage CPUs for the right workers:
                # s1: w1,w2 use c1 | s2: w3,w4 use c2 | s3: w5,w6 use c3
                w_avg_all = [None]*6
                w_avg_all[0:2] = [ul._curve(f"batch_size_{b}", "worker_1", "avg", c1),
                                ul._curve(f"batch_size_{b}", "worker_2", "avg", c1)]
                w_avg_all[2:4] = [ul._curve(f"batch_size_{b}", "worker_3", "avg", c2),
                                ul._curve(f"batch_size_{b}", "worker_4", "avg", c2)]
                w_avg_all[4:6] = [ul._curve(f"batch_size_{b}", "worker_5", "avg", c3),
                                ul._curve(f"batch_size_{b}", "worker_6", "avg", c3)]

                # master avg (can use total_avg as the master stage runtime proxy);
                # if you later add an explicit 'master' curve, swap it here.
                master_avg = ul.pipeline_proxy_per_stage([c1, c2, c3], batch=b)

                # Instances: per-stage CPUs + fixed master
                inst_master = Instance(cpu=master_fixed["vcpu"], mem=master_fixed["mem"], gpu=None)
                mem1 = choose_stage_mem(c1, 0, ul.models[bk])  # bk = f"batch_size_{b}"
                mem2 = choose_stage_mem(c2, 1, ul.models[bk])
                mem3 = choose_stage_mem(c3, 2, ul.models[bk])
                if None in (mem1, mem2, mem3):
                    return None  # infeasible due to min_mem or coupling

                inst_workers = [
                    Instance(cpu=c1, mem=mem1, gpu=None),
                    Instance(cpu=c1, mem=mem1, gpu=None),
                    Instance(cpu=c2, mem=mem2, gpu=None),
                    Instance(cpu=c2, mem=mem2, gpu=None),
                    Instance(cpu=c3, mem=mem3, gpu=None),
                    Instance(cpu=c3, mem=mem3, gpu=None),
                ]
                # print('worker instances are:', [inst.cpu for inst in inst_workers], [inst.mem for inst in inst_workers], 'master instance is:', inst_master.cpu, inst_master.mem, 'worker latencies are:', w_avg_all, 'master latency is:', master_avg)
                cost = FunctionCost().calculate_total_cost_with_workers(
                    master_latency=master_avg,
                    worker_latencies=w_avg_all,
                    batch_size=b,
                    master_instance=inst_master,
                    worker_instances=inst_workers
                )
                return cost, p99, inst_master, inst_workers

            while improved:
                improved = False
                for stage_idx, cur in enumerate([c1, c2, c3]):
                    best_stage = None
                    for cand in vcpu_grid:
                        cand_tuple = (cand if stage_idx==0 else c1,
                                    cand if stage_idx==1 else c2,
                                    cand if stage_idx==2 else c3)
                        res = evaluate_tuple(*cand_tuple)
                        
                        if res is None:
                            continue
                        cost, p99, m, ws = res
                        if (best_stage is None) or (cost < best_stage[0]):
                            best_stage = (cost, cand_tuple, p99, m, ws, b)
                            
                    if best_stage is not None:
                        # update stage if it improves overall cost
                        old_tuple = (c1, c2, c3)
                        if (best_local is None) or (best_stage[0] < best_local[0]):
                            (c1, c2, c3) = best_stage[1]
                            best_local = best_stage
                            if best_stage[1] != old_tuple:
                                improved = True

            if best_local is not None:
                cost, (c1f, c2f, c3f), p99, inst_master, inst_workers, best_batch = best_local
                # print('hereeee')
                # cfg = {
                #     "batch": b,
                #     "stages": [
                #         {"cpu": c1f, "mem": mem_from_cpu(c1f), "workers": 2},
                #         {"cpu": c2f, "mem": mem_from_cpu(c2f), "workers": 2},
                #         {"cpu": c3f, "mem": mem_from_cpu(c3f), "workers": 2},
                #     ],
                #     "master": {"cpu": master_fixed["vcpu"], "mem": master_fixed["mem"]}
                # }
                if (best is None) or (cost < best[0]):
                    # print("best", best)
                    best = (cost, (c1f, c2f, c3f), p99, inst_master, inst_workers, best_batch)
                    
        if best is None:
            print("No feasible partitioned configuration found.")
            return None
        
        best_cost, (c1f, c2f, c3f), best_p99, inst_master, inst_workers, best_batch = best

        # master = best_cfg_dict["master"]
        # inst_master = Instance(cpu=master["cpu"], mem=master["mem"], gpu=0)
        # b = best_cfg_dict["batch"]

        # flatten total CPU/mem for bookkeeping (sum of all worker resources)
        total_cpu = sum(w.cpu for w in inst_workers) + inst_master.cpu
        total_mem = sum(w.mem for w in inst_workers) + inst_master.mem
        aggregate_instance = Instance(cpu=total_cpu, mem=total_mem, gpu=None)
        # print(best_cost, best_batch, best_p99)
        # TODO: T_eq was set to 0 initially. Verify if this is correct.
        best_cfg = Cfg(
            instance=aggregate_instance,
            batch_size=best_batch,
            cost=best_cost,
            rps=rps_total,
            slo=self.slo,
            timeout=T_eq or 0.0,
            function_type="partitioned",           # <— important
            topology={
                "master": {"cpu": inst_master.cpu, "mem": inst_master.mem},
                "workers": [{"cpu": w.cpu, "mem": w.mem} for w in inst_workers],
                "per_stage_cpu": [c1f, c2f, c3f],
                "p99_ms": best_p99
            }
        )
        # print("best config is:", best_cfg)
        return best_cfg

    # def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
    #     if is_gpu is False:
    #         return self.get_config_with_one_platform_cpu(Res)
    #     else:
    #         return self.get_config_with_one_platform_gpu(Res, B)
    
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        if is_gpu:
            # print('B is:', B)
            return self.get_config_with_one_platform_gpu(Res, B)
        else:
            # print('1')
            return self.get_config_with_one_platform_cpu(Res)

    
    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int], function_type: List[str] =FUNCTION_TYPE, force_cpu_only = False):
        cpu_cfg = None
        best_cfg = None
        partitioned_cfg = None
        if "partitioned" in function_type:
            # print('hereeeeeee')
            partitioned_cfg = self.get_config_with_one_platform_partitioned()
            if self.arrival_rate >= PARTITION_RPS_THRESHOLD:
                partitioned_cfg = None
            # print(partitioned_cfg)
            
            
        if "CPU" in function_type:
            # TODO: Verify the commented line below
            # if len(self.arrival_rates) == 1:
            cpu_cfg = self.get_config_with_one_platform(Res_CPU, B_CPU, False)
                # print('ummmda')
                
        gpu_cfg = None
        
        if "GPU" in function_type and not force_cpu_only:
            gpu_cfg = self.get_config_with_one_platform(Res_GPU, B_GPU, True)
            # print("gpu config is:", gpu_cfg, "cpu config is:", cpu_cfg, "partitioned config is:", partitioned_cfg)
        if cpu_cfg is None:
            print('CPU config is None #################################################')
            best_cfg = gpu_cfg
        else:
            # print('here1')
            cpu_cfg.update(gpu_cfg)
            best_cfg = cpu_cfg
        
        if best_cfg is None or best_cfg.partition_or_not(partitioned_cfg):
            # print('heree2')
            best_cfg = partitioned_cfg

        # print('best config is:', best_cfg)
        # print('cpu config:' , cpu_cfg)
        # print('gpu config:' , gpu_cfg)
        # print('partitioned config:' , partitioned_cfg)
        # print("Best configuration selected:", best_cfg)
        return best_cfg

def NewFunctionCfg(config: dict) -> FunctionCfg:
    algorithm = config["algorithm"]
    if algorithm == "BATCH":
        return BATCH(config)
    elif algorithm == "MBS":
        return MBS(config)
    elif algorithm == "Harmony":
        # print('here33')
        return Harmony(config)
    else:
        return Harmony(config)
