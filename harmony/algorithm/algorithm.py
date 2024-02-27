from typing import List, Union
import numpy as np
from abc import ABC, abstractmethod
from bayes_opt import BayesianOptimization

from harmony.core.cost import Multi_Cost, equivalent_timeout
from harmony.core.util import Instance, Cfg, Mem, Apps
from harmony.core.latency import CPULatency, CPULatency_AVG, GPULatency, GPULatency_AVG


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

    def get_max_timeout(self, instance: Instance, batch_size: int, slo : float) -> float:
        if instance.gpu is None:
            return slo - self.cpu_lat_cal.lat_max(instance, batch_size)
        else:
            return slo - self.gpu_lat_cal.lat_max(instance, batch_size)

class BATCH(FunctionCfg):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int], function_type : List[str] = ["CPU"]):
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
                    cpu = res / 3
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
            cpu = res / 3
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
                cpu = res / 3
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
    
    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name)
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
    
    def get_config(self, apps : Apps, function_type : List[str] = ["CPU", "GPU"]) -> Union[Cfg, None]:
        a, b = apps.get_rps_slo()
        self.arrival_rates = a
        self.slos = b

        self.arrival_rate = sum(self.arrival_rates)
        self.slo = min(self.slos)

        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        B_CPU = list(range(B_CPU_low, B_CPU_high+1, 1))
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+1, 0.05))
        Res_CPU = [round(res, 2) for res in Res_CPU]

        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        B_GPU = list(range(B_GPU_low, B_GPU_high+1,1))
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))

        return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU, function_type)

    
    def get_config_with_one_platform_gpu(self, Res: List, B: List[int]):
        rpses = self.arrival_rates
        lat_cal = self.gpu_lat_cal
        cfg = None
        B = list(reversed(B))
        for res in Res:
            for b in B:
                gpu = self.mem_cal.get_gpu_gpu_mem(b)
                if res < gpu:
                    continue
                cpu = res / 3
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
                t_eq, total_rps = equivalent_timeout(timeouts, rpses)
                if b == 1:
                    cost = self.cost_cal.cost_with_multi_timeout_and_rps(timeouts, rpses, b, lat_cal, ins)
                    tmp = Cfg(ins, b, cost, rpses, self.slos, timeouts)
                    if cfg is None:
                        cfg = tmp
                    else:
                        cfg.update(tmp)
                    break
                if 1 + total_rps * t_eq > b:
                    cost = self.cost_cal.cost_with_multi_timeout_and_rps(timeouts, rpses, b, lat_cal, ins)
                    tmp = Cfg(ins, b, cost, rpses, self.slos, timeouts)
                    if cfg is None:
                        cfg = tmp
                    else:
                        cfg.update(tmp)
                    break
        return cfg
    
    def get_config_with_one_platform_cpu(self, Res: List):
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
    
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        if is_gpu is False:
            return self.get_config_with_one_platform_cpu(Res)
        else:
            return self.get_config_with_one_platform_gpu(Res, B)
    
    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int], function_type: List[str] = ["CPU", "GPU"]):
        cpu_cfg = None
        if "CPU" in function_type:
            if len(self.arrival_rates) == 1:
                cpu_cfg = self.get_config_with_one_platform(Res_CPU, B_CPU, False)
        gpu_cfg = None
        if "GPU" in function_type:
            gpu_cfg = self.get_config_with_one_platform(Res_GPU, B_GPU, True)
        if cpu_cfg is None:
            return gpu_cfg
        else:
            return cpu_cfg.update(gpu_cfg)

    

def NewFunctionCfg(config: dict) -> FunctionCfg:
    algorithm = config["algorithm"]
    if algorithm == "BATCH":
        return BATCH(config)
    elif algorithm == "MBS":
        return MBS(config)
    elif algorithm == "Harmony":
        return Harmony(config)
    else:
        return Harmony(config)
