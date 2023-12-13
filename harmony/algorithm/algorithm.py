from typing import List, Union, Generator
import numpy as np
from abc import ABC, abstractmethod
from bayes_opt import BayesianOptimization
import copy


from harmony.core.cost import FunctionCost
from harmony.core.util import Instance, Cfg, batch_distribution, Mem, App, Apps
from harmony.core.latency import CPULatency, CPULatency_AVG, GPULatency, GPULatency_AVG


class FunctionCfg(ABC):
    def __init__(self, config) -> None:
        self.config = config
        self.model_name = self.config['model_name']
        self.function_type = ["CPU"]
        self.model_config = config["model_config"]
        self.mem_cal = Mem(self.model_config, self.model_name)
        self.mini_cpu = self.mem_cal.get_mini_cpu_batch(self.config["B_CPU"][1])
        self.get_lat_cal()
        self.get_cost_cal()

    @abstractmethod
    def get_config(self, apps : Apps) -> Union[Cfg, None]:
        pass

    @abstractmethod
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        pass

    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency_AVG(
            self.model_config[self.model_name]['CPU'], self.model_name)
        # TODO: support both A10 and T4
        self.gpu_lat_cal = GPULatency_AVG(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
    
    def get_cost_cal(self) -> None:
        self.cost_cal = FunctionCost()

    def get_max_timeout(self, instance: Instance, batch_size: int, slo : float) -> float:
        if instance.gpu is None:
            return slo - self.cpu_lat_cal.lat_max(instance, batch_size)
        else:
            return slo - self.gpu_lat_cal.lat_max(instance, batch_size)

# TODO: support BATCH for baseline
class BATCH(FunctionCfg):
    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int]):
        cpu_cfg = None
        if "CPU" in self.function_type:
            cpu_cfg = self.get_config_with_one_platform(Res_CPU, B_CPU, False)
        gpu_cfg = None
        if "GPU" in self.function_type:
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
                tau = (b-1)/rps

                time_out = self.get_max_timeout(ins, b, self.slo)
                # constraint check
                if time_out < 0:
                    continue
                if b == 1:
                    time_out = 0
                p = batch_distribution(rps, b, time_out)
                lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
                cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
                tmp = Cfg(ins, b, cost, rps, self.slo, float(time_out), lat)
                if cfg is None:
                    cfg = tmp
                else:
                    cfg.update(tmp)
        return cfg

    def get_config(self, apps : Apps) -> Union[Cfg, None]:
        a, b = apps.get_rps_slo()
        return self.get_config_(sum(a), min(b))

    def get_config_(self, arrival_rate: float, slo: float) -> Union[Cfg, None]:
        self.arrival_rate = arrival_rate
        self.slo = slo

        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        B_CPU = list(range(B_CPU_low, B_CPU_high+1, 1))
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+1, 0.05))
        Res_CPU = [round(res, 2) for res in Res_CPU]

        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        B_GPU = list(range(B_GPU_low, B_GPU_high+1,1))
        # TODO: support both A10 and T4
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))

        return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU)

class Bayesian(BATCH):
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
        if batch == 1:
            time_out = 0
        p = batch_distribution(rps, batch, time_out)
        lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
        cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
        tmp = Cfg(ins, batch, cost, rps, self.slo, time_out, lat)
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
            p = batch_distribution(rps, batch, time_out)
            lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
            cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
            return -cost
        return minimize_cost_helper

    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
            pbounds = {'batch': (B[0], B[-1]), 'res': (Res[0], Res[-1])}
            optimizer = BayesianOptimization(f=self.minimize_cost(is_gpu), pbounds=pbounds, random_state=1, verbose=0)
            optimizer.maximize(init_point=1, n_iter=20)
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
        self.function_type = ["CPU", "GPU"]
    
    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name)
        # TODO: support both A10 and T4
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
    
    def get_config(self, apps : Apps) -> Union[Cfg, None]:
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
        # TODO: support both A10 and T4
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))

        return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU)

    
    def get_config_with_one_platform_gpu(self, Res: List, B: List[int]):
        rpses = self.arrival_rates
        lat_cal = self.gpu_lat_cal
        cfg = None

        for res in Res:
            for b in B:
                gpu = self.mem_cal.get_gpu_gpu_mem(b)
                if res < gpu:
                    break
                cpu = res / 3
                mem = self.mem_cal.get_gpu_mem(res, b)
                if mem is None:
                    break
                ins = Instance(cpu, mem, res)
                time_out = self.get_max_timeout(ins, b, self.slo)
                # constraint check
                if time_out < 0:
                    break
                if b == 1:
                    time_out = 0

                total_rps = sum(rpses)
                total_rps_sub = total_rps
                timeouts = [self.get_max_timeout(ins, b, slo) for slo in self.slos]
                P = []
                total_p = 1
                for i in range(len(timeouts)-1, -1, -1):
                    cur_p = total_p
                    for k in range(0, i):
                        cur_p *= np.exp(-rpses[i] * (timeouts[i] - timeouts[k]))
                    P = [cur_p * rpses[i] / total_rps_sub] + P
                    total_p -= cur_p * rpses[i] / total_rps_sub
                    total_rps_sub -= rpses[i]
                # print(P)
                lat = 0
                cost = 0
                
                for i in range(len(rpses)):
                    time_out = timeouts[i]
                    tau = (b-1) / total_rps
                    p = batch_distribution(total_rps, b, time_out)
                    lat += lat_cal.lat_with_probability(ins, p, time_out, tau)[1] * P[i]
                    cost += self.cost_cal.cost_with_probability(ins, p, lat_cal) * P[i]

                tmp = Cfg(ins, b, cost, rpses, self.slos, timeouts, lat)
                if cfg is None:
                    cfg = tmp
                else:
                    cfg.update(tmp)
        return cfg
    
    def get_config_with_one_platform_cpu(self, Res: List, B: List[int]):
        cfg = None
        for b in B:
            params = self.model_config[self.model_name]['CPU']['avg']['Exponential'][b-1]

            def cost_func_1d(c):
                return params[0] * (1-c/params[1]) * np.exp(-c/params[1]) + params[2]
            
            def slo_lat_func():
                if self.slo - params[2] <= 0:
                    return np.inf
                return np.log((self.slo - params[2]) / params[0]) * (-params[1])
            
            def slo_mem_func(b):
                return self.mini_cpu[b-1]

            res = None
            cost = None

            slo_lat_res = slo_lat_func()
            slo_mem_res = slo_mem_func(b)
            slo_res = max(slo_lat_res, slo_mem_res)

            if slo_res > Res[-1]:
                break

            Res = [r for r in Res if r >= slo_res]
            for r in Res:
                c = cost_func_1d(r)
                if cost is None or abs(c) < abs(cost):
                    cost = c
                    res = r
            if res is None or cost is None:
                break
            
            min_res = []
            if b == 1:
                min_res.append(res)
            else:
                if cfg is not None and cfg.cost <= cost:
                    break
                else:
                    min_res = [r for r in Res if r >= res]
            
                    range_num = 10
                    min_res.append(res)
                    i = 0
                    while True:
                        if len(min_res) >= range_num:
                            break
                        i += 1
                        x = round(res + 0.05 * i, 2) 
                        if x <= Res[-1]:
                            min_res.append(x)
                        else:
                            break
            
            # print(min_res)
            tmp = super().get_config_with_one_platform(min_res, [b], False)
            if cfg is None:
                cfg = tmp
            else:
                cfg.update(tmp)
        return cfg
    
    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool):
        if is_gpu is False:
            return self.get_config_with_one_platform_cpu(Res, B)
        else:
            return self.get_config_with_one_platform_gpu(Res, B)



def NewFunctionCfg(algorithm: str, config: dict) -> FunctionCfg:
    if algorithm == "BATCH":
        return BATCH(config)
    elif algorithm == "Bayesian":
        return Bayesian(config)
    elif algorithm == "Harmony":
        return Harmony(config)
    else:
        raise Exception("algorithm not support")
    

class Compose:
    def __init__(self, apps: List[App], algorithm) -> None:
        self.apps = apps
        self.apps = [app for app in self.apps if app.rps > 0]
        self.apps.sort(key=lambda app: app.slo)
        for i in range(len(self.apps)):
            self.apps[i].index = i
        self.algorithm = algorithm
        self.x_high = 0.5
        self.x_low = 0.1
    
    def post_check(self, apps_list : List[Apps]) -> bool:
        if self.algorithm == "Harmony":
            for apps in apps_list:
                x = apps.apps
                # strategy 1
                slo_min = apps.slo
                slo_max = slo_min
                for xi in x:
                    slo_max = max(slo_max, xi.slo)
                if slo_max - slo_min >= self.x_high:
                    return False
                # strategy 2
                indexes = [xi.index for xi in x]
                indexes.sort()
                for i in range(len(indexes)-1):
                    if indexes[i+1] - indexes[i] > 1:
                        return False
            return True
        return True


    def pre_process(self, apps: List[Apps]) -> List[Apps]:
        if self.algorithm == "Harmony":
            while len(apps) >= 4:
                cur_low = self.x_low + 0.0002
                cur_index = -1
                for i in range(1, len(apps)):
                    if apps[i].slo - apps[i-1].slo <= cur_low + 0.0001:
                        cur_low = apps[i].slo - apps[i-1].slo
                        cur_index = i
                if cur_index == -1:
                    break
                else:
                    apps[cur_index-1].add(apps[cur_index])
                    del apps[cur_index]
        return apps

    def get_apps(self) -> Generator[List[Apps], None, None]:
        origin_apps = self.apps
        if self.algorithm == "Bayesian":
            total_rps = sum([app.rps for app in origin_apps])
            for i in range(1, len(origin_apps)+1):
                copy_apps = copy.deepcopy(origin_apps)
                if i == 1:
                    yield [Apps(copy_apps)]
                else:
                    sub_rps = total_rps / i
                    app_list = [Apps([])]
                    for j in range(len(copy_apps)):
                        app = copy_apps[j]
                        while app_list[-1].get_rps() + app.rps >= sub_rps + 0.01:
                            tmp = sub_rps - app_list[-1].get_rps()
                            app1 = App(app.name, app.slo, tmp)
                            app2 = App(app.name, app.slo, app.rps - tmp)
                            app_list[-1].add(app1)
                            app_list.append(Apps([]))
                            app = app2
                        
                        app_list[-1].add(app)
                        if j == len(copy_apps) - 1:
                            break
                        elif app_list[-1].get_rps() >= sub_rps:
                                app_list.append(Apps([])) 
                    yield app_list
        elif self.algorithm == "BATCH":
            yield [Apps([app]) for app in origin_apps]
        else:
            apps = [Apps([app]) for app in origin_apps]
            apps = self.pre_process(apps)
            for apps_list in self.get_apps_helper(apps):
                if self.post_check(apps_list):
                    yield apps_list
                else:
                    continue

    def get_apps_helper(self, apps : Union[List[Apps], None] = None) -> Generator[List[Apps], None, None]:
        if apps is not None:
            n = len(apps)
            if n == 1:
                yield apps
            else:
                total_set = set(apps)
                cur = apps[0]
                for i in range(0, n):
                    for app in self.combine(apps[1:], i):
                        rest = list(total_set - set([cur] + app))
                        if len(rest) == 0:
                            app_list = []
                            for a in app:
                                app_list.extend(a.apps)
                            yield [Apps(cur.apps + app_list)]
                        else:
                            for sub_app in self.get_apps_helper(rest):
                                app_list = []
                                for a in app:
                                    app_list.extend(a.apps)
                                yield [Apps(cur.apps+app_list)] + sub_app
        

            
                

    def combine(self, apps: List[Apps], n) -> Generator[List[Apps], None, None]:
        if len(apps) < n:
            return None
        elif n == 0:
            yield []
        elif n == 1:
            for app in apps:
                yield [app]
        else:
            cur = [apps[0]]
            for app in self.combine(apps[1:], n-1):
                yield cur + app
            for app in self.combine(apps[1:], n):
                yield app

