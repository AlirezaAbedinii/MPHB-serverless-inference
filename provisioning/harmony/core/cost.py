import math
import numpy as np
from harmony.core.latency import Latency
from harmony.core.util import Instance, batch_distribution
from typing import List, Tuple


class FunctionCost():
    def __init__(self) -> None:
        # self.cpu_cost = 0.00009
        # self.mem_cost = 0.000009
        # self.gpu_cost = 0.00011
        # self.invocation_cost = 0.009 / 10000
        
        self.cpu_cost = 0.000016
        self.mem_cost = 0.000016 * 0.15
        self.invocation_cost = 0.00000012
        self.IDLE_GPU_PENALTY_05 = 1
        self.gpu_cost = 0.000034 * self.IDLE_GPU_PENALTY_05
        
        

    def cost(self, duration: float, batch: int, instance: Instance, billed_second : bool = True, total_rps: float = None) -> float:
        if instance.gpu is None or billed_second is False:
            gpu = 0
        else:
            gpu = instance.gpu 
            # duration = math.ceil(duration)
        cost_per_req = (self.invocation_cost +
                (instance.cpu * self.cpu_cost +
                 instance.mem * self.mem_cost +
                    gpu * self.gpu_cost) * duration) / batch
        
        # if instance.gpu is not None and billed_second and total_rps is not None and total_rps > 0:
        #     # Apply GPU idle billing adjustment when RPS is provided.
        #     cost_per_req = adjust_gpu_cost_for_idle(cost_per_req, instance, total_rps, self)
        return cost_per_req

    


    def calculate_total_cost_with_workers(self, master_latency: float,
                                        worker_latencies: list,
                                        batch_size: int,
                                        master_instance: Instance,
                                        worker_instances: Instance,
                                        billed_second: bool = True) -> float:
        """
        Compute total cost: master + all workers
        """
        
        
        # Master cost (already computed by HB, but here for clarity)
        master_cost = self.cost(
            duration=master_latency/1000,
            batch=batch_size,
            instance=master_instance,
            billed_second=billed_second
        )
        if worker_instances is None:
            return master_cost
        # Workers: sum individual costs
        worker_total_cost = 0.0
        workers_costs = []
        for index, worker_latency in enumerate(worker_latencies):
            worker_cost = self.cost(
                duration=worker_latency/1000,
                batch=batch_size,
                instance=worker_instances[index],
                billed_second=billed_second
            )
            worker_total_cost += worker_cost
            workers_costs.append(worker_cost)

        total_invocation_cost = len(worker_latencies)*self.invocation_cost
        # print(total_invocation_cost)
        total_cost = master_cost + worker_total_cost
        return total_cost

    
    
    def cost_with_distribution(self, time_out: float, rps: float, batch_max: int, lat_cal: Latency, instance: Instance) -> float:
        if batch_max == 1:
            return self.cost(lat_cal.lat_avg(instance, 1), 1, instance)
        p = batch_distribution(rps, batch_max, time_out)
        return self.cost_with_probability(instance, p, lat_cal)

    def cost_with_probability(self, instance: Instance, probability: List[float], lat_cal: Latency) -> float:
        c = 0.0
        for i in range(len(probability)):
            c += self.cost(lat_cal.lat_avg(instance, i + 1),
                           i+1, instance) * probability[i]
        return c

class sort_helper:
    def __init__(self, rps, t):
        self.rps = rps
        self.t = t
def equivalent_timeout(timeouts : List[float], rps : List[float]) -> Tuple[float, float]:
    assert len(timeouts) == len(rps)
    n = len(timeouts)
    if n == 1:
        return timeouts[0], rps[0]
    h = [sort_helper(rps[i], timeouts[i]) for i in range(n)]
    h.sort(key=lambda x: x.t)
    rps = [i.rps for i in h]
    timeouts = [i.t for i in h]
    rps_total = sum(rps)
    if n == 2:
        return timeouts[0] + rps[1] / rps_total * np.exp(-rps[0] * timeouts[1] - timeouts[0]), rps_total
    else:
        t, r = equivalent_timeout(timeouts[0:2], rps[0:2])
        return equivalent_timeout([t] + timeouts[2:], [r] + rps[2:])

class Multi_Cost(FunctionCost):
    def cost_with_multi_timeout_and_rps(self, time_out: List[float], rps: List[float], batch_max: int, lat_cal: Latency, instance: Instance) -> float:
        if batch_max == 1:
            return self.cost(lat_cal.lat_avg(instance, 1), 1, instance)
        t, r = equivalent_timeout(time_out, rps)
        # b_avg = min(batch_max, int(r * t) + 1)
        b_avg = batch_max
        return self.cost(lat_cal.lat_avg(instance, b_avg), b_avg, instance)


def adjust_gpu_cost_for_idle(cost_per_req: float, instance: Instance, total_rps: float, fc: FunctionCost) -> float:
    
    c0 = cost_per_req
    cpu_rate = instance.cpu * fc.cpu_cost + instance.mem * fc.mem_cost
    gpu_active_rate = instance.gpu * fc.gpu_cost
    duration = (c0 - fc.invocation_cost) / (cpu_rate + gpu_active_rate)
    if duration <= 0:
        return c0
    T = 1 / total_rps
    idle_time = max(0, T - duration)
    IDLE_RATIO = 0.5/2.1
    gpu_idle_rate = gpu_active_rate * IDLE_RATIO
    cost = c0 + gpu_idle_rate * idle_time
    return cost