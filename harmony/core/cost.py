import math
import numpy as np
from harmony.core.latency import Latency
from harmony.core.util import Instance, batch_distribution
from typing import List, Tuple


class FunctionCost():
    def __init__(self) -> None:
        self.cpu_cost = 0.00009
        self.mem_cost = 0.000009
        self.gpu_cost = 0.00011
        self.invocation_cost = 0.009 / 10000

    def cost(self, duration: float, batch: int, instance: Instance, billed_second : bool = True) -> float:
        if instance.gpu is None or billed_second is False:
            gpu = 0
        else:
            gpu = instance.gpu
            duration = math.ceil(duration)
        return (self.invocation_cost +
                (instance.cpu * self.cpu_cost +
                 instance.mem * self.mem_cost +
                    gpu * self.gpu_cost) * duration) / batch

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
        b_avg = min(batch_max, int(r * t) + 1)
        return self.cost(lat_cal.lat_avg(instance, b_avg), b_avg, instance)
