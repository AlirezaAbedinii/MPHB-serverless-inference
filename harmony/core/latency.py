from typing import List, Tuple, Union
import math
from harmony.core.util import Instance, batch_distribution
import numpy as np
from abc import ABC, abstractmethod


class Latency(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        pass
    
    def lat_with_distribution(self, time_out : float, rps : float, batch_max : int, instance : Instance) -> Tuple[float, float]:
        if batch_max == 1:
            lat = self.lat_avg(instance, 1)
            return lat, lat
        p = batch_distribution(rps, batch_max, time_out)
        tau = (batch_max - 1) / rps
        return self.lat_with_probability(instance, p, time_out, tau)

    def lat_with_probability(self, instance : Instance, probability : List[float], time_out : float, tau : float) -> Tuple[float, float]:
        tmp = 0.0
        for i in range(len(probability)):
            tmp += probability[i] * (i+1)
        for i in range(len(probability)):
            probability[i] = probability[i] * (i+1) / tmp

        l = 0.0
        for i in range(len(probability)):
            l += self.lat_avg(instance, i + 1) * probability[i]
        wait_avg = time_out * (1 - probability[-1]) + min(time_out, tau) * probability[-1]
        return l, l + wait_avg

class CPULatency(Latency):
    def __init__(self, params: dict, model_name: str, fitting_metod : str = 'Exponential') -> None:
        super().__init__()
        self.model_name = model_name
        self.fitting_metod = fitting_metod

        self.params_avg = params['avg'][self.fitting_metod]
        self.params_max = params['max'][self.fitting_metod]

    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            g = self.params_avg[batch_size-1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return G
        elif self.fitting_metod == 'Polynomial':
            f = self.params_avg['f']
            g = self.params_avg['g']
            k = self.params_avg['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return F / G + k[0]
        return np.Inf

    def lat_max(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            g = self.params_max[batch_size-1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return G
        elif self.fitting_metod == 'Polynomial':
            f = self.params_max['f']
            g = self.params_max['g']
            k = self.params_max['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return F / G + k[0]
        return np.Inf


class CPULatency_AVG(CPULatency):
    def lat_max(self, instance: Instance, batch_size: int) -> float:
        return self.lat_avg(instance, batch_size)

class GPULatency(Latency):
    def __init__(self, params: dict, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.g1 = params['l1']
        self.g2 = params['l2']
        self.t = params['t']
        self.G = params['G']

        self.a = None
        self.b = None

        if 'a' in params:
            self.a = params['a']
        if 'b' in params:
            self.b = params['b']
    

    def lat_avg(self, instance: Instance, batch_size: int, a : Union[float, None] = None, b : Union[float, None] = None)->float:
        gpu = instance.gpu
        c = instance.cpu
        if c > 1:
            c = 1

        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if a is None:
            a = 1
        if b is None:
            b = 0

        L = self.g1 * batch_size + self.g2
        L1 = L * a
        L2 = L * b
        L = L1
        return self.G / gpu * L + L2 / c

    def lat_max(self, instance: Instance, batch_size: int, scale = 1.2, a : Union[float, None] = None, b : Union[float, None] = None)->float:
        gpu = instance.gpu
        c = instance.cpu
        if c > 1:
            c = 1

        if a is None:
            a = self.a
        if b is None:
            b = self.b
        
        if a is None:
            a = 1
        if b is None:
            b = 0
        
        if gpu == 24:
            scale = 1
        L = self.g1 * batch_size + self.g2
        L1 = L * a
        L2 = L * b
        L = L1 
        n = math.ceil(L / (gpu * self.t))
        # scale: overhead
        return ((self.G - gpu) * n * self.t + L) * scale + L2 / c

class GPULatency_AVG(GPULatency):
    def lat_max(self, instance: Instance, batch_size: int, scale = 1.2, a : Union[float, None] = None, b : Union[float, None] = None)->float:
        return self.lat_avg(instance, batch_size, a, b)
