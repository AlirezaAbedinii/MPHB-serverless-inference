from typing import List, Tuple, Union, Dict
import math
from harmony.core.util import Instance, batch_distribution
import numpy as np
from abc import ABC, abstractmethod
import json
from harmony.config import get_config


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
        
        if gpu == self.G:  # full card: no fragmentation overhead
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

class EndToEndCPULatency(Latency):
    def __init__(self, params: dict, fit_type='Exponential'):
        super().__init__()
        self.params = params['min'][fit_type]  # assuming structure is: {"min": {"Exponential": {...}}}

    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        return self._lat(instance.cpu, batch_size)

    def lat_max(self, instance: Instance, batch_size: int) -> float:
        return self._lat(instance.cpu, batch_size)

    def _lat(self, cpu: float, batch_size: int) -> float:
        if batch_size not in self.params:
            raise ValueError(f"Unsupported batch size {batch_size}. Only 1 and 2 supported.")
        a, b, c = self.params[batch_size]
        return a * np.exp(b * cpu) + c


class UnifiedLatency:
    def __init__(self, batch_size: int = 1, model_type: str = "avg") -> None:
        """
        Unified latency model loader.
        :param config: Full configuration dict containing `latency_models`.
        :param batch_size: Current batch size (e.g., 1, 2).
        :param model_type: One of "min", "avg", or "max".
        """
        self.batch_key = f"batch_size_{batch_size}"
        self.model_type = model_type
        
        cfg = get_config()
        lat_models = cfg.get("partitioned_latency_models", {})
        
        # with open(config_path, "r") as f:
        #     config = json.load(f)


        # NEW: keep both the full tree and the selected batch slice
        self.models = lat_models
        self.latency_models = self.models.get(self.batch_key, {})  # current-batch sub-tree (workers/total)
        # self.latency_models = config.get("latency_models", {}).get(self.batch_key, {})

        # ===================== NEW (architecture-driven) =====================
        architecture = self.models.get("architecture", {"stages": 3, "worker_per_stage": 2})
        self.stages = architecture["stages"]                     # NEW
        self.worker_per_stage = architecture["worker_per_stage"] # NEW
        self.total_workers = self.stages * self.worker_per_stage # NEW
        # ====================================================================

        
        
    def lat(self, vcpu: float, worker_id: Union[int, None] = None) -> float:
        """
        Compute latency using the fitted exponential model.
        :param vcpu: number of vCPUs for the function.
        :param worker_id: index (1-based) of worker; None for total latency.
        :return: predicted latency (ms)
        """
        # select total or worker
        if worker_id is None:
            params = self.latency_models.get("total", {}).get(self.model_type)
        else:
            key = f"worker_{worker_id}"
            params = self.latency_models.get(key, {}).get(self.model_type)
        if params is None:
            raise ValueError(f"No latency model for worker={worker_id}, type={self.model_type}")
        a, b, c = params
        # exponential model: a * exp(b * vcpu) + c
        return a * np.exp(b * vcpu) + c

    # --- latency.py (additions to UnifiedLatency) ---

    def _curve(self, batch_key, key, which, vcpu):
        # key: "total" or "worker_1".."worker_6"
        coeffs = (self.models.get(batch_key, {})
                              .get(key, {})
                              .get(which, None))
        if not coeffs:
            return None
        a, b, c = coeffs
        # print("printasdasd:::",a,b,c, key, which, a * math.exp(b * vcpu) + c)
        # note: your config stores negative b for decay; this matches exp(b*vcpu)
        return a * math.exp(b * vcpu) + c

    def worker_latencies_avg_old(self, vcpu, batch=1):
        bk = f"batch_size_{int(batch)}"
        vals = []
        for wid in range(1, 7):
            key = f"worker_{wid}"
            val = self._curve(bk, key, "avg", vcpu)
            if val is None:
                raise ValueError(f"Missing avg curve for {bk}.{key}")
            vals.append(val)
        return vals
    
    def worker_latencies_avg(self, vcpu, batch=1):
        bk = f"batch_size_{int(batch)}"
        vals = []
        for wid in range(1, self.total_workers + 1):
            key = f"worker_{wid}"
            val = self._curve(bk, key, "avg", vcpu)
            if val is None:
                raise ValueError(f"Missing avg curve for {bk}.{key}")
            vals.append(val)
        return vals

    def worker_latencies_p99(self, vcpu, batch=1):
        bk = f"batch_size_{int(batch)}"
        vals = []
        for wid in range(1, self.total_workers + 1):
            key = f"worker_{wid}"
            val = self._curve(bk, key, "max", vcpu)
            if val is None:
                # fallback: inflate avg by factor from config
                avg = self._curve(bk, key, "avg", vcpu)
                infl = self.models.get("p99_factor_worker", 1.15)  # conservative default
                val = avg * infl
            vals.append(val)
        return vals

    def total_lat_avg(self, vcpus, batch=1):
        bk = f"batch_size_{int(batch)}"
        val = self._curve(bk, "total", "avg", vcpus)
        if val is not None:
            return val
        # fallback: proxy if 'total' not present (max of stage service + comm)
        return self._pipeline_proxy(vcpus, batch, which="avg")

    def total_lat_p99(self, vcpu, batch=1):
        bk = f"batch_size_{int(batch)}"
        val = self._curve(bk, "total", "max", vcpu)
        if val is None:
            # fallback: inflate avg or use proxy p99
            avg = self.total_lat_avg(vcpu, batch)
            infl = self.models.get("p99_factor_total", 1.20)
            val = avg * infl
        return val

    # --- helper used only when 'total' curve is missing ---
    def _pipeline_proxy(self, vcpu, batch=1, which="avg", stage_count=3):
        """Approximate end-to-end by max(stage service) + comm term.
        Stage service time is max(worker1, worker2) for each stage (1..3)."""
        if which == "avg":
            w = self.worker_latencies_avg(vcpu, batch)
        else:
            w = self.worker_latencies_p99(vcpu, batch)

        # group workers into 3 stages: (1,2), (3,4), (5,6)
        # s1 = max(w[0], w[1])
        # s2 = max(w[2], w[3])
        # s3 = max(w[4], w[5])
        # base = s1 + s2 + s3
        # ===================== MODIFIED =====================
        base = 0.0
        for s in range(self.stages):
            start = s * self.worker_per_stage
            end = start + self.worker_per_stage
            base += max(w[start:end])
        stage_count = self.stages
        # ===================================================

        # comm term from config; defaults are small but non-zero
        comm_cfg = self.models.get("comm_overhead", {})
        theta0 = comm_cfg.get("theta0_ms", 200.0)
        theta1 = comm_cfg.get("theta1_ms_per_batch", 5.0)
        return base + theta0*stage_count + theta1 * float(batch)

    def pipeline_proxy_per_stage(self, vcpus_per_stage, batch=1, which="avg"):
        """
        Compute end-to-end latency (avg or p99) for a pipeline where each stage
        can have its own vCPU configuration.
        :param vcpus_per_stage: list of vCPUs, one per stage (e.g. [c1, c2, c3])
        :param batch: batch size key
        :param which: "avg" or "max" (P99)
        :return: predicted end-to-end latency (ms)
        """
        bk = f"batch_size_{int(batch)}"
        model_cfg = self.models.get(bk, {})

        # Group workers dynamically (3 stages × 2 workers default)
        # stage_workers = [
        #     ("worker_1", "worker_2"),
        #     ("worker_3", "worker_4"),
        #     ("worker_5", "worker_6"),
        # ]
        # ===================== MODIFIED =====================
        stage_workers = []
        w = 1
        for _ in range(self.stages):
            stage_workers.append(
                tuple(f"worker_{i}" for i in range(w, w + self.worker_per_stage))
            )
            w += self.worker_per_stage
        # ===================================================

        def w_curve(worker_key, cpu):
            val = self._curve(bk, worker_key, which, cpu)
            if val is None and which == "max":
                avg = self._curve(bk, worker_key, "avg", cpu)
                infl = self.models.get("p99_factor_worker", 1.15)
                val = avg * infl
            return val

        s_lat = []
        for stage_idx, workers in enumerate(stage_workers):
            cpu = vcpus_per_stage[stage_idx]
            w_lat = [w_curve(w, cpu) for w in workers]
            s_lat.append(max(w_lat))

        comm_cfg = self.models.get("comm_overhead", {})
        theta0 = comm_cfg.get("theta0_ms", 200.0)
        theta1 = comm_cfg.get("theta1_ms_per_batch", 8.0)
        # return sum(s_lat) + theta0 * len(stage_workers) + theta1 * float(batch)
        # ===================== MODIFIED =====================
        return sum(s_lat) + theta0 * self.stages + theta1 * float(batch)
        # ===================================================
