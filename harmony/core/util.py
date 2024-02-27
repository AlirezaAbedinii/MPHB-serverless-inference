from typing import  Union, List, Any, Tuple
import numpy as np
from scipy.linalg import expm


def batch_distribution(lam: float, B: int, T: float) -> List[float]:
    init_state = [0] * B
    init_state[0] = 1

    Q = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i != B-1 and j != B-1 and i == j:
                Q[i][j] = -lam
            elif j == i+1:
                Q[i][j] = lam

    p = [0.0] * B
    pTmp = np.dot(np.array(init_state), expm(Q * T))
    for i in range(B):
        if i < B-1:
            p[i] = pTmp[i]
        else:
            p[i] = 1 - sum(p)
    return p


class Mem:
    def __init__(self, model_config: dict, model_name: str) -> None:
        self.model_config = model_config[model_name]
        # CPU
        self.a, self.b = self.model_config["CPU"]["mem"]
        # GPU
        self.mem = self.model_config["GPU"]["mem"]
        self.gpu_mem = self.model_config["GPU"]["gpu_mem"]

    def get_mem(self, cpu, mem):
        mem = mem / 1024
        if 4 * cpu < mem:
            return None
        elif cpu > mem:
            mem = cpu
        mem = int(mem * 1024)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return mem / 1024

    def get_mini_cpu_batch(self, max_batch: int):
        cpus = []
        for batch in range(1, max_batch+1):
            mem = self.a * batch + self.b
            mem = int(mem)
            if mem % 64 != 0:
                mem = ((mem // 64) + 1) * 64
            cpus.append(mem / 4 / 1024)
        return cpus

    def get_cpu_mem(self, cpu: float, batch: int):
        mem = self.a * batch + self.b
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return self.get_mem(cpu, mem)

    def get_gpu_mem(self, gpu: float, batch: Union[int, None] = None):
        mem = self.mem
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        cpu_min = gpu / 3
        mem = mem / 1024
        if mem > cpu_min * 4:
            return None
        return mem

    def get_gpu_gpu_mem(self, batch: int):
        for i in range(len(self.gpu_mem)):
            if batch <= self.gpu_mem[i]:
                return i+1
        return len(self.gpu_mem) + 1


class Instance:
    def __init__(self, cpu: float, mem: Union[float, None], gpu: Union[int, None]) -> None:
        self.cpu = cpu
        if mem is None:
            self.mem = self.cpu * 2
        else:
            self.mem = mem
        self.gpu = gpu
    
    def __eq__(self, other : "Instance") -> bool:
        return self.cpu == other.cpu and self.mem == other.mem and self.gpu == other.gpu

    def set_cpu(self, cpu: float):
        self.cpu = cpu

    def set_mem(self, mem: float):
        self.mem = mem

    def set_gpu(self, gpu: int):
        self.gpu = gpu


class Cfg:
    def __init__(self, instance: Instance, batch_size: int, cost: float, 
                rps: Union[float, List[float]], slo: Union[float, List[float]], 
                timeout: Union[float, List[float]]) -> None:
        self.instance = instance
        self.batch_size = batch_size
        self.cost = cost
        self.rps = rps
        self.timeout = timeout
        self.slo = slo

    def set_apps(self, apps):
        self.apps = apps

    def __str__(self):
        ret = "cpu:\t\t%0.2f" % self.instance.cpu + "\n" + \
            "batch:\t\t%d" % self.batch_size + "\n" + \
            "rps:\t\t" + str(self.rps) + "\n" + \
            "timeout:\t" + str(self.timeout) + "\n" + \
            "cost:\t\t%0.3e" % self.cost + "\n" \
            "slo:\t\t" + str(self.slo) + "\n"

        if self.instance.gpu is not None:
            ret = "gpu:\t\t%d" % self.instance.gpu + "\n" + ret
        return "\n" + ret
        # return "\n---------------------------------\n" + ret + "---------------------------------\n"
    
    def update(self, cfg : Union["Cfg", None]):
        if cfg is not None and cfg.cost < self.cost:
            self.instance = cfg.instance
            self.batch_size = cfg.batch_size
            self.cost = cfg.cost
            self.rps = cfg.rps
            self.timeout = cfg.timeout
            self.slo = cfg.slo
        return self

class App:
    def __init__(self, name: str, slo: float, rps: float) -> None:
        self.name = name
        self.slo = slo
        self.rps = rps
        self.index = 0
    
    def __str__(self):
        return self.name + " " + str(round(self.slo, 1)) + " " + str(round(self.rps,1))
    
    def __repr__(self) -> str:
        return self.__str__()

class Apps:
    def __init__(self, apps: List[App]) -> None:
        self.apps = apps
        self.threshold = None
        if len(apps) > 0:
            self.apps_rps = sum(app.rps for app in self.apps)
            self.slo = min(self.apps, key=lambda x: x.slo).slo
        else:
            self.slo = np.Inf
            self.apps_rps = 0

    def add(self, app: Union[App, 'Apps', List[Any]]):
        if isinstance(app, App):
            self.apps.append(app)
            self.apps_rps += app.rps
            self.slo = min(self.slo, app.slo)
        else:
            if isinstance(app, Apps):
                app_list = app.apps
            else:
                app_list = app
            for app in app_list:
                self.add(app)
    
    def get_rps_slo(self) -> Tuple[List[float], List[float]]:
        self.apps.sort(key=lambda app: app.slo)
        rpses = []
        slos = []
        for app in self.apps:
            rpses.append(app.rps)
            slos.append(app.slo)
        return rpses, slos

    def remove(self, name: str):
        self.apps = [app for app in self.apps if app.name != name]
        self.apps_rps = sum(app.rps for app in self.apps)
        self.slo = min(self.apps, key=lambda x: x.slo).slo
    
    def set_cfg(self, cfg : Cfg):
        self.cfg = cfg

    def get_apps(self):
        self.apps.sort(key=lambda app: app.slo)
        return self.apps

    def get_rps(self):
        return self.apps_rps
    
    def __str__(self):
        self.apps.sort(key=lambda app: app.slo)
        name = [app.name for app in self.apps]
        return str(name)

    def __repr__(self) -> str:
        return self.__str__()


def timeout_rps_value(app1 : App, app2: App):
    if app1.slo > app2.slo:
        return timeout_rps_value(app2, app1)
    r1 = app1.rps
    r2 = app2.rps
    s1 = app1.slo
    s2 = app2.slo
    p = r2 / (r1 + r2) * np.exp(-r2 * (s2 - s1))
    return p
