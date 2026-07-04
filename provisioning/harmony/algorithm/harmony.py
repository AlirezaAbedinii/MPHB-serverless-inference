import time
from harmony.algorithm.algorithm import NewFunctionCfg
from harmony.config import get_config
from typing import List, Union
import copy

from harmony.core.util import App, Apps, Cfg
from harmony.core.cost import adjust_gpu_cost_for_idle  # CHANGE: import GPU idle billing helper

FUNCTION_TYPE = ["CPU",  "GPU", 'partitioned']
# FUNCTION_TYPE = ["GPU"]
threshold = {}
def InitThreshold(apps : List[App]):
    global threshold
    for app in apps:
        slo = app.slo
        if slo in threshold:
            continue
        rps_low = 0.01
        rps_high = 40
        function_cfger = NewFunctionCfg(get_config())
        while rps_high - rps_low > 0.001:
            rps = (rps_high + rps_low) / 2
            cfg = function_cfger.get_config(Apps([App("test", slo, rps)]))
            assert cfg is not None
            if cfg.instance.gpu is not None:
                rps_high = rps
            else:
                rps_low = rps
        threshold[slo] = rps_high

def GetThreshold(slo : float) -> float:
    return threshold[slo]

class HarmonyGroup():
    def __init__(self, apps: List[App], function_type = FUNCTION_TYPE, ratio : float = 1) -> None:
        self.apps = apps
        self.function_provider = NewFunctionCfg(get_config())
        self.config = None
        self.function_provision(function_type)
        self.ratio = ratio
        assert self.config is not None
        
    def group_cost(self) -> float:
        assert self.config is not None
        # return self.config.cost * self.ratio
        # base = self.config.cost * self.ratio
        base = self.config.cost  # CHANGE: start with raw config cost before ratio

        if self.config.instance.gpu is not None:
            total_rps = sum(app.rps for app in self.apps)
            # T = 1.0 / total_rps
            # s = 118 / 1000.0  # predicted GPU duration
            # # idle_penalty = 1.0 + 0.24 * max(0.0, (T - s) / s)
            # a = 2.1
            # i = 0.5
            # idle_penalty = (a * s + i * max(0.0, T - s)) / (a * s)
            # idle_penalty = max(1.0, 1)
            # # idle_penalty = 1.0 +  (8*(max(0.0, (T - s) / s)))/(34*s)
            # print('idle penalty:', idle_penalty, ' for group with apps:', self.apps, total_rps)
            # return base * idle_penalty
            fcost = self.function_provider.cost_cal  # CHANGE: reuse provisioner cost calculator for GPU idle billing
            # base = adjust_gpu_cost_for_idle(base, self.config.instance, total_rps, fcost)  # CHANGE: apply GPU idle billing model
        return base * self.ratio  # CHANGE: apply group ratio after idle adjustment (if any)
         
    def function_provision(self, function_type) -> Union[Cfg, None]:
        # print(function_type)
    
        self.config = self.function_provider.get_config(Apps(self.apps), function_type=function_type)
        # print(self.config)
        return self.config
    
    def merge(self, groups : List['HarmonyGroup']) -> bool:
        # print('Merging groups:')
        cost_before = self.group_cost() + sum([group.group_cost() for group in groups])
        ratio_total = self.ratio
        new_apps = copy.deepcopy(self.apps)
        for group in groups:
            new_apps += copy.deepcopy(group.apps)
            ratio_total += group.ratio
        # new_group = HarmonyGroup(new_apps, function_type=["GPU"], ratio=ratio_total)
        new_group = HarmonyGroup(new_apps, ratio=ratio_total)
        print('Cost before merge:', cost_before, ' Cost after merge:', new_group.group_cost())
        if new_group.group_cost() <= cost_before:
            self.apps = new_group.apps
            self.config = new_group.config
            self.ratio = new_group.ratio
            return True
        else:
            return False
    
    def __str__(self) -> str:
        return str(self.config)

    def __repr__(self) -> str:
        return "group_ratio: {}".format(round(self.ratio, 2)) + self.__str__()


def InitGroups(apps : List[App]) -> List[HarmonyGroup]:
    apps.sort(key=lambda app: app.slo)
    total_rps = sum([app.rps for app in apps])
    groups = []
    for app in apps:
        groups.append(HarmonyGroup([app], ratio=app.rps/total_rps))
        # print(f"Initialized group for app {app.name} with SLO {app.slo} and RPS {app.rps}: {groups[-1]}")
        # print(groups[-1].config)
    return groups

def NextContinusCPU(groups : List[HarmonyGroup], start_id = 0):
    n = len(groups)
    i = start_id

    # Skip GPU groups until we find first non-GPU
    while i < n and groups[i].config.instance.gpu is not None:
        i += 1

    if i >= n:
        return 0, 0

    start = i
    slo0 = groups[start].apps[0].slo
    thr = GetThreshold(slo0)

    total_rps = 0.0
    end = start

    # Grow segment while still non-GPU and under threshold
    while end < n and groups[end].config.instance.gpu is None:
        for a in groups[end].apps:
            total_rps += a.rps

        if total_rps > thr:
            break

        end += 1

    return start, end

def NextGPU(groups : List[HarmonyGroup], start_id = 0):
    """
    Find the next pair (or short segment) where merging involves GPU.
    Return (start, end) meaning attempt merging groups[start:end].
    Usually end = start+2.
    """
    n = len(groups)
    for i in range(start_id, n - 1):
        g1 = groups[i]
        g2 = groups[i+1]

        # GPU merge if at least one uses GPU
        if g1.config.instance.gpu is not None or g2.config.instance.gpu is not None:
            return i, i + 2

    return 0, 0

def total_cost(groups : List[HarmonyGroup]) -> float:
    return sum([group.group_cost() for group in groups])

def Harmony(apps : List[App], function_type = FUNCTION_TYPE) -> Union[List[HarmonyGroup], float]:
    print('Starting Harmony Grouping Algorithm')
    InitThreshold(apps)
    cost_change = []
    stage = [0]
    global threshold
    # print(threshold)
    t1 = time.time()
    groups = InitGroups(apps)
    # print('groups: ', groups)
    cost_change.append(total_cost(groups))
    # print(total_cost(groups))
    # print(groups)
    start_id = 0
    # while True:
    #     start_id, end_id = NextContinusCPU(groups, start_id)
    #     if end_id == 0:
    #         break
    #     elif end_id - start_id == 1:
    #         start_id = end_id
    #         continue
    #     else:
    #         is_merge = groups[start_id].merge(groups[start_id+1:end_id])
    #         if is_merge:
    #             groups = groups[:start_id+1] + groups[end_id:]
    #             start_id = start_id + 1
    #             cost_change.append(total_cost(groups))
    #             stage.append(1)
    #             # print(total_cost(groups))
    #             print('umad')
    #         else:
    #             start_id = end_id
    # ---- FIXED CPU MERGING BLOCK ----
    start_id = 0
    while True:
        start_id, end_id = NextContinusCPU(groups, start_id)
        if end_id == 0:
            break
        if end_id - start_id == 1:
            start_id = end_id
            continue

        # collect merged apps
        merged_apps = []
        merged_ratio = 0.0
        for g in groups[start_id:end_id]:
            merged_apps += copy.deepcopy(g.apps)
            merged_ratio += g.ratio

        # IMPORTANT: allow ALL platforms (CPU/GPU/partitioned)
        merged_group = HarmonyGroup(
            merged_apps,
            function_type=function_type,
            ratio=merged_ratio
        )

        cost_before = sum(g.group_cost() for g in groups[start_id:end_id])
        cost_after = merged_group.group_cost()

        if cost_after < cost_before:
            groups = groups[:start_id] + [merged_group] + groups[end_id:]
            cost_change.append(total_cost(groups))
            stage.append(1)
            print('cost before merge:', cost_before, ' Cost after merge:', cost_after)
            # Continue trying to merge around same index
        else:
            start_id = end_id

    start_id = 0
    while True:
        start_id, end_id = NextGPU(groups, start_id)
        if end_id == 0:
            break

        merged_apps = []
        merged_ratio = 0.0
        for g in groups[start_id:end_id]:
            merged_apps += copy.deepcopy(g.apps)
            merged_ratio += g.ratio

        merged_group = HarmonyGroup(
            merged_apps,
            function_type=function_type,
            ratio=merged_ratio
        )

        # Only accept GPU merge if merged config actually uses GPU
        if merged_group.config.instance.gpu is None:
            start_id += 1
            continue

        cost_before = sum(g.group_cost() for g in groups[start_id:end_id])
        cost_after = merged_group.group_cost()

        if cost_after < cost_before:
            groups = groups[:start_id] + [merged_group] + groups[end_id:]
            cost_change.append(total_cost(groups))
            stage.append(2)
            print('cost before merge:', cost_before, ' Cost after merge:', cost_after)
        else:
            start_id += 1

    t2 = time.time()
    print("Time cost: ", 1000 * (t2-t1))
    print(cost_change)
    print(stage)
    print("Final provisioning plan: ", groups, "total_cost: ",  total_cost(groups))
    return groups, total_cost(groups)

