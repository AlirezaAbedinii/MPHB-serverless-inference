import time
from harmony.algorithm.algorithm import NewFunctionCfg
from harmony.config import get_config
from typing import List, Union
import copy

from harmony.core.util import App, Apps, Cfg

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
    def __init__(self, apps: List[App], function_type = ["CPU", "GPU"], ratio : float = 1) -> None:
        self.apps = apps
        self.function_provider = NewFunctionCfg(get_config())
        self.config = None
        self.function_provision(function_type)
        self.ratio = ratio
        assert self.config is not None
        
    def group_cost(self) -> float:
        assert self.config is not None
        return self.config.cost * self.ratio
    
    def function_provision(self, function_type) -> Union[Cfg, None]:
        self.config = self.function_provider.get_config(Apps(self.apps), function_type=function_type)
        return self.config
    
    def merge(self, groups : List['HarmonyGroup']) -> bool:
        cost_before = self.group_cost() + sum([group.group_cost() for group in groups])
        ratio_total = self.ratio
        new_apps = copy.deepcopy(self.apps)
        for group in groups:
            new_apps += copy.deepcopy(group.apps)
            ratio_total += group.ratio
        new_group = HarmonyGroup(new_apps, function_type=["GPU"], ratio=ratio_total)
        if new_group.group_cost() < cost_before:
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
    return groups

def NextContinusCPU(groups : List[HarmonyGroup], group_id = 0):
    for i in range(group_id, len(groups)):
            if groups[i].config.instance.gpu is None:
                start_id = i
                end_id = i
                threshold = GetThreshold(groups[i].apps[0].slo)
                total_rpps = 0
                for g in groups[i:]:
                    assert g.config is not None
                    if g.config.instance.gpu is None:
                        end_id += 1
                        total_rpps += sum([app.rps for app in g.apps])
                        if total_rpps > threshold:
                            return start_id, end_id
                    else:
                        return start_id, end_id
    return 0, 0

def NextGPU(groups : List[HarmonyGroup], group_id = 0):
    for i in range(group_id, len(groups)):
        if groups[i].config.instance.gpu is None and i + 1 < len(groups) and groups[i+1].config.instance.gpu is not None:
            return i, i+2
        elif groups[i].config.instance.gpu is not None and i + 1 < len(groups):
            return i, i+2
    return 0, 0

def total_cost(groups : List[HarmonyGroup]) -> float:
    return sum([group.group_cost() for group in groups])

def Harmony(apps : List[App]):
    InitThreshold(apps)
    cost_change = []
    stage = [0]
    global threshold
    # print(threshold)
    t1 = time.time()
    groups = InitGroups(apps)
    cost_change.append(total_cost(groups))
    # print(total_cost(groups))
    # print(groups)
    start_id = 0
    while True:
        start_id, end_id = NextContinusCPU(groups, start_id)
        if end_id == 0:
            break
        elif start_id - end_id == 1:
            start_id = end_id
            continue
        else:
            is_merge = groups[start_id].merge(groups[start_id+1:end_id])
            if is_merge:
                groups = groups[:start_id+1] + groups[end_id:]
                start_id = start_id + 1
                cost_change.append(total_cost(groups))
                stage.append(1)
                # print(total_cost(groups))
                # print(groups)
            else:
                start_id = end_id
    start_id = 0 
    while True:
        start_id, end_id = NextGPU(groups, start_id)
        if end_id == 0:
            break
        else:
            is_merge = groups[start_id].merge(groups[start_id+1:end_id])
            if is_merge:
                groups = groups[:start_id+1] + groups[end_id:]
                start_id = start_id
                cost_change.append(total_cost(groups))
                stage.append(2)
                # print(groups)
            else:
                start_id = start_id + 1
    t2 = time.time()
    print("Time cost: ", 1000 * (t2-t1))
    print(cost_change)
    print(stage)
    return groups, total_cost(groups)

