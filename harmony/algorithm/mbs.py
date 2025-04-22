
from typing import List
import time
from numpy import Inf
from harmony.core.util import App, Apps
from harmony.algorithm.algorithm import NewFunctionCfg
from harmony.config import get_config
import copy

class MBSGroup():
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
    
    def function_provision(self, function_type : List[str] = ["CPU", "GPU"]) -> None:
        self.config = self.function_provider.get_config(Apps(self.apps), function_type=function_type)

    def __str__(self) -> str:
        return str(self.config)
    
    def __repr__(self) -> str:
        return "group_ratio: {}".format(round(self.ratio, 2)) + self.__str__()

def Partition(apps : List[App], i : int) -> List[MBSGroup]:
    groups : List[MBSGroup] = []
    total_rps = sum([app.rps for app in apps])
    copy_apps = copy.deepcopy(apps)

    sub_rps = total_rps / i
    app_list : List[App] = []
    cur_rps = 0

    for j in range(len(copy_apps)):
        app = copy_apps[j]
        while cur_rps + app.rps >= sub_rps + 0.01:
            tmp = sub_rps - cur_rps
            app1 = App(app.name, app.slo, tmp)
            app_list.append(app1)
            app2 = App(app.name, app.slo, app.rps - tmp)
            groups.append(MBSGroup(app_list,  ratio = 1 / i))
            cur_rps = 0
            app_list = []
            app = app2
        else:
            cur_rps += app.rps
            app_list.append(app)
        
        if j == len(copy_apps) - 1:
            if cur_rps > 0:
                groups.append(MBSGroup(app_list, ratio = 1 / i))
            break
    return groups

def total_cost(groups : List[MBSGroup]) -> float:
    return sum([group.group_cost() for group in groups])

def MBS(apps : List[App]):
    g_cost = Inf
    g_groups = None
    t1 = time.time()
    for i in range(1, len(apps)+1):
        groups = Partition(apps, i)
        current_cost = 0 
        for group in groups:
            current_cost += group.group_cost()
        if current_cost < g_cost:
            g_cost = current_cost
            g_groups = groups
    assert g_groups is not None
    t2 = time.time()
    print("Time cost: ", 1000 * (t2-t1))
    return g_groups, total_cost(g_groups)
    
