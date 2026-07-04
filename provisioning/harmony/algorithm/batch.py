from harmony.algorithm.algorithm import NewFunctionCfg
from harmony.config import get_config
from typing import List, Union
from harmony.core.util import App, Apps, Cfg
import time
import numpy as np

class BatchGroup():
    def __init__(self, apps : List[App], function_type = ["CPU"], ratio : float = 1) -> None:
        self.apps = apps
        self.function_provider = NewFunctionCfg(get_config())
        self.config = None
        self.function_provision(function_type)
        assert self.config is not None
        self.ratio = ratio
    
    def function_provision(self, function_type) -> Union[Cfg, None]:
        self.config = self.function_provider.get_config(Apps(self.apps), function_type=function_type, need_one = True)
        return self.config
    
    def group_cost(self) -> float:
        if self.config is None:
            return np.inf
        return self.config.cost * self.ratio

    def __str__(self) -> str:
        return str(self.config)
    
    def __repr__(self) -> str:
        return "group_ratio: {}".format(round(self.ratio, 2)) + self.__str__()


def InitGroups(apps : List[App]) -> List[BatchGroup]:
    total_rps = sum([app.rps for app in apps])
    groups = []
    for app in apps:
        groups.append(BatchGroup([app], ratio=app.rps / total_rps))
    return groups

def total_cost(groups : List[BatchGroup]) -> float:
    return sum([group.group_cost() for group in groups])

def Batch(apps : List[App]):
    t1 = time.time()
    groups = InitGroups(apps)
    t2 = time.time()
    print("Time cost: ", 1000 * (t2-t1))
    return groups, total_cost(groups)

