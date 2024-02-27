'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 18:02:40
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-12-14 06:35:14
FilePath: /CSInference/csinference/__init__.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
from harmony.config import init_global_config
init_global_config()

from harmony.algorithm.algorithm import NewFunctionCfg
from harmony.algorithm.group import Algorithm
from harmony.core.latency import GPULatency,CPULatency
from harmony.core.util import Instance, App, Apps
from harmony.core.cost import FunctionCost
from harmony.serverless.profiler import Profiler

__version__ = "0.0.0"

__all__ = [
    "NewFunctionCfg",
    "GPULatency",
    "CPULatency",
    "Instance",
    "FunctionCost",
    "App",
    "Apps",
    "Profiler",
    "Algorithm"
]

