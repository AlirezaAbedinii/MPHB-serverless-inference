import json
import os
import time

import harmony

import argparse
from harmony.algorithm.algorithm import Compose

from harmony.core.util import App, Cfgs, timeout_rps_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='harmony')
    parser.add_argument('--config', type=str,
                        default='conf', help='config path')

    args = parser.parse_args()

    with open(os.path.join(args.config, "config.json"), 'r') as f:
        config = json.load(f)
    with open(os.path.join(config['cfg_path'], 'model.json'), 'r') as f:
        config["model_config"] = json.load(f)

    apps_list = [App("app1", 0.5, 5), App("app2", 0.8, 10), App("app3", 1.0, 15)]\

    algorithm = "Harmony"
    function_cfger = harmony.NewFunctionCfg(algorithm, config)
    total_rps = sum(app.rps for app in apps_list)
    comps = Compose(apps_list, algorithm=algorithm)
    cfgs = Cfgs()
    for com in comps.get_apps():
        tmp = Cfgs()
        for apps in com:
            cfg = function_cfger.get_config(apps)
            if cfg is None:
                break
            ratio = apps.get_rps() / total_rps
            cfg.set_ratio(ratio)
            tmp.add(cfg) 
        tmp.set_com(com)
        cfgs.update(tmp)
    print(cfgs)