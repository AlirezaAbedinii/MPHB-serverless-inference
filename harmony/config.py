import os
import json

def init_global_config():
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "../")
    set_config(path)

def set_config(path):
    global g_config
    with open(os.path.join(path, "conf/config.json"), 'r') as f:
        g_config = json.load(f)
    with open(os.path.join(path, g_config['cfg_path'], 'model.json'), 'r') as f:
        g_config["model_config"] = json.load(f)
    print(g_config["algorithm"])

def get_config():
    global g_config
    return g_config