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
    
    # Load partitioned modelling
    # 3️⃣ OPTIONAL: load partitioned modeling (config2.json)
    # TODO: config2 for wrn50-5 and config3 for wrn50-4
    try:
        part_path = os.path.join(path, g_config["cfg_path"], "config2.json")
        if os.path.exists(part_path):
            with open(part_path, "r") as f:
                part_cfg = json.load(f)
            # store under a clear key
            g_config["partitioned_config"] = part_cfg
            # optionally link its latency_models for easy access
            g_config["partitioned_latency_models"] = part_cfg.get("latency_models", {})
            print("[config] Loaded partitioned model data from config2.json.")
        else:
            g_config["partitioned_config"] = {}
            g_config["partitioned_latency_models"] = {}
            print("[config] No config2.json found; skipping partitioned models.")

        print(f"[config] Algorithm = {g_config.get('algorithm','?')}, Model = {g_config.get('model_name','?')}")
    except Exception as e:
        print(f"[config] Error loading partitioned modeling: {e}")
        g_config["partitioned_config"] = {}
        g_config["partitioned_latency_models"] = {}

def get_config():
    global g_config
    return g_config