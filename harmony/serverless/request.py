from typing import List
import numpy as np
import harmony
import harmony.serverless.serverless as serverless
import time
import pandas as pd
import harmony.core.util as util
import threading
import queue
from alibabacloud_fc20230330.client import Client as FC20230330Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_fc20230330 import models as fc20230330_models
from alibabacloud_tea_util import models as util_models
import random
import harmony.config as config


def read_traces(file_name: str):
    data = pd.read_csv(file_name, header=None)
    return data.values.flatten()

class Trace:
    def __init__(self, trace_path, num_count = 10, scale = 1.0, start_time = 0) -> None:
        self.trace = read_traces(trace_path) * scale
        assert len(self.trace) >= num_count + start_time, "trace length is less than num_count"
        self.trace = self.trace[start_time: start_time + num_count]

    def get_trace(self):
        return self.trace


class Application:
    def __init__(self, trace: Trace, slo : int, name) -> None:
        self.trace = trace
        self.slo = slo
        self.name = name

    def get_app(self, iter):
        rps = self.trace.get_trace()[iter]
        return util.App(self.name, self.slo, float(rps) / 60)
    
class Result:
    def __init__(self, cfg : util.Cfg, requests : List[serverless.ServerlessRequest]) -> None:
        self.cfg = cfg
        self.requests = requests


def generate_requests_helper(apps : List[util.App], batch_size: int, time_out: List[float], inter_arrival: List[float], duration_min: float, config: dict, ins : util.Instance, lat_cal, q : queue.Queue):
    total_rps = sum(inter_arrival)
    weights : List[float] = [app.rps / total_rps for app in apps]
    indexes = list(range(len(apps)))
    
    que = queue.Queue()
    threads = []
    time_stamp = 0.0
    total_delay = 0.0
    delay = 0.0
    count = 0
    queues = []
    t_out = []
    bs_out = []
    current_timeout = -1
    while True:
        index = random.choices(indexes, weights = weights, k = 1)[0]
        if count == 0:
            if len(time_out) == 1:
                current_timeout = time_out[0]
            else:
                current_timeout = time_out[index]
        else:
            if len(time_out) > 1:
                current_timeout = min(current_timeout, time_out[index] + total_delay)
        app = apps[index]
        queues.append(serverless.ServerlessRequest(app.slo, total_delay, app_name = app.name))
        count += 1
        delay = np.random.exponential(scale= 1.0/total_rps, size=None)

        if count < batch_size and total_delay + delay <= current_timeout:
            total_delay += delay
        elif count == batch_size:
            t_out.append(total_delay)
            bs_out.append(batch_size)
            for i in range(len(queues)):
                queues[i].wait_time = total_delay - queues[i].arrival_time
            t = serverless.Serverless(queues, config["function_url"], ins, lat_cal, que)
            t.start()
            threads.append(t)
            queues = []
            count = 0
            total_delay = 0.0
            time.sleep(delay)
        else:
            time.sleep(current_timeout - total_delay)
            delay_reminder = delay - (current_timeout - total_delay)
            total_delay = current_timeout
            t_out.append(total_delay)
            bs_out.append(count)
            for i in range(len(queues)):
                queues[i].wait_time = total_delay - queues[i].arrival_time
            t = serverless.Serverless(queues, config["function_url"], ins, lat_cal, que)
            t.start()
            threads.append(t)
            queues = []
            count = 0
            total_delay = 0.0
            time.sleep(delay_reminder)
        
        time_stamp += delay
        if time_stamp >= duration_min * 60:
            break

    for t in threads:
        t.join()
    request_result = []
    while que.empty() is False:
        request_result.extend(que.get())
    q.put(request_result)
    return request_result

def generate_requests(cfg : util.Cfg, duration_min: int, config: dict, lat_cal, q : queue.Queue):
    batch_size = cfg.batch_size
    time_out = cfg.timeout
    inter_arrival = cfg.rps
    
    if not isinstance(time_out, List):
        time_out = [time_out]
    if not isinstance(inter_arrival, List):
        inter_arrival = [inter_arrival]
    assert isinstance(time_out, List), "time_out must be list"
    assert isinstance(inter_arrival, List), "inter_arrival must be list"
    return generate_requests_helper(cfg.apps, batch_size, time_out, inter_arrival, duration_min, config, cfg.instance, lat_cal, q)


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
        access_key_id: str,
        access_key_secret: str,
    ) -> FC20230330Client:
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret
        )
        config.endpoint = f'1712221082273020.cn-shanghai.fc.aliyuncs.com'
        return FC20230330Client(config)

    @staticmethod
    def main(
        function_name,
        cpu,
        mem,
        gpu = None 
    ) -> bool:
        conf = config.get_config()
        access_key_id = conf["access_key_id"]
        access_key_secret = conf["access_key_secret"]
        client = Sample.create_client(access_key_id, access_key_secret)
        if gpu is None:
            update_function_input = fc20230330_models.UpdateFunctionInput(
                cpu=cpu,
                memory_size=mem
            )
        else:
            update_function_input = fc20230330_models.UpdateFunctionInput(
                cpu=cpu,
                memory_size=mem,
                gpu_config=fc20230330_models.GPUConfig(gpu * 1024, "fc.gpu.ampere.1")
            )
        update_function_request = fc20230330_models.UpdateFunctionRequest(
            body=update_function_input
        )
        runtime = util_models.RuntimeOptions()
        headers = {}
        succuess = True
        try:
            client.update_function_with_options(function_name, update_function_request, headers, runtime)
        except Exception as error:
            print(error)
            succuess = False
        return succuess

class Function:
    def __init__(self, function_name : str, function_url : str, cfg : util.Cfg, config : dict) -> None:
        self.function_name = function_name
        self.function_url = function_url
        self.cfg = cfg
        self.threads = []
        self.q = queue.Queue()
        self.config = config

        self.lat_cal = None
        # for test
        if self.function_url == "":
            model_name = config["model_name"]
            model_config = config["model_config"]
            if self.cfg.instance.gpu is None:
                self.lat_cal = harmony.CPULatency(model_config[model_name]['CPU'], model_name)
            else:
                self.lat_cal = harmony.GPULatency(model_config[model_name]['GPU']['A10'], model_name)

    def eq_cfg(self, cfg: util.Cfg):
        if self.cfg.instance == cfg.instance:
            return True
        return False
    
    def bind_cfg(self, cfg : util.Cfg):
        self.cfg = cfg
    
    def set_cfg(self, cfg : util.Cfg):
        if self.function_name != "":
            c = cfg.instance.cpu
            # if c > 8:
            #     c = round(c, 1)
            m = cfg.instance.mem
            if m  < c:
                m = c
            g = cfg.instance.gpu
            m = int(m * 1024)
            if m % 64 != 0:
                m = (m // 64) * 64 + 64
            if g is None:
                if Sample.main(self.function_name, c, m) is True:
                    print("cpu function change resource success")
                else:
                    print("cpu function change resource fail")
                    exit(0)
            else:
                if Sample.main(self.function_name, c, m, g) is True:
                    print("gpu function change resource success")
                else:
                    print("gpu function change resource fail")
                    exit(0)
        self.cfg = cfg
    
    def start(self, duration_min : int):
        config = {
            "function_url" : self.function_url
        }
        t = threading.Thread(target=generate_requests, args=(self.cfg, duration_min, config, self.lat_cal, self.q))
        t.start()
        self.threads.append(t)
    
    def finish(self):
        for t in self.threads:
            t.join()
        self.threads = []
        requests_result = self.q.get()
        return Result(self.cfg, requests_result)

# class Function:
#     def __init__(self, function_name: str, function_url: str, cfg: util.Cfg, config: dict) -> None:
#         self.function_name = function_name
#         self.function_url = function_url
#         self.cfg = cfg
#         self.threads = []
#         self.q = queue.Queue()
#         self.config = config

#         print(f"Simulated: Initialized function {function_name} with config {cfg}.")

#     def eq_cfg(self, cfg: util.Cfg):
#         # Simulate configuration equality check
#         return self.cfg == cfg

#     def bind_cfg(self, cfg: util.Cfg):
#         self.cfg = cfg
#         print(f"Simulated: Bound new config to function {self.function_name}.")

#     def set_cfg(self, cfg: util.Cfg):
#         # Simulate setting new configuration
#         print(f"Simulated: Setting config for {self.function_name} with CPU: {cfg.instance.cpu}, "
#               f"Memory: {cfg.instance.mem}, GPU: {cfg.instance.gpu}.")
#         self.cfg = cfg

#     def start(self, duration_min: int):
#         print(f"Simulated: Starting function {self.function_name} for {duration_min} minutes.")
#         # Simulate request processing
#         t = threading.Thread(target=self._simulate_processing, args=(duration_min,))
#         t.start()
#         self.threads.append(t)

#     def _simulate_processing(self, duration_min: int):
#         # Simulate some work by sleeping
#         time.sleep(duration_min * 60)
#         print(f"Simulated: Function {self.function_name} completed processing.")

#     def finish(self):
#         for t in self.threads:
#             t.join()
#         self.threads = []
#         print(f"Simulated: Function {self.function_name} finished.")
#         # Return a simulated result
#         return Result(self.cfg, [])
