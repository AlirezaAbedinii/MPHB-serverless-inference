import random
import threading
import requests
import json
import harmony.core.cost as cscost
import harmony.core.util as util

class HttpFunction():
    def __init__(self, function_url, function_name=None) -> None:
        super().__init__()
        self.function_name = function_name
        self.function_url = function_url

    def invoke(self, params) -> float:
        for _ in range(5):
            response = requests.get(self.function_url, params=params)
            if response.status_code == 200:
                return float(json.loads(response.text)['infMs']) / 1000
        return -1.0

class ServerlessRequest():
    def __init__(self, slo, arrival_time: float = 0, latency: float = 0, wait_time: float = 0, app_name = "") -> None:
        self.arrival_time = arrival_time
        self.latency = latency
        self.wait_time = wait_time
        self.slo = slo
        self.app_name = app_name
        self.cost = 0


class ServerlessForProfile(threading.Thread):
    def __init__(self, num_count, function_url, params, que) -> None:
        super().__init__()
        self.num_count = num_count
        self.params = params
        self.function = HttpFunction(function_url)
        self.que = que

    def send_request(self):
        return self.function.invoke(self.params)

    def run(self):
        data = []
        for _ in range(self.num_count):
            batch_latency = self.send_request()
            for _ in range(5):
                if batch_latency > 0:
                    data.append(batch_latency)
                    break

        self.que.put(data)


class Serverless(threading.Thread):
    def __init__(self, requests, function_url, ins : util.Instance, lat_cal, que) -> None:
        super().__init__()
        self.requests = requests
        self.function = HttpFunction(function_url)
        self.que = que
        self.cost_cal = cscost.FunctionCost()
        self.ins = ins
        self.lat_cal = lat_cal

    def send_request(self):
        params = {
            "BATCH": str(len(self.requests)),
        }
        return self.function.invoke(params)

    def send_test(self):
        lat_min = self.lat_cal.lat_avg(self.ins, len(self.requests))
        lat_max = self.lat_cal.lat_max(self.ins, len(self.requests))
        return random.gauss(lat_min, (lat_max - lat_min) / 2.33)

    def run(self):
        if self.lat_cal is not None:
            batch_latency = self.send_test()
        else:
            batch_latency = self.send_request()
        if batch_latency < 0:
            batch_latency = 0
        cost = self.cost_cal.cost(batch_latency, len(self.requests), self.ins)
        for request in self.requests:
            request.latency = batch_latency
            request.cost = cost
        self.que.put(self.requests)
