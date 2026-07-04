import random
import threading
import requests
import json
import harmony.core.cost as cscost
import harmony.core.util as util
import time

# WORKER_INSTANCES = {
#     "from1to2Worker1": util.Instance(16, 8, 32),
    
#     "from11to11Worker1": util.Instance(4, 2, 8),
#     "from11to11Worker2": util.Instance(4, 2, 8),
#     "from11to11Worker3": util.Instance(4, 2, 8),

#     "from15to15Worker1": util.Instance(4, 2, 8),
#     "from15to15Worker2": util.Instance(4, 2, 8),
#     "from15to15Worker3": util.Instance(4, 2, 8),
    
#     "from16to16Worker1": util.Instance(4, 2, 8),
#     "from16to16Worker2": util.Instance(4, 2, 8),
#     "from16to16Worker3": util.Instance(4, 2, 8),
    
#     "from20to20Worker1": util.Instance(4, 2, 8),
#     "from20to20Worker2": util.Instance(4, 2, 8),
#     "from20to20Worker3": util.Instance(4, 2, 8),
    
#     "from21to21Worker1": util.Instance(16, 8, 32),

# }
WORKER_INSTANCES = [
    
    util.Instance(1, 2, None),
    util.Instance(1, 2, None),
    util.Instance(1, 4, None),
    util.Instance(1, 4, None),

    util.Instance(1, 2, None),
    util.Instance(1, 2, None),

]

class HttpFunction():
    def __init__(self, function_url, function_name=None) -> None:
        super().__init__()
        self.function_name = function_name
        self.function_url = function_url

    def invoke(self, params) -> float:
        for _ in range(5):
            response = requests.get(self.function_url, params=params)
            if response.status_code == 200:
                return float(json.loads(response.text)['total_time']) 
        return -1.0
    
    def full_invoke(self, params) -> dict:
        for _ in range(2):
            response = requests.get(self.function_url, params=params)
            if response.status_code == 200:
                return json.loads(response.text)
        return {"response": -1.0}

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
        time.sleep(0.)
        # return self.function.invoke(self.params)
        return self.function.full_invoke(self.params)

    def run(self):
        total_times = []
        all_worker_latencies = []
        for _ in range(self.num_count):
            result = self.send_request()
            if result and "total_time" in result and "latencies" in result:
                total_times.append(result["total_time"])
                all_worker_latencies.append(result["latencies"])  # list of lists

        self.que.put({
        "total_times": total_times,
        "worker_latencies": all_worker_latencies
    })


class Serverless(threading.Thread):
    def __init__(self, requests, function_url, ins : util.Instance, lat_cal, que, worker_instances=None) -> None:
        super().__init__()
        self.requests = requests
        self.function = HttpFunction(function_url)
        self.que = que
        self.cost_cal = cscost.FunctionCost()
        self.ins = ins
        self.lat_cal = lat_cal
        self.worker_instances = worker_instances
        
    def _adjust_gpu_idle_if_needed(self, cost):
        # CHANGE: apply GPU idle billing using observed arrival intervals
        if self.ins.gpu is None:
            return cost
        try:
            max_arrival = max((req.arrival_time for req in self.requests), default=0.0)
        except Exception:
            max_arrival = 0.0
        total_rps = len(self.requests) / max_arrival if max_arrival > 0 else None
        if total_rps is None or total_rps <= 0:
            return cost
        return cscost.adjust_gpu_cost_for_idle(cost, self.ins, total_rps, self.cost_cal)

    def send_request(self):
        params = {
            "batch": str(len(self.requests)),
        }
        return self.function.full_invoke(params)

    def send_test(self):
        lat_min = self.lat_cal.lat_avg(self.ins, len(self.requests))
        lat_max = self.lat_cal.lat_max(self.ins, len(self.requests))
        return random.gauss(lat_min, (lat_max - lat_min) / 2.33)

    def test_run(self):
        if self.lat_cal is not None:
            # Synthetic latency, no worker involvement
            batch_latency = self.send_test()
            worker_latencies = []
        else:
            # Get full response from master
            response_json = self.send_request()
            print(response_json)
            exit
            batch_latency = float(response_json["total_time"]) / 1000.0  # convert ms → sec
            worker_latencies = response_json.get("latencies", [])
            
            # batch_latency = self.send_request()
        if batch_latency < 0:
            batch_latency = 0
        # cost = self.cost_cal.cost(batch_latency, len(self.requests), self.ins)
        
        cost = self.cost_cal.calculate_total_cost_with_workers(
        master_latency=batch_latency,
        worker_latencies=worker_latencies,
        batch_size=len(self.requests),
        master_instance=self.ins,
        worker_instances=WORKER_INSTANCES,
        billed_second=True
        )
        # cost = self._adjust_gpu_idle_if_needed(cost)  # CHANGE: apply GPU idle adjustment when applicable

        
        
        for request in self.requests:
            request.latency = batch_latency
            request.cost = cost
        self.que.put(self.requests)

    
    def run(self):
        if self.lat_cal is not None:
            # Synthetic latency, no worker involvement
            batch_latency = self.send_test()
            worker_latencies = []
        else:
            # Real HTTP call; be robust to different response formats
            try:
                response_json = self.send_request()
            except Exception as e:
                print("[Serverless][WARN] send_request failed:", e)
                response_json = {}

            # Parse total latency with fallbacks:
            #  - partitioned: total_time (ms)
            #  - single: infMs (ms)
            try:
                if "total_time" in response_json:
                    batch_ms = float(response_json["total_time"])
                elif "infMs" in response_json:
                    batch_ms = float(response_json["infMs"])
                else:
                    raise KeyError("Missing total_time / infMs")
            except Exception as e:
                print("[Serverless][WARN] Failed to parse latency from response:", e,
                      "; response_json =", response_json)
                batch_ms = 0.0

            batch_latency = batch_ms   # it uses ms /convert ms → seconds later if needed

            # Worker latencies may be present only for partitioned
            # - partitioned master: 'latencies' or 'worker_latencies'
            # - single: none, so default to []
            worker_latencies = response_json.get(
                "latencies",
                response_json.get("worker_latencies", [])
            )

        if batch_latency < 0:
            batch_latency = 0

        # Cost calculation: same worker-aware formula as before.
        # For single functions, worker_latencies will be [], so cost
        # effectively reduces to master cost.
        cost = self.cost_cal.calculate_total_cost_with_workers(
            master_latency=batch_latency,
            worker_latencies=worker_latencies,
            batch_size=len(self.requests),
            master_instance=self.ins,
            worker_instances=self.worker_instances,
            billed_second=True,
        )
        # cost = self._adjust_gpu_idle_if_needed(cost)  # CHANGE: apply GPU idle adjustment when applicable

        batch_sec = batch_ms / 1000.0  # convert ms → sec
        # Attach latency, cost, and worker-level info to each request
        # batch_size = max(1, len(self.requests))
        # per_request_cost = cost / batch_size
        for request in self.requests:
            request.latency = batch_sec
            request.cost = cost
            try:
                # NEW: expose per-batch info on each request so experiments.py
                # can log worker latencies and total latency for partitioned runs.
                # request.worker_latencies = worker_latencies
                # request.total_latency = batch_latency
                request.total_latency_ms = batch_ms
                request.total_latency = batch_sec
                request.worker_latencies_ms = worker_latencies
                request.worker_latencies = [w / 1000.0 for w in worker_latencies]
            except Exception:
                # If ServerlessRequest is extended in the future, we don't want
                # attribute errors to break the run.
                pass

        self.que.put(self.requests)