from harmony.serverless.serverless import ServerlessForProfile
import queue

class Profiler():
    def __init__(self, function_url, batch = 1, client_num = 1, num_count = 10) -> None:
        self.function_url = function_url
        self.num_count = num_count
        params = {
            "BATCH": str(batch)
        }
        self.que = queue.Queue()
        self.client = [
            ServerlessForProfile(self.num_count, self.function_url, params, self.que) 
            for _ in range(client_num)
            ]
    
    def run(self):
        for client in self.client:
            client.start()
        for client in self.client:
            client.join()
        
        latency = []
        while not self.que.empty():
            latency.extend(self.que.get())

        return latency