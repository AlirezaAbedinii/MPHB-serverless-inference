import harmony
import csv
from harmony.serverless.request import Sample
import numpy as np
import harmony.core.util as util
from harmony.core.cost import FunctionCost
# -------- SETTINGS --------
master_url = "https://cmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
master_name = "cmaster"

workers = [
    "cfrom0to10worker1", "cfrom0to10worker2", "cfrom0to10worker3", "cfrom0to10worker4",
    "cfrom11to15worker1", "cfrom11to15worker2", "cfrom11to15worker3", "cfrom11to15worker4",
    "cfrom16to19worker1", "cfrom16to19worker2", "cfrom16to19worker3", "cfrom16to19worker4",
    "cfrom21to21worker1", "cfrom21to21worker2", "cfrom21to21worker3", "cfrom21to21worker4"
]
complex_workers = ["cfrom21to21worker1", "cfrom21to21worker2", "cfrom21to21worker3", "cfrom21to21worker4"]

WORKER_INSTANCES = [
    
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    util.Instance(1, 2, 0),
    

]




master_cpu_configs = [1, 1.5, 2]
master_mem_configs = [1024, 2048, 3072, 4096]  # in MB


batch_sizes = [1, 2]
cpu_configs = [0.35, 0.5, 0.75, 1, 1.5, 2]        # worker CPU values
memory = [512, 1024, 2048, 1024*3, 1024*4]
complex_mem = 1024
csv_file_path = 'CPU_profiling_v4.csv'
detailed_csv_path = 'CPU_profiling_per_request.csv'

client_num = 2     # threads sending requests in parallel
num_count = 2  # number of requests per client
cost_calculator = FunctionCost()


tcpu = 0.5
tmem = 512
for worker in workers:
                    if worker in complex_workers:
                        Sample.main(worker, cpu=tcpu, mem=max(complex_mem, tmem))
                    else:
                        Sample.main(worker, cpu=tcpu, mem=max(complex_mem, tmem))



# try:
#     with open(csv_file_path, 'r') as f:
#         pass
# except FileNotFoundError:
#     with open(csv_file_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['vCPU', 'Memory', 'Batch_size', 'Avg_latency(ms)', 'Min_latency(ms)',
#                          'Max_latency(ms)', 'Count', 'Total_cost($)'])


# # -------- FIX MASTER CONFIG --------
# # Sample.main(master_name, cpu=2, mem=4, gpu=1)

# # -------- START PROFILING --------
# for master_cpu in master_cpu_configs:
#     for master_mem in master_mem_configs:
#         # (OPTIONAL: Skip bad config combos)
#         if master_mem < master_cpu * 1024 or master_mem > master_cpu * 4 * 1024:
#             continue
        
#         # -------- Reconfigure Master Function --------
#         Sample.main(master_name, cpu=master_cpu, mem=master_mem)
#         for cpu in cpu_configs:
#             for mem in memory:
#                 if mem < cpu*1024 or mem>cpu*4*1024:
#                     continue
                
#                 # Reconfigure all workers
#                 for worker in workers:
#                     if worker in complex_workers:
#                         Sample.main(worker, cpu=cpu, mem=max(complex_mem, mem))
#                     else:
#                         Sample.main(worker, cpu=cpu, mem=max(complex_mem, mem))

#                 for b in batch_sizes:
#                     try:
#                         print(f"[RUNNING] Master CPU={master_cpu}, Master Mem={master_mem}, Worker CPU={cpu}, Mem={mem}, Batch={b}")
#                         profiler = harmony.Profiler(master_url, batch=b, client_num=client_num, num_count=num_count)
#                         latencies, worker_latencies = profiler.run()
#                     except Exception as e:
#                         print("Worker's Exception:", e)    
#                     avg_latency = np.mean(latencies)
#                     min_latency = np.min(latencies)
#                     max_latency = np.max(latencies)
#                     count = len(latencies)
                    
                    
                    
#                     # Check and create header if not exist
#                     try:
#                         with open(detailed_csv_path, 'r') as f:
#                             pass
#                     except FileNotFoundError:
#                         with open(detailed_csv_path, 'w', newline='') as f:
#                             writer = csv.writer(f)
#                             header = ['Master_vCPU', 'Master_Mem_MB', 'vCPU', 'Memory', 'Batch_size', 'Total_latency(ms)'] + [f'Worker_{i}_lat(ms)' for i in range(1, 17)]
#                             writer.writerow(header)

#                     # Save per-request results
#                     with open(detailed_csv_path, 'a', newline='') as f:
#                         writer = csv.writer(f)
#                         for req_idx in range(len(latencies)):
#                             row = [master_cpu, master_mem, cpu, mem, b, latencies[req_idx]] + worker_latencies[req_idx]
#                             writer.writerow(row)
                    
                    
#                     #########
#                     # Flatten worker latencies (num_requests x num_workers) => [lat,...]
#                     flat_worker_latencies = [lat for entry in worker_latencies for lat in entry]

#                     # Assign all worker instances same config for simplicity
#                     master_instance = util.Instance(2, 4, 0)  # Modify if needed
#                     worker_instances = [util.Instance(cpu, mem/1024, 0)] * len(flat_worker_latencies)

#                     # cost = cost_calculator.calculate_total_cost_with_workers(
#                     #     master_latency=avg_latency / 1000,  # ms → s
#                     #     worker_latencies=flat_worker_latencies,
#                     #     batch_size=b,
#                     #     master_instance=master_instance,
#                     #     worker_instances=worker_instances
#                     # )
                    
                    
#                     print(f"[PROFILE] cpu={cpu}, mem={mem}, batch={b}, avg_latency={avg_latency:.2f}ms")

#                     with open(csv_file_path, 'a', newline='') as f:
#                         writer = csv.writer(f)
#                         writer.writerow([cpu, mem, b, avg_latency, min_latency, max_latency])
