import harmony
import csv
from harmony.serverless.request import Sample
import numpy as np

# -------- SETTINGS --------
master_url = "https://master-logpu-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
master_name = "master-logpu"

workers = [
    "from0to10worker1", "from0to10Worker2", "from0to10Worker3log", "from0to10Worker4log",
    "from11to15Worker1log", "from11to15Worker2log", "from11to15Worker3log", "from11to15Worker4log",
    "from16to19Worker1log", "from16to19Worker2log", "from16to19Worker3log", "from16to19Worker4log",
    "from21to21Worker1log", "from21to21Worker2log", "from21to21Worker3log", "from21to21Worker4log"
]

batch_sizes = [1, 2, 4, 8, 12, 16]
cpu_configs = [0.5, 0.75, 1, 1.5]        # worker CPU values
gpu_configs = [16]           # optional: try multiple GPU configs
csv_file_path = 'end_to_end_profiling.csv'

client_num = 2     # threads sending requests in parallel
num_count = 2      # number of requests per client


# -------- FIX MASTER CONFIG --------
Sample.main(master_name, cpu=2, mem=4, gpu=1)

# -------- START PROFILING --------
for gpu in gpu_configs:
    for cpu_percentage in cpu_configs:
        cpu = cpu_percentage
        # cpu = round(cpu/0.05)*0.05
        mem = min(int(cpu * 1024 * 4), 1024 * 32)
        print("mem 0:", mem)
        mem = max(mem, 1024 * gpu)
        print("mem 1:", mem)
        if mem % 64 != 0:
            mem = mem + 64 - mem % 64

        print("mem 2:", mem)
        
        # Reconfigure all workers
        for worker in workers:
            Sample.main(worker, cpu=cpu, mem=mem, gpu=gpu)

        for b in batch_sizes:
            # Optional warmup
            try:
                harmony.Profiler(master_url, batch=b, client_num=client_num, num_count=1).run()
                print("[SUCCESS]: masters profiling is done")
            except Exception as e:
                print("masters exception: ", e)
            # Real profiling
            try:
                profiler = harmony.Profiler(master_url, batch=b, client_num=client_num, num_count=num_count)
                latencies = profiler.run()
            except Exception as e:
                print("Worker's Exception:", e)    
            avg_latency = np.mean(latencies)

            print(f"[PROFILE] cpu={cpu}, mem={mem}, gpu={gpu}, batch={b}, avg_latency={avg_latency:.2f} ms")

            # Save to CSV
            with open(csv_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for latency in latencies:
                    writer.writerow([cpu, mem, gpu, b, latency])
