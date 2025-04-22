import harmony
import csv
from harmony.serverless.request import Sample
import numpy as np

if __name__ == "__main__":
    function_url = "your function url here"
    function_name = "your function name here"
    batch = [1, 3, 5, 7, 9]
    cpu = [1, 4, 6, 8]
    gpu = 4
    csv_file_path = 'output.csv'

    client_num = 4
    num_count = 25
    for c in cpu:
        # update function
        m = min(int(c * 1024 * 4), int(1024 * 32))
        if m % 64 != 0:
            m = m + 64 - m % 64
        Sample.main(function_name, c, m, gpu)
        data = []
        avg = []
        for b in batch:
            profiler = harmony.Profiler(function_url, batch = b, client_num = client_num, num_count = 2)
            profiler.run()
            
            profiler = harmony.Profiler(function_url, batch = b, client_num = client_num, num_count = num_count)
            latency = profiler.run()
            print(c, b)
            data.append(latency)
            avg.append(np.mean(latency))
        print(avg)
        print(data)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)