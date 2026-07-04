import harmony
import csv
from harmony.serverless.request import Sample
import numpy as np
import harmony.core.util as util
from harmony.core.cost import FunctionCost
# -------- SETTINGS --------
master_url = "https://wmaster-xxxxxxxxxx.us-east-1.fcapp.run/invoke"
master_name = "w5master"

workers = [
    # "from0to17Worker1", "from0to17Worker2",
    #  "from18to29Worker1", "from18to29Worker2",
    "from30to33Worker1", "from30to33Worker2"
    
    #  "wcfrom18to29Worker1", "wcfrom18to29Worker2"
    # "wcfrom0to17Worker1", "wcfrom0to17Worker2",
    # "wcfrom18to29Worker1", "wcfrom18to29Worker2"
]
# complex_workers = ["cfrom21to21worker1", "cfrom21to21worker2", "cfrom21to21worker3", "cfrom21to21worker4"]

WORKER_INSTANCES = [
    
    util.Instance(0.75, 2, 0),
    util.Instance(0.75, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    # util.Instance(1, 2, 0),
    

]


cpu = 0.75
mem = 3072
gpu = 0

# Single
# Sample.main(single_worker, cpu = cpu, mem = mem)

# MP
for worker in workers:
    Sample.main(worker, cpu=cpu, mem=mem)

                    

