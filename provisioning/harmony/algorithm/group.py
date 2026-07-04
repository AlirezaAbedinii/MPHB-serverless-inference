from typing import List
from harmony.algorithm.batch import Batch
from harmony.algorithm.harmony import Harmony
from harmony.algorithm.mbs import MBS
from harmony.config import get_config
from harmony.core.util import App

def Algorithm(apps : List[App]):
    config = get_config()
    algorithm = config["algorithm"]
    if algorithm == "BATCH":
        return Batch(apps)
    elif algorithm == "MBS":
        return MBS(apps)
    elif algorithm == "Harmony":
        return Harmony(apps)
    else:
        return Harmony(apps)