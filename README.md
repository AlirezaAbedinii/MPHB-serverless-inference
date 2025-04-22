# HarmonyBatch

_HarmonyBatch_, a cost-efficient resource provisioning framework designed to achieve predictable performance for multi-SLO DNN inference with heterogeneous serverless functions

## Prototype of HarmonyBatch
_HarmonyBatch_  comprises mainly three modules: a model profiler, a performance predictor, and a function provisioner. The model profiler profiles the model with both CPU and GPU functions to acquire model-specific and hardware-specific coefficients. The performance predictor can estimate the inference cost using our performance model. It then guides the function provisioner to identify an appropriate group strategy and function provisioning plans that guarantee the SLOs of all applications. The configurations provided by _HarmonyBatch_ include both batch-related configurations and resource-related configurations. The batch-related configurations will be sent to the batch manager to control the request queue and resource-related configurations will be sent to the serverless platform to update the function. 

![](images/framework.png)


## Model the Optimization Problem
Given a set of inference applications 
$` \mathcal{W} = \{ w_{1}, w_{2}, ...., w_{n} \} `$ sharing the same DNN model with the inference latency SLO $`\mathcal{L} = \{ l^{w_1}, l^{w_2}, ..., l^{w_n} \}`$ and request arrival rates $`\mathcal{R} = \{ r^{w_1}, r^{w_2}, ..., r^{w_n} \}`$. We categorize the application set $\mathcal{W}$ into several groups $`\mathcal{G} = \{ \mathcal{X}_{1}, \mathcal{X}_{2}, ..., \mathcal{X}_{m} \}`$. Each group $`\mathcal{X} = \{w_{j}, w_{j+1}, ...\}`$ is provisioned with an appropriate CPU or GPU function, with the aim of meeting application SLO requirements while minimizing the total monetary cost. 
Based on our DNN inference performance and cost models, we can formulate the optimization problem as
```math
\begin{align}
    \min_{\mathcal{G}, \mathcal{F}, \mathcal{B}, \mathcal{T}}  & Cost = \sum_{\mathcal{X} \in \mathcal{G}} \eta^{\mathcal{X}} \cdot C^{\mathcal{X}} \\
    s.t. \ \ \ \ 
    &  M^{\mathcal{X}} \leq m^{\mathcal{X}}, \  \forall \ \mathcal{X} \in \mathcal{G} \\
    &  t^w + L_{max}^{t} \leq l^w, \  \forall \ w \in \mathcal{X}, \ \mathcal{X} \in \mathcal{G}
\end{align}
```
where $\eta^{\mathcal{X}}$ is the proportion of the request arrival rate of group $\mathcal{X}$ to the total request arrival rate. $C^{\mathcal{X}}$ is the average monetary cost of group $\mathcal{X}$.
Each group $\mathcal{X}$ is configured with a function of resource $f^{\mathcal{X}} \in \mathcal{F}$ (i.e., a tuple of vCPU cores $c^{\mathcal{X}}$ and GPU memory $m^{\mathcal{X}}$, $f^{\mathcal{X}} = [c^{\mathcal{X}}, m^{\mathcal{X}}]$).

## Getting Started

### Installation
```shell
git clone https://github.com/HarmonyBatch/HarmonyBatch
pip install requirements.txt
```

### Run the Prototype System

#### Test the Algorithm

Set up the model name (i.g., VGG19) and algorithm name (i.g., HarmonyBatch) within the configuration file in `conf/config.json`.
Set the application SLOs and arrival rates in `main.py` and run the algorithm: 
```python
cd HarmonyBatch
python3 main.py
```
After runing the code, you will get the provisioning plan as follows for example:
```
Provisioning plan:
The configurations of the group 0 is:
cpu:            1.60
batch:          1
rps:            5
timeout:        0.0
cost:           4.350e-05
slo:            0.5
----
The configurations of the group 1 is:
...
```

#### Run the Experimens with Trace
Set up the model name (i.g., VGG19), algorithm name (i.g., HarmonyBatch), application num and the SLOs of each applications within the configuration file in `conf/config.json`. Running the code for simulation:
```python
cd HarmonyBatch
python3 experiments.py
```
After running the code, you will get the `result.csv` file, which including the data every minute as follows :
```
SLO violations, cost, inference count, predict cost
```
If you want to evaluate the _HarmonyBatch_ with real function invocation, you need to deploy your DNN model on serverless function platform and replace the function url and function name in `experiments.py` in advance.
