# Notice

This project contains code derived from the following open-source research projects:

---

**Gillis: Serving Large Neural Networks in Serverless Functions with Automatic Model Partitioning**
Minchen Yu, Zhifeng Jiang, Hok Chun Ng, Wei Wang, Ruichuan Chen, and Bo Li. ICDCS 2021.
Original repository: https://github.com/MincYu/gillis-open-source (MIT License)

Used in `partitioning/`. Extensions made here:

- Cost-aware (ROI-based) partition template selection in the latency-optimal (DP) policy
- AWS Lambda deployment tooling updates for exported partition workspaces
- Monolithic-CPU cost/latency baseline registration for ROI scoring
- Cost-aware filtering of candidate partition templates
- Exporters for additional model architectures (Wide-ResNet variants and others)

---

**HarmonyBatch: Batching multi-SLO DNN Inference with Heterogeneous Serverless Functions**
Jiabin Chen, Fei Xu, Yikun Gu, Li Chen, Fangming Liu, and Zhi Zhou. IWQoS 2024.
Original repository: https://github.com/icloud-ecnu/HarmonyBatch (MIT License)

Used in `provisioning/`. Extensions made here:

- Partitioned (multi-function pipeline) execution as a third provisioning mode alongside CPU and GPU
- Unified pipeline latency model for partitioned execution
- Partitioned cost model (master + worker functions)
- Partition-aware group consolidation in the provisioning algorithm
- Adaptation and deployment on Alibaba Cloud Function Compute, with profiling and experiment tooling

---

All rights in the original Gillis and HarmonyBatch code remain with their respective authors.
