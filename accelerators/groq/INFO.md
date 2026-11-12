# Groq TSP / LPU Architecture & Model Topology Guide

## Hardware Architecture: The Deterministic LPU
Groq processors (GroqChip / GroqCard) are built on a **Tensor Streaming Processor (TSP)** architecture that replaces hardware dynamic scheduling with deterministic compiler-scheduled execution.

* **Zero External DRAM:** The chip features **no HBM or GDDR**. It relies entirely on **230 MB of on-chip SRAM** delivering **80 TB/s on-die memory bandwidth**.
* **Deterministic Execution:** No branch predictors, hardware caches, or out-of-order execution logic. Instruction streams and memory hops are computed ahead of time.
* **Compute Performance:** 750 TOPS INT8, 188 TFLOPs FP16 at 275W TDP.

---

## Architectural Class Support

| Model Family Archetype | Architectural Primitives | Support Status | Scaling Considerations |
| :--- | :--- | :--- | :--- |
| **Dense Autoregressive Decoders** | Standard MHA, GQA, RoPE | **Production Ready** | Highest token-generation speeds in the industry due to SRAM bandwidth. |
| **Sparse Mixture-of-Experts** | Dynamic routing | **Production Ready (Cluster-scale)** | Requires large multi-rack topologies to store all expert weights across distributed SRAM. |
| **Dynamic-Shape Models** | Variable sequence lengths, dynamic runtime branching | **Constrained** | Dynamic tensor shapes require compilation padding or multi-graph ahead-of-time planning. |

---

## Parameter Footprint & Topology Requirements
Because each chip contains only 230 MB of SRAM, **no modern LLM fits on a single Groq chip**:
* **Multi-Chip Nodes (GroqNode — 8 chips):** Stored footprint $\approx 1.8\text{ GB}$ of SRAM. Fits small embeddings or sub-1B models.
* **Rack Clusters (GroqRack — 64+ to 256+ chips):** Required to distribute 8B to 70B parameter models across the deterministic RealScale optical mesh.

---

## Developer Interface & Registry Role
* Groq uses a proprietary ISA and compiler toolchain. Developers do not write raw assembly or C++ device kernels.
* The Basalt Registry tracks **model graph decomposition recipes, compiler partition strategies, and multi-chip placement configs** for Groq deployments.
