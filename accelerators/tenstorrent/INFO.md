# Tenstorrent Architecture & Model Topology Guide

## Hardware & Interconnect Topology
Tenstorrent Wormhole (and upcoming Blackhole) silicon operates on a **spatial 2D mesh dataflow architecture** where compute and memory are distributed across Tensix cores connected via an on-chip Network-on-Chip (NOC).

* **Core Compute Microarchitecture:** Each Tensix core contains 5 dedicated Baby RISC-V microcontrollers (BRISC for master scheduling, NCRISC for NOC data movement, and TRISC0/1/2 for matrix and vector SIMD math engines).
* **Memory Subsystem:** 
  * **L1 SRAM:** 1.5 MB dedicated per Tensix core (108 MB on n150, 192 MB on n300). Compute operations stream directly across L1 buffers over the NOC without continuous DRAM access.
  * **Off-Chip DRAM:** GDDR6 memory (288 GB/s on n150, 576 GB/s aggregate on dual-chip n300).
* **Clustering Interconnect:** On-chip Ethernet links and external QSFP-DD ports route data directly between Tensix meshes across multiple cards without host PCIe bottlenecking.

---

## Architectural Class Support

| Model Family Archetype | Architectural Primitives | Support Status | Scaling Considerations |
| :--- | :--- | :--- | :--- |
| **Dense Autoregressive Decoders** | RoPE, RMSNorm, SwiGLU, Multi-Head Attention (MHA), Grouped-Query Attention (GQA) | **Native / High Maturity** | Primary production target for TT-NN and `vllm-tt-plugin`. Full context lengths supported via chunked prefill. |
| **Sparse Mixture-of-Experts (MoE)** | Top-K gating dispatch, dynamic token routing, sparse FFN experts | **Emerging / Active Development** | Requires sub-device core slicing and low-latency NOC scatter/gather primitives. |
| **Encoder-Decoder Architectures** | Bidirectional self-attention, cross-attention projections | **Supported** | Validated on speech recognition topologies (e.g., Whisper architecture). |
| **Diffusion Transformers (DiT)** | Spatial attention, cross-attention conditioning, 2D/3D tensor reshaping | **Experimental** | Tracked under experimental multi-modal pipelines (`tt_dit`). |

---

## Parameter Footprint & Hardware Tiers

* **Single-Chip Tier (Wormhole n150 — 12 GB GDDR6, 72 Tensix cores):**
  * Target footprint: Models up to ~8B parameters operating in BFP8 or tightly packed BF16.
  * Concurrency: Best suited for low-to-moderate batch sizes ($BS \le 8$) or single-stream interactive decode.
* **Dual-Chip Workstation Tier (Wormhole n300 — 24 GB GDDR6, 144 Tensix cores):**
  * Target footprint: Comfortable deployment of 7B–14B parameter models with larger KV-cache reservations.
  * Concurrency: High-throughput batch scaling ($BS = 32$) with full context execution.
* **Multi-Device Cluster Tier (T3000 / Galaxy / QuietBox — 8 to 32+ Chips):**
  * Target footprint: 70B+ parameter architectures.
  * Parallelism strategy: Tensor Parallelism (TP) and Data Parallelism (DP) orchestrated over the native optical TT-Fabric collective communication library (CCL).

---

## Precision & Quantization Regimes
* **BFLOAT16:** Full numerical accuracy for weights and activations.
* **BFP8 / FP8 (E4M3, E5M2):** Hardware-native acceleration for dense matrix multiplications, achieving peak theoretical throughput (up to 466 TFLOPs on n300).
* **Low-Bit Int:** Experimental integer quantization formats for memory-constrained edge deployment.

---

## Active Kernel Gaps & Bounty Scope
* **PagedAttention Optimization:** Minimizing L1 SRAM spillover during high-concurrency decoding with non-power-of-two page sizes.
* **Fused Operators:** Fused RoPE + QKV projections and Fused SwiGLU activations to eliminate GDDR6 roundtrips.
* **Dynamic Dispatch:** Low-latency dynamic kernel launch on Tensix cores for dynamic-shape generation phases.
