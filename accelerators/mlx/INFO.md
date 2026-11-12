# Apple Silicon (MLX) Architecture & Model Topology Guide

## Hardware & Unified Memory Architecture
Apple Silicon (M2/M3/M4 Max and Ultra) is built on a **Unified Memory Architecture (UMA)** where the CPU, GPU, and Apple Neural Engine (ANE) access a shared pool of high-speed LPDDR5/LPDDR5X memory without PCIe data replication.

* **Memory Bandwidth & Capacity:** Up to **192 GB unified memory** at **800 GB/s** (M2 Ultra) or **128 GB** at **546 GB/s** (M4 Max).
* **Compute Engines:** 
  * Metal GPU (up to 76 cores delivering 27.2 TFLOPs FP32 / 54.4 TFLOPs FP16).
  * 16/32-core Apple Neural Engine dedicated to low-power matrix operations.

---

## Architectural Class Support

| Model Family Archetype | Architectural Primitives | Support Status | Scaling Considerations |
| :--- | :--- | :--- | :--- |
| **Dense Autoregressive Decoders** | GQA, RoPE, RMSNorm, SwiGLU | **Native / High Adoption** | First-class support via MLX and MLX-LM. Ideal for single-user interactive inference. |
| **Audio & Speech Recognition** | Encoder-Decoder, 1D convolution frontends | **Native** | Highly optimized (e.g., Whisper pipelines). |
| **Vision-Language & Diffusion** | Cross-attention, CLIP vision encoders, U-Net / DiT | **Native** | High memory pool allows local image generation without VRAM thrashing. |

---

## Parameter Footprint & Hardware Tiers
* **Desktop Workstation Tier (64 GB – 128 GB Unified RAM):**
  * Target footprint: 32B–70B parameter models quantized to 4-bit / 8-bit executing entirely in local memory.
* **High-End Studio Tier (192 GB Unified RAM):**
  * Target footprint: 70B models in native 8-bit/BF16, or large MoE models.

---

## Active Kernel Gaps & Bounty Scope
* **Server-Grade Continuous Batching:** MLX currently lacks production-grade dynamic continuous batching and multi-tenant scheduling.
* **Metal PagedAttention:** Paged KV-cache management optimized for Apple Silicon threadgroup memory.
* **Multi-Node RPC Interconnect:** Distributed inference coordination across networked Apple Silicon instances.
