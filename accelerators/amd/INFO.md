# AMD Instinct Architecture & Model Topology Guide

## Hardware & Interconnect Topology
AMD Instinct (CDNA3 architecture, MI300X/MI325X) utilizes modular 3D vertical die stacking and 2.5D packaging to deliver large unified memory pools on a single accelerator socket.

* **Compute Die Configuration:** 8 Accelerated Compute Dies (XCDs) stacked on 4 I/O Dies (IODs), delivering 304 Compute Units (19,456 Stream Processors) with 256 MB of AMD Infinity Cache.
* **Memory Subsystem:** **192 GB of HBM3** across an 8,192-bit interface providing **5.3 TB/s peak bandwidth**.
* **Interconnect:** 8 Infinity Fabric links per OAM module providing 896 GB/s bidirectional GPU-to-GPU bandwidth within an 8-GPU node.

---

## Architectural Class Support

| Model Family Archetype | Architectural Primitives | Support Status | Scaling Considerations |
| :--- | :--- | :--- | :--- |
| **Dense Autoregressive Decoders** | MHA, GQA, RoPE, RMSNorm, Linear Attention | **Production Ready** | Full compatibility with upstream vLLM and HuggingFace pipelines. |
| **Sparse Mixture-of-Experts (MoE)** | High-parameter sparse expert routing | **Production Ready** | The 192 GB HBM3 pool allows running large MoE architectures with zero offloading. |
| **Long-Context Reasoning Models** | Deep attention layers, multi-latent attention (MLA) | **Production Ready** | Supported natively via AOTriton and FlashAttention-2 HIP ports. |

---

## Parameter Footprint & Hardware Tiers

* **Single-Accelerator Tier (MI300X — 192 GB HBM3):**
  * Target footprint: Models up to **70B parameters unquantized (BF16)** fit completely on a single card without tensor parallel communication overhead.
  * Quantized footprint: Models up to ~400B parameters in FP8.
* **8-GPU Node Tier (1.5 TB HBM3 per server):**
  * Target footprint: Full 405B+ dense architectures and massive MoE models serving with high concurrency.

---

## Active Kernel Gaps & Bounty Scope
* **FlashDecoding Concurrency:** Micro-tuning FlashDecoding kernels for high-concurrency decode phases.
* **FP8 GEMM Parity:** Optimizing custom ROCm/HIP FP8 matrix multiplication kernels to match native CUTLASS performance on Hopper.
* **Compiler Stability:** Mitigating performance regressions across minor ROCm compiler releases.
