# NVIDIA Hopper / Blackwell Architecture (Baseline Reference)

## Hardware Architecture Overview
NVIDIA Hopper (H100 SXM5) and Blackwell (B200) serve as the **industry reference baseline** for numerical parity, compiler maturity, and performance verification across the Basalt Registry.

* **Hopper (H100 SXM5):** 132 SMs, 80 GB HBM3 @ 3.35 TB/s, 1,979 TFLOPs dense FP8 (3,958 TFLOPs with 2:4 sparsity), 989.5 TFLOPs dense BF16/FP16, NVLink 4 (900 GB/s).
* **Blackwell (B200):** Dual-die package, 192 GB HBM3e @ 8.0 TB/s, 4,500 TFLOPs dense FP8 (9,000 TFLOPs sparse), 2,250 TFLOPs dense BF16/FP16, NVLink 5 (1.8 TB/s).

---

## Architectural Class Coverage
NVIDIA CUDA/TensorRT-LLM provides Day-0, 100% reference coverage across every modern machine learning archetype:
* Dense Decoders (MHA, GQA, MLA)
* Sparse Mixture-of-Experts (MoE)
* State-Space Models (Mamba, SSM)
* Vision-Language & Multimodal Models (VLMs)
* Diffusion Transformers (DiT)

---

## Role in the Basalt Registry
1. **The Mathematical Golden Truth:** Every custom kernel submitted for alternative silicon (Tenstorrent, AMD, Apple) is verified against NVIDIA CUDA/PyTorch reference outputs to guarantee numerical precision (`torch.allclose()`).
2. **Efficiency Benchmark:** Bounties evaluate performance deltas against Hopper/Blackwell hardware utilization rates (MFU) and capital cost per token generated.
