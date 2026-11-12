# Basalt Kernel Registry

The open-source kernel repository for high-performance AI inference on alternative and open silicon architectures.

---

## Overview

Modern open-weights models often encounter missing operators, inefficient memory layouts, or sub-optimal kernel fusions when targeting non-CUDA accelerators. 

The Basalt Registry is a hardware-first monorepo that houses optimized compute kernels (C++, TT-Metalium, HIP, Metal) designed to bridge these gaps and maximize hardware efficiency across alternative silicon.

---

## Hardware Backends

Each accelerator target maintains its own architecture specifications, memory constraints, and ecosystem details:

* [**Tenstorrent**](accelerators/tenstorrent/INFO.md) — TT-Metalium and TT-NN kernels for Wormhole Tensix dataflow architecture.
* [**AMD Instinct**](accelerators/amd/INFO.md) — ROCm / HIP and AOTriton kernels for CDNA architectures (MI300X).
* [**Apple Silicon**](accelerators/mlx/INFO.md) — Metal Shading Language and MLX implementations leveraging unified memory.
* [**Groq**](accelerators/groq/INFO.md) — Deterministic graph compilation profiles and partition recipes for Tensor Streaming Processors.
* [**NVIDIA**](accelerators/nvidia/INFO.md) — Reference CUDA and Triton baselines used for numerical parity and latency benchmarking.

---

## Kernel Developer Bounties

Basalt partners with open-source systems developers to expand operator support on alternative hardware:

1. **Targeted Bounties:** Bounties are established when model profiling reveals missing operators, CPU fallbacks, or memory bottlenecks on supported hardware.
2. **Verification Standards:** Every contribution must pass automated numerical correctness checks against PyTorch/CUDA golden references (`torch.allclose`) and demonstrate verified latency improvements on target hardware.
3. **Royalty Model:** In addition to upfront cash bounties, approved kernels deployed to the Basalt managed inference platform are tracked for ongoing margin-sharing based on inference volume.

---

## Repository Structure

```text
registry/
├── accelerators/
│   ├── tenstorrent/      # TT-Metalium kernels & tests for Wormhole
│   ├── amd/              # ROCm / HIP implementations for Instinct
│   ├── mlx/              # Metal / MLX kernels for Apple Silicon
│   ├── groq/             # Graph compilation profiles for Groq LPU
│   └── nvidia/           # Golden numerical reference implementations
├── CONTRIBUTING.md       # Contribution guidelines, testing, and PR workflow
└── README.md
```

---

## Getting Started

To explore developer guides, environment setup, and verification criteria, see [CONTRIBUTING.md](CONTRIBUTING.md).
