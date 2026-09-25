# GPU, unified memory and other accelerators

noulo runs every local model through ONNX Runtime, which can offload work to an
accelerator through **execution providers**:

| Device | Hardware | How to get it |
|---|---|---|
| `coreml` | Apple Silicon GPU and Neural Engine (unified memory) | Built into the standard `onnxruntime` wheel on macOS |
| `cuda` | NVIDIA GPUs | `pip uninstall onnxruntime && pip install onnxruntime-gpu` (NVIDIA driver + CUDA 12) |
| `directml` | Any DirectX 12 GPU on Windows (NVIDIA, AMD, Intel) | `pip uninstall onnxruntime && pip install onnxruntime-directml` |
| `rocm` | AMD GPUs on Linux | `onnxruntime-rocm` build for your ROCm version |
| `cpu` | Any CPU | Always available |

The GPU packages *replace* `onnxruntime` (they provide the same Python module), so install
exactly one of them.

## Choosing the device

```bash
noulo config set device auto     # default (see below)
noulo config set device gpu      # always use the best available accelerator
noulo config set device coreml   # a specific one; fails with an install hint if it's missing
noulo config set device cpu      # never use an accelerator
noulo restart
```

`/config` in the interactive session does the same. `noulo status`, `/api/v1/info`
(`"device"`), the session banner and the frontend's Engine panel all show where the active
model actually runs.

Safety nets:

- The CPU provider is always listed after the accelerator, so any operation the accelerator
  can't run (common for INT8/INT4 kernels) runs on the CPU instead of failing.
- If the accelerator fails to initialise, noulo logs a warning and falls back to the CPU;
  it doesn't refuse to start.
- CoreML compiles the model the first time it's loaded (tens of seconds for larger models).
  The compiled model is cached in `data/run/accelerator-cache/`, so later starts are fast.

## What `auto` does (and why)

`auto` only uses an accelerator where it's expected to be faster than the CPU:

| Machine | Model type | `auto` uses |
|---|---|---|
| NVIDIA (CUDA), AMD (ROCm), Windows GPU (DirectML) | Unquantised FP32/FP16 models (the "larger" tier) | **the GPU** |
| NVIDIA / AMD / Windows GPU | Quantised INT8/INT4 models (basic tier, LLMs) | CPU: most INT8/INT4 kernels have no GPU implementation, so the work would bounce between GPU and CPU |
| Apple Silicon (CoreML) | Any | **CPU**: measured slower on CoreML for every model type (table below) |
| No accelerator | Any | CPU |

`gpu` or an explicit device (`coreml`, `cuda`, ...) overrides this and always uses the
accelerator.

**Why CoreML loses here:** noulo's inputs have variable sequence lengths. CoreML only
compiles parts of the graph with fixed shapes, so each request hops between the Neural
Engine or GPU and the CPU, and the copying costs more than the acceleration saves. That's
what the measurements below show on an M1 Pro, which has an integrated GPU and Neural Engine
sharing unified memory, but no discrete GPU.

**Not measured in this build:** CUDA, ROCm and DirectML weren't available on the development
machine, so the `auto` rule for them follows ONNX Runtime's kernel coverage rather than a
measurement. Check it on your hardware: `noulo benchmark --model <id> --device cpu` versus
`--device gpu`.

## Measured on an Apple M1 Pro

Same test split, same machine, back to back (`noulo benchmark --device cpu|coreml`):

| Model | Device | P50 | P95 | Peak RAM | Start-up | Choice | Noul | Score MAE |
|---|---|---|---|---|---|---|---|---|
| `nli-deberta-v3-xsmall-int8` (INT8) | **CPU** | **7.8 ms** | **32 ms** | **363 MiB** | **0.7 s** | 67.5% | 91.0% | 0.205 |
| | CoreML | 764 ms | 887 ms | 1531 MiB | 14.4 s | 65.0% | 88.0% | 0.203 |
| `zeroshot-deberta-v3-base-fp32` (FP32) | **CPU** | **28 ms** | **129 ms** | **1059 MiB** | **1.9 s** | 85.0% | 91.0% | 0.115 |
| | CoreML | 328 ms | 528 ms | 2479 MiB | 9.4 s (cached compile) | 85.0% | 91.0% | 0.115 |
| `lfm2-1.2b-q4` (INT4 LLM) | **CPU** | **856 ms** | **1050 ms** | **911 MiB** | **3.1 s** | 87.5% | 85.0% | 0.221 |
| | CoreML | stopped: not finished after 20 min (the whole CPU run takes ~3 min) | | | | | | |

On this machine the CPU wins every time, by 12× to 100×. That's why `auto` doesn't use CoreML.
The FP32 model gives identical answers on CoreML; the INT8 model's answers shift slightly
because its quantised kernels run differently.

Measure your own hardware:

```bash
noulo benchmark --model zeroshot-deberta-v3-base-fp32 --device cpu
noulo benchmark --model zeroshot-deberta-v3-base-fp32 --device gpu
```

Benchmarks default to `--device cpu`, so the numbers shown in the model menus stay comparable
across machines.
