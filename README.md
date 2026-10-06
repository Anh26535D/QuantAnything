# QuantAnything: ONNX Post-Training Quantization (PTQ)

A framework-agnostic implementation of static post-training quantization for
ONNX models, demonstrated on the **YOLOv8 Nano classification model**. It
depends only on the open standards **ONNX** (model format) and **NumPy**
(tensors). There is no Keras / TensorFlow / PyTorch at run time. The hot
loops (convolution, matrix multiplication, requantization) are implemented as
C++ kernels. Supports fake quantization (Q/DQ simulation) and integer-only
static inference with 32/64-bit integer accumulation.

---

## 1. Project Directory Structure

```
├── quantization/                 # Core quantization package
│   ├── __init__.py              # Public API
│   ├── onnx_graph.py            # ONNX -> NumPy graph (constant folding, toposort)
│   ├── float_ops.py             # NumPy reference implementation of ONNX operators
│   ├── quant_container.py       # Calibration + float / fake-quant / integer execution
│   ├── quant_layers.py          # Quantized layers (Conv, Gemm, BN, LUT activations, ...)
│   ├── utils.py                 # Scale / zero-point / fixed-point multiplier maths
│   ├── native.py                # ctypes loader for the C++ kernels + NumPy fallbacks
│   ├── kernels/
│   │   ├── qa_kernels.cpp       # C++ kernels (GEMM, conv, requantize, ...)
│   │   └── CMakeLists.txt       # Optional ahead-of-time build
│   ├── evaluator.py             # Calibration and classification evaluation pipeline
│   └── tests/                   # pytest suite (kernels, ops vs onnxruntime, layers, e2e)
├── download_and_convert.py       # Export YOLOv8n-cls to ONNX (needs `ultralytics`)
├── quantization_test.ipynb       # Interactive validation notebook
├── quantization_architecture.md  # Detailed technical architecture design
└── pyproject.toml               # Dependencies (managed with `uv`)
```

---

## 2. Current Quantization Approach

1. **Model format**: the ONNX file *is* the model. `OnnxGraph` loads it into
   NumPy arrays, folds constants, removes `Identity`/`Dropout` and
   topologically sorts the nodes. Layout stays ONNX-native (NCHW, OIHW).
2. **Execution engine (`QuantContainer`)**: walks the nodes in order and keeps
   a `{tensor name: array}` environment. Quantization parameters live per
   *tensor* (`container.qparams`), so every layer inherits its input scale
   from its producer and publishes its own output scale.
3. **Integer-only inference math**:
   - **Conv / Gemm / MatMul**: weights are quantized per output channel (or
     per tensor), activations asymmetric/symmetric. Zero-point-centred
     operands go through the C++ GEMM/conv kernels (`int16 x int16 -> int32`
     when provably overflow-free, `int32 x int32 -> int64` otherwise).
   - **Fixed-point requantization**: scale differences are applied as
     $$V_{\text{out}} = \text{mbqm}(V_{\text{accum}} + \text{bias}_{\text{int}}, M_0, \text{shift}) + Z_{\text{out}}$$
     with a 32-bit multiplier $M_0$ and a bit-shift, computed with 128-bit
     intermediates (no overflow even for 16-bit models).
   - **Activations** (`Sigmoid`, `Relu`, `Tanh`, `HardSwish`, `Clip`, ...) are
     evaluated through an integer lookup table built at calibration time.
     SiLU is exported by ONNX as `Sigmoid` + `Mul` and is handled as such.
   - **Scale-preserving ops** (`Reshape`, `Flatten`, `Split`, `Slice`,
     `Transpose`, `Pad`, `MaxPool`, ...) copy their input parameters
     (`Pad` pads with the zero point, `MaxPool` with `qmin`).
   - **Scale matching**: `Add`, `Sub`, `Mul`, `Concat` and `GlobalAveragePool`
     precompute multipliers that align the input scales with the output scale.
   - **Anything else** (e.g. `Softmax`) falls back to
     dequantize -> NumPy float op -> quantize.
4. **Persistence**: `container.save("q.json")` stores the per-tensor
   scales / zero points; `QuantContainer.load(onnx_model, "q.json")` rebuilds a
   calibrated container with bit-identical results.

### C++ kernels

`quantization/kernels/qa_kernels.cpp` exposes a plain C ABI that is loaded with
`ctypes`: no Python headers, pybind11 or build backend are needed. On first
import it is compiled with the system C++ compiler (`g++`/`clang++`,
OpenMP and `-march=native` when available) and cached. If no compiler exists,
or `QA_DISABLE_NATIVE=1` is set, equivalent pure NumPy fallbacks are used (the
test-suite checks that both backends are **bit-exact**).

| Variable | Meaning |
| :--- | :--- |
| `QA_DISABLE_NATIVE=1` | force the NumPy fallbacks |
| `QA_LIB=/path/libqa_kernels.so` | load a pre-built library (e.g. from CMake) |
| `QA_MARCH=x86-64-v2` | `-march=` for the JIT build (default `native`; empty disables) |
| `CXX` | C++ compiler to use |
| `OMP_NUM_THREADS` | thread count of the kernels |

---

## 3. Usage

```python
import numpy as np
from quantization import QuantContainer

quant = QuantContainer("yolov8n-cls.onnx", quant_type="int8")
quant.calibrate(calibration_batches)            # NCHW float32 arrays

probs_fake = quant.quantized_infer(x, mode="quantize")  # Q/DQ simulation
probs_int = quant.quantized_infer(x, mode="integer")    # integer-only math
report = quant.compare_layers(x, mode="integer")        # per-layer error
quant.save("yolov8n-cls.qparams.json")
```

Supported ONNX operators: `Conv` (2-D, grouped/depthwise), `Gemm`, `MatMul`,
`BatchNormalization`, `Add`, `Sub`, `Mul`, `Div`, `Concat`, `Split`, `Slice`,
`Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Transpose`, `Pad`,
`MaxPool`, `AveragePool`, `GlobalAveragePool`, `GlobalMaxPool`, `ReduceMean`,
`Sigmoid`, `Relu`, `LeakyRelu`, `Tanh`, `HardSigmoid`, `HardSwish`, `Elu`,
`Clip`, `Softmax`, `LogSoftmax`, `Cast`, `Shape`, `Gather`, ... Unsupported
operators raise a descriptive `UnsupportedOp` error.

---

## 4. Timeline for Improvements

| Phase | Milestone | Expected Deliverable | Timeline |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Evaluation & Analysis** | - Run full ImageNet-val subsampled benchmarks for INT8 vs INT16.<br>- Profile memory usage and latency. | Weeks 1 - 2 |
| **Phase 2** | **Dynamic Activation Quantization** | - Support asymmetric activation boundaries (dynamic bounds) for SiLU.<br>- Implement outlier-aware quantization (like clipping top 0.01% activation ranges). | Weeks 3 - 4 |
| **Phase 3** | **Per-Channel Activation / Weight Scaling**| - Tune zero points specifically for non-linear residual pathways. | Weeks 5 - 6 |
| **Phase 4** | **Mixed-Precision Optimization** | - Selectively keep critical activations or residual additions in INT16 while keeping standard Conv2D layers in INT8.<br>- Deploy a command-line conversion utility. | Weeks 7 - 8 |
| **Phase 5** | **Edge Deployment Simulation** | - Emit QDQ / QLinear ONNX models for other runtimes.<br>- SIMD micro-kernels (AVX2/NEON) for the C++ GEMM. | Weeks 9 - 10 |

---

## 5. How to Run

### Install Dependencies
Ensure you have `uv` installed, then:
```bash
uv sync                    # runtime + dev dependencies
uv sync --extra export     # + ultralytics, only to export YOLOv8 to ONNX
```

### Export the Demo Model
```bash
uv run python download_and_convert.py
```

### Run Unit Tests
```bash
uv run python -m pytest quantization/tests/
```
The tests build small ONNX graphs in-process, compare the NumPy operators with
`onnxruntime`, the C++ kernels with the NumPy fallbacks, and the integer path
with the fake-quantization path.

### Run Evaluation
```bash
uv run python -m quantization.evaluator --onnx yolov8n-cls.onnx \
    --dataset dataset/val --quant-type int8 --mode integer
uv run jupyter notebook quantization_test.ipynb
```

### Optional: ahead-of-time kernel build
```bash
cmake -S quantization/kernels -B build && cmake --build build
QA_LIB=$PWD/build/libqa_kernels.so uv run python -m pytest quantization/tests/
```
