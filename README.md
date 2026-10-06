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
│   ├── graph_passes.py          # Pattern fusions (LayerNorm, GELU/SiLU LUT, MatMul+bias)
│   ├── float_ops.py             # NumPy reference implementation of ONNX operators
│   ├── quant_container.py       # Calibration + float / fake-quant / integer execution
│   ├── quant_layers.py          # Quantized layers (Conv, Gemm, BN, LUT activations, ...)
│   ├── utils.py                 # Scale / zero-point / fixed-point multiplier maths
│   ├── native.py                # ctypes loader for the C++ kernels + NumPy fallbacks
│   ├── kernels/
│   │   ├── qa_kernels.cpp       # C++ kernels (GEMM, conv, requantize, ...)
│   │   └── CMakeLists.txt       # Optional ahead-of-time build
│   ├── reid.py                  # ReID dataset parsing, embeddings, mAP / CMC
│   ├── evaluator.py             # Calibration and classification evaluation pipeline
│   └── tests/                   # pytest suite (kernels, ops vs onnxruntime, layers, e2e)
├── download_vit.py               # Get a static-batch ViT ONNX (HF download or timm export)
├── reid_zero_shot.py             # Zero-shot ReID benchmark: float vs quantized backbone
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
   - **Graph fusions** (`graph_passes.py`, on by default): PyTorch-style
     decomposed LayerNorm becomes one `LayerNormalization`; element-wise
     subgraphs with one dynamic input (SiLU, decomposed GELU, `x * s + t`)
     become a single lookup table; `MatMul + Add(bias)` joins the integer
     accumulator; static `Shape -> Gather -> Expand` chains are folded.
     Exact algebraic folds: `Conv + BatchNorm` -> `Conv`, LayerNorm
     gain/shift into the following linear layers, the attention `1/sqrt(d)`
     scale into the query weights, and `Gather(LayerNorm(x))` ->
     `LayerNorm(Gather(x))`. Tidy-ups: no-op `Cast` removal, fused
     element-wise subgraphs that are a known activation are renamed
     (`Gelu`, `Silu`, `Mish`, `HardSwish`, ...), chains of constant
     `Mul`/`Add` are merged, a constant `Mul` after a `Conv`/`MatMul`/`Gemm`
     is folded into its weights, and the attention scale is folded into the
     QKV projection whether it sits on the logits or on `q` and `k^T`
     separately (timm). On a timm `vit_tiny_patch16_224` export the graph
     goes from 483 nodes to 259 (no `Cast`, `Shape`, `Sqrt`, `Div`, `Erf` or
     `Mul` left, 12 named `Gelu`). They keep the float function (checked against
     ONNX Runtime) but are **not** enough for int8 ViTs, see below.
   - **Anything else** (e.g. `Softmax`, `LayerNormalization`, activation x
     activation `MatMul`) falls back to dequantize -> NumPy float op ->
     quantize, so those layers are *simulated*, not integer-only.
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
`Clip`, `Erf`, `Gelu`, `Pow`, `Expand`, `LayerNormalization`, `Softmax`,
`LogSoftmax`, `Cast`, `Shape`, `Gather`, ... Unsupported
operators raise a descriptive `UnsupportedOp` error.

---

### Vision Transformers

`tests/test_vit.py` pushes a small ViT (patch-embed Conv, cls token, MHSA with
activation x activation `MatMul`, GELU MLP, LayerNorm) through the pipeline.
Float parity with `onnxruntime` is ~1e-6. On a random-weight 4-block ViT
(dim 64) logit cosine similarity vs float was ~0.99998 for **int16** but only
~0.92 for **int8** (naive per-tensor min/max): the residual stream has
outliers. Outlier-aware calibration (Phase 2) is the next lever.

### Export the optimized model

The graph passes normally run in memory when a model is loaded. To get them
as a file:
```bash
uv run python optimize_onnx.py vit_tiny.onnx -o vit_tiny_opt.onnx --check
```
The result is a standard ONNX model (opset >= 20 when `Gelu` appears) that
runs in any runtime; `--check` compares it with the original on ONNX Runtime.
On a timm `vit_tiny_patch16_224` export: 483 -> 307 nodes (the 48 bias `Add`
nodes are fused again when the file is loaded back into the quantizer).

### Getting a real ViT without PyTorch

`vit_from_npz.py` downloads Google's official ImageNet-21k -> 1k ViT checkpoints
(`ti16`, `s16`, `b16`, from Google Cloud Storage) and writes a static-batch
ONNX (`LayerNormalization` + `Gelu`, opset 20). It needs neither PyTorch,
timm nor Hugging Face. `ti16` classifies the PyTorch-hub dog photo as
*Samoyed* (0.803, identical to ONNX Runtime; max logit diff 1.6e-5).
On that real model **int16 keeps the exact top-3 (cos 0.9997) while naive
min/max int8 collapses (cos 0.13)**: the late residual stream has massive
activations (abs-max ~570 vs 99.9th percentile ~15), so one int8 step is
~4.5. Mixed precision or outlier-aware calibration is needed for int8 ViTs.

Measured on that model (fake-quant, int8 weights, cosine of the logits vs
float; graph folds applied): everything int8 **0.17**; residual `Add` outputs
int16 **0.65**; + LayerNorm int16 **0.63**; + all `MatMul` outputs int16
**0.92**; all non-Gemm activations int16 (i.e. W8A16) **0.9976**. The loss is
spread over the residual stream, the linear / attention outputs and
softmax / GELU, so W8A16 is the realistic target, not per-tensor A8.

### Zero-shot ReID benchmark

`reid_zero_shot.py` uses the embedding of a pretrained (not ReID-trained)
backbone with cosine distance on a Market-1501 style dataset (`query/` and
`bounding_box_test/`, files `<pid>_c<cam>s<seq>_*.jpg`) and reports mAP /
Rank-1/5/10 for the float model (ONNX Runtime) and for each quantization
setting, plus the feature cosine similarity to float.

```bash
uv run python vit_from_npz.py --variant ti16 --embedding --output vit_ti16_emb.onnx
uv run python reid_zero_shot.py --onnx vit_ti16_emb.onnx \
    --root Market-1501-v15.09.15 --max-ids 100 --distractors 1000 \
    --quant-types int8 int16 --modes integer
```

A classification ONNX also works (the head is cut automatically). The
simulator is slow on the full ~19.7k-image gallery, so subsample with
`--max-ids` / `--distractors` first. Use `--norm` to match the checkpoint
(`half`, `imagenet`, `clip`).

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
