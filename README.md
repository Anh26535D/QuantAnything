# YOLOv8-cls Post-Training Quantization (PTQ) Framework

This repository contains a manual, framework-agnostic implementation of static post-training quantization (PTQ) for the **YOLOv8 Nano Classification Model** using Keras 3.0. The framework supports fake-quantization (Q/DQ simulation) and integer-only static inference simulation with 32-bit integer accumulation.

---

## 1. Project Directory Structure

```
├── quantization/               # Core quantization package
│   ├── __init__.py            # Module entry-point
│   ├── quant_container.py     # topological execution DAG graph manager
│   ├── quant_layers.py        # Custom layer quantizers (Conv2D, BN, Dense, GAP, etc.)
│   ├── utils.py               # Low-level scaling, zp, multiplier & bit-shift utils
│   ├── evaluator.py           # Calibration and classification evaluation pipeline
│   └── tests/                 # Unit test suite (pytest)
│       ├── conftest.py        # Test search path configuration
│       ├── test_container.py  # Graph container test cases
│       ├── test_conv2d.py     # Convolution quantizer tests
│       ├── test_input.py      # Scaling & zero point math tests
│       ├── test_integer.py    # Integer-only pipeline integration tests
│       └── test_other_layers.py # Pooling, Add, Concat, and BN quantizer tests
├── quantization_architecture.md# Detailed technical architecture design
├── quantization_test.ipynb     # Interactive Jupyter notebook for validation
├── download_and_convert.py     # Script to convert ONNX models to Keras
└── README.md                  # Project overview (this file)
```

---

## 2. Current Quantization Approach

1. **Model Conversion**: Re-targets weights from PyTorch (`.pt`) to Keras 3.0 via an intermediate ONNX graph representation to ensure platform portability.
2. **Topological Execution Engine (`QuantContainer`)**: Walks the model layers in verified topological order, storing intermediate tensor states within a NumPy-based cache mapping values using native Keras tensor IDs.
3. **Integer-Only Inference Math**:
   - **Quantized Convolution & Dense Matrix Multiplications**: Weights are quantized per-tensor or per-channel, while inputs are quantized asymmetric/symmetric. 
   - **Quantized Rescaling Multiplier**: Scale differences between inputs, weights, and outputs are handled via fixed-point approximations using a 32-bit multiplier ($M_0$) and integer bit-shifts ($\text{shift}$):
     $$V_{\text{out}} = \text{multiply\_by\_quantized\_multiplier}(V_{\text{accum}} + \text{bias}_{\text{int}}, M_0, \text{shift}) + Z_{\text{out}}$$
   - **Intermediate Scaling Inheritance**: Features passed through shape transformations (e.g., `Reshape`, `ZeroPadding2D`, `Cropping2D`, lambda splits) inherit their parent scale and zero-point parameters to eliminate calculation drift.
   - **Scale Matching**: Operations like `Add`, `Concatenate`, and `GlobalAveragePooling2D` precompute rescaling factors to safely align different scales prior to calculation.
4. **Quantized Serialization (Save & Load)**:
   - **`save(model_path, metadata_path)`**: Serializes the layer-by-layer quantization scales, zero-points, and rescaling multipliers/shifts to a JSON file (using `NumpyEncoder` to handle numpy scalars cleanly). Quantizes Keras model weights to integer values and exports the model to a `.keras` file.
   - **`load(model_path, metadata_path)`**: Reconstructs the `QuantContainer` graph wrapper from the loaded `.keras` file and restores the quantization states on the graph to enable out-of-the-box integer static inference.

---

## 3. Timeline for Improvements

| Phase | Milestone | Expected Deliverable | Timeline |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Evaluation & Analysis** | - Run full ImageNet-val subsampled benchmarks for INT8 vs INT16.<br>- Profile memory usage and latency. | Weeks 1 - 2 |
| **Phase 2** | **Dynamic Activation Quantization** | - Support asymmetric activation boundaries (dynamic bounds) for SiLU.<br>- Implement outlier-aware quantization (like clipping top 0.01% activation ranges). | Weeks 3 - 4 |
| **Phase 3** | **Per-Channel Activation / Weight Scaling**| - Support per-channel kernel weight quantization in depthwise and dense operations.<br>- Tune zero points specifically for non-linear residual pathways. | Weeks 5 - 6 |
| **Phase 4** | **Mixed-Precision Optimization** | - Selectively keep critical activations or residual additions in INT16 while keeping standard Conv2D layers in INT8.<br>- Deploy a command-line conversion utility. | Weeks 7 - 8 |
| **Phase 5** | **Edge Deployment Simulation** | - Add support for TFLite flatbuffer compatibility checks.<br>- Validate execution layout conversion (NCHW to NHWC). | Weeks 9 - 10 |

---

## 4. How to Run

### Install Dependencies
Ensure you have `uv` installed, then set up the environment and run:
```bash
uv sync
```

### Run Unit Tests
Verify the quantization operators and integer pipeline logic using `pytest`:
```bash
uv run python -m pytest quantization/tests/
```

### Run Evaluation Notebook
Launch the Jupyter instance or execute the main verification steps directly:
```bash
uv run jupyter notebook quantization_test.ipynb
```
