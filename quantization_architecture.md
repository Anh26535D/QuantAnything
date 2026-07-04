# YOLOv8-cls Manual Quantization Architecture

This document describes the manual quantization architecture implemented for the YOLOv8 nano classification model. The system converts the PyTorch model to Keras 3.0 via ONNX, builds a custom computational graph representation of the model, and performs post-training static quantization (PTQ) layer by layer.

---

## 1. Model Compilation & Conversion Flow

The following diagram illustrates the lifecycle of the model from the initial PyTorch weights down to the custom quantized container:

```mermaid
graph TD
    A[PyTorch Model: yolov8n-cls.pt] -->|Export| B[ONNX Model: yolov8n-cls.onnx]
    B -->|Sanitize Names: Replace '/' and ':' with '_'| C[Sanitized ONNX Model]
    C -->|onnx2keras: Keras 3.0 Adapter| D[Keras Functional Model]
    D -->|Instantiate| E[QuantContainer Graph]
    
    style A fill:#EE4B2B,stroke:#333,stroke-width:2px,color:#fff
    style B fill:#005C9E,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#00A86B,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#D00000,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#8A2BE2,stroke:#333,stroke-width:2px,color:#fff
```

---

## 2. QuantContainer: Custom Computational Graph

The `QuantContainer` fully controls the calibration and inference workflows by representing the model as a Directed Acyclic Graph (DAG) and running execution in topological order:

```mermaid
graph TB
    subgraph QuantContainer Execution Pass
        A[Input NumPy Array] -->|Feed| B[Input Tensor Cache]
        B -->|Retrieve Inputs| C[Layer 1: InputQuantize]
        C -->|Compute & Store| D[Tensor Cache]
        
        D -->|Retrieve Inputs| E[Layer 2: Conv2DQuantize]
        E -->|Compute & Store| F[Tensor Cache]
        
        D -->|Retrieve Inputs| G[Layer 3: ShapingQuantize split]
        G -->|Compute & Store| H[Tensor Cache]
        
        F & H -->|Retrieve Inputs| I[Layer 4: AddQuantize]
        I -->|Compute & Store| J[Tensor Cache]
        
        J -->|Topological Pass...| K[...]
        K -->|Final Output| L[Model Outputs]
    end

    style B fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
    style D fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
    style F fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
    style H fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
    style J fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
```

During execution:
1. `QuantContainer` initializes a `tensor_values` cache mapping each original `KerasTensor`'s python ID (`id(tensor)`) to its computed NumPy value.
2. It loops through `model.layers` in order (guaranteed to be topologically sorted).
3. For each layer, it looks up the computed values of the layer's input tensors (`layer.input`) from the cache, runs the quantized layer wrapper, and writes the output back to the cache under the layer's output tensors (`layer.output`).

---

## 3. Simulated (Fake) Quantization Data Path

Each `QuantLayer` (e.g. `Conv2DQuantize`) simulates INT8 quantization effects on the hardware using a Quantize/Dequantize (Q/DQ) pipeline:

```mermaid
graph TD
    %% Input Path
    In[Float32 Input X] -->|Input Q/DQ| InQ[Quantize / Dequantize]
    InQ -->|Scale: s_x, ZP: zp_x| InFake[Fake-Quantized Input X_hat]
    
    %% Weight Path
    W[Float32 Weights W] -->|Weight Q/DQ| WQ[Quantize / Dequantize]
    WQ -->|Scale: s_w, ZP: 0 | WFake[Fake-Quantized Weights W_hat]
    
    %% Bias Path
    B[Float32 Bias b] -->|Bias Q/DQ| BQ[Quantize / Dequantize]
    BQ -->|Scale: s_x * s_w, ZP: 0| BFake[Fake-Quantized Bias b_hat]
    
    %% Computation
    InFake & WFake & BFake -->|Execute Original Layer| Op[Conv2D / Dense Computation]
    Op -->|Float32 Output Y| OutQ[Output Q/DQ]
    
    %% Output Path
    OutQ -->|Scale: s_y, ZP: zp_y| OutFake[Fake-Quantized Output Y_hat]
    
    style InQ fill:#ffcc00,stroke:#333
    style WQ fill:#ffcc00,stroke:#333
    style BQ fill:#ffcc00,stroke:#333
    style OutQ fill:#ffcc00,stroke:#333
```

### Quantization Formulas
For any tensor $V$ (inputs, weights, outputs):
- **Quantization (Float to Int)**:
  $$q = \text{clip}\left(\text{round}\left(\frac{V}{\text{scale}}\right) + \text{zp},\ q_{min},\ q_{max}\right)$$
- **Dequantization (Int to Float)**:
  $$\hat{V} = (q - \text{zp}) \times \text{scale}$$
- **Fake Quantization**:
  $$\hat{V} = \text{dequantize}(\text{quantize}(V))$$

### Calibration Schemes
- **Weights**: Symmetric signed quantization (zero point is 0). Default is per-channel for convolutional and linear layers.
- **Activations (Inputs/Outputs)**: Asymmetric unsigned/signed quantization based on the specified `quant_type`. Default is signed `int8` (range $[-128, 127]$).
- **Biases**: Quantized to 32-bit integers with a scale factor of $\text{scale}_{input} \times \text{scale}_{weight}$, ensuring they can be added directly to the raw integer products.
