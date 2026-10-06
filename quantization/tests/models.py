"""Builders for small ONNX test models (no deep-learning framework needed)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


class ModelBuilder:
    """Tiny helper that accumulates nodes / initializers of an ONNX graph."""

    def __init__(self, seed=0, opset=13):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.inits = []
        self.opset = opset
        self._n = 0

    def _name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, value, base="const"):
        name = self._name(base)
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def weight(self, *shape, scale=0.3, base="w"):
        return self.const(
            (self.rng.standard_normal(shape) * scale).astype(np.float32), base
        )

    def op(self, op_type, inputs, n_out=1, **attrs):
        outs = [self._name(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(
            helper.make_node(
                op_type, inputs, outs, name=self._name(op_type), **attrs
            )
        )
        return outs[0] if n_out == 1 else outs

    def conv(
        self,
        x,
        cin,
        cout,
        k=3,
        stride=1,
        pad=None,
        groups=1,
        bias=True,
        dilation=1,
    ):
        pad = k // 2 if pad is None else pad
        inputs = [x, self.weight(cout, cin // groups, k, k)]
        if bias:
            inputs.append(self.weight(cout, scale=0.1, base="b"))
        return self.op(
            "Conv",
            inputs,
            kernel_shape=[k, k],
            strides=[stride, stride],
            pads=[pad] * 4,
            dilations=[dilation, dilation],
            group=groups,
        )

    def silu(self, x):
        return self.op("Mul", [x, self.op("Sigmoid", [x])])

    def build(self, input_name, input_shape, output, output_shape=None):
        graph = helper.make_graph(
            self.nodes,
            "test",
            [
                helper.make_tensor_value_info(
                    input_name, TensorProto.FLOAT, input_shape
                )
            ],
            [
                helper.make_tensor_value_info(
                    output, TensorProto.FLOAT, output_shape
                )
            ],
            self.inits,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset)]
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model


def build_yolo_like(seed=0, size=32, classes=10, width=8):
    """A miniature YOLOv8-cls: Conv-SiLU stem, C2f-style block, GAP, Gemm."""
    b = ModelBuilder(seed)
    x = "images"
    y = b.silu(b.conv(x, 3, width, 3, stride=2))
    y = b.silu(b.conv(y, width, 2 * width, 3, stride=2))
    # C2f: split -> bottleneck -> concat
    a, c = b.op(
        "Split",
        [y, b.const(np.array([width, width], dtype=np.int64))],
        n_out=2,
        axis=1,
    )
    d = b.silu(b.conv(c, width, width, 3))
    d = b.op("Add", [c, b.silu(b.conv(d, width, width, 3))])
    y = b.op("Concat", [a, c, d], axis=1)
    y = b.silu(b.conv(y, 3 * width, 4 * width, 1))
    # depthwise + pointwise
    y = b.silu(b.conv(y, 4 * width, 4 * width, 3, groups=4 * width))
    y = b.silu(b.conv(y, 4 * width, 8 * width, 1))
    y = b.op("GlobalAveragePool", [y])
    y = b.op("Flatten", [y], axis=1)
    y = b.op(
        "Gemm",
        [y, b.weight(classes, 8 * width), b.weight(classes, scale=0.1)],
        transB=1,
    )
    y = b.op("Softmax", [y], axis=1)
    return b.build(x, ["N", 3, size, size], y, ["N", classes])
