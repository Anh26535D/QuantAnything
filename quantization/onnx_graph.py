"""Light-weight ONNX graph representation backed by NumPy arrays.

This is the only place that touches the ``onnx`` protobuf API. Everything
downstream (float reference execution, calibration, quantized inference)
works on :class:`OnnxGraph`, so no deep-learning framework is required.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass
class Node:
    """One ONNX operator instance."""

    name: str
    op_type: str
    inputs: list
    outputs: list
    attrs: dict = field(default_factory=dict)

    def attr(self, key, default=None):
        """Attribute value or ``default``."""
        return self.attrs.get(key, default)

    def __repr__(self):
        return (
            f"Node({self.op_type} {self.name!r}: "
            f"{self.inputs} -> {self.outputs})"
        )


@dataclass
class ValueInfo:
    """Name and (possibly symbolic) shape of a graph input."""

    name: str
    shape: list
    dtype: np.dtype


LINEAR_DOMAIN = "quantanything"


def _linear_layer_function(opset):
    """ONNX local function ``LinearLayer(X, W, B) = MatMul(X, W) + B``."""
    return helper.make_function(
        LINEAR_DOMAIN,
        "LinearLayer",
        ["X", "W", "B"],
        ["Y"],
        [
            helper.make_node("MatMul", ["X", "W"], ["XW"]),
            helper.make_node("Add", ["XW", "B"], ["Y"]),
        ],
        [helper.make_opsetid("", opset)],
        doc_string="Linear layer: Y = X @ W + B (MatMul with fused bias).",
    )


def _clean(attrs):
    """Node attributes in a form ``helper.make_node`` accepts."""
    out = {}
    for k, v in (attrs or {}).items():
        if isinstance(v, np.ndarray):
            v = numpy_helper.from_array(v)
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def _attr_value(attribute):
    value = helper.get_attribute_value(attribute)
    if isinstance(value, onnx.TensorProto):
        return numpy_helper.to_array(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__"):
        items = list(value)
        return [
            i.decode("utf-8") if isinstance(i, bytes) else i for i in items
        ]
    return value


def _inferred_dtypes(model):
    """``{tensor: numpy dtype}`` from (inferred) ONNX type information."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # best effort
        inferred = model
    g = inferred.graph
    dtypes = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        et = vi.type.tensor_type.elem_type
        if et:
            dtypes[vi.name] = helper.tensor_dtype_to_np_dtype(et)
    return dtypes


def _static_shapes(model):
    """``{tensor: shape}`` for every tensor whose shape is fully static."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # shape inference is best effort
        inferred = model
    shapes = {}
    g = inferred.graph
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = [
            d.dim_value if d.HasField("dim_value") else None
            for d in tt.shape.dim
        ]
        if all(d is not None for d in dims):
            shapes[vi.name] = dims
    return shapes


class OnnxGraph:
    """A topologically sorted ONNX graph with constants resolved to NumPy.

    Attributes:
        nodes: Operators in execution order.
        initializers: Constant tensors (weights and folded constants).
        inputs: Real graph inputs (initializers excluded).
        outputs: Names of the graph outputs.
    """

    def __init__(self, nodes, initializers, inputs, outputs, opset=13):
        self.nodes = list(nodes)
        self.initializers = dict(initializers)
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.opset = opset
        self._static_shapes = {}
        self._dtypes = {}
        self.fusions = {}
        self._index()

    def _index(self):
        self.producers = {}
        self.consumers = defaultdict(list)
        for node in self.nodes:
            for out in node.outputs:
                self.producers[out] = node
            for inp in node.inputs:
                if inp:
                    self.consumers[inp].append(node)

    # ----------------------------------------------------------- loading ----
    @classmethod
    def from_model(cls, model, optimize=True):
        """Builds a graph from a ``ModelProto`` or an ``.onnx`` file path.

        ``optimize`` applies the pattern fusions of
        :mod:`quantization.graph_passes`.
        """
        if isinstance(model, OnnxGraph):
            return model
        if isinstance(model, (str, Path)):
            model = onnx.load(str(model))
        graph = model.graph
        opset = next(
            (
                o.version
                for o in model.opset_import
                if o.domain in ("", "ai.onnx")
            ),
            13,
        )

        initializers = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }
        nodes = []
        used = set()
        for i, n in enumerate(graph.node):
            attrs = {a.name: _attr_value(a) for a in n.attribute}
            name = n.name or f"{n.op_type}_{i}"
            while name in used:  # layer names are dictionary keys
                name += f"_{i}"
            used.add(name)
            nodes.append(
                Node(name, n.op_type, list(n.input), list(n.output), attrs)
            )

        inputs = []
        for vi in graph.input:
            if vi.name in initializers:
                continue
            tt = vi.type.tensor_type
            shape = [
                d.dim_value
                if d.HasField("dim_value")
                else (d.dim_param or None)
                for d in tt.shape.dim
            ]
            dtype = helper.tensor_dtype_to_np_dtype(tt.elem_type)
            inputs.append(ValueInfo(vi.name, shape, dtype))
        outputs = [o.name for o in graph.output]

        g = cls(nodes, initializers, inputs, outputs, opset)
        g._static_shapes = _static_shapes(model)
        g._dtypes = _inferred_dtypes(model)
        g._fold_constants()
        g._remove_identities()
        g._toposort()
        if optimize:
            from quantization import graph_passes  # avoids an import cycle

            g.fusions = graph_passes.run_passes(g)
            g._toposort()
        return g

    # ------------------------------------------------------ graph passes ----
    def _fold_constants(self):
        """Evaluates every node whose inputs are all constant."""
        from quantization import float_ops  # local: avoids an import cycle

        known = set(self.initializers)
        remaining = []
        for node in self.nodes:
            if (
                node.op_type == "Shape"
                and node.inputs[0] in self._static_shapes
            ):
                # Fully static input shape: the result is a constant.
                shape = np.array(
                    self._static_shapes[node.inputs[0]], dtype=np.int64
                )
                self.initializers[node.outputs[0]] = shape[
                    node.attr("start", 0) : node.attr("end", None)
                ]
                known.add(node.outputs[0])
            elif node.op_type == "Constant":
                self.initializers[node.outputs[0]] = float_ops.constant_value(
                    node
                )
                known.add(node.outputs[0])
            elif node.inputs and all(i in known for i in node.inputs if i):
                args = [
                    self.initializers[i] if i else None for i in node.inputs
                ]
                for name, val in zip(
                    node.outputs, float_ops.run_node(node, args, self.opset)
                ):
                    self.initializers[name] = val
                    known.add(name)
            else:
                remaining.append(node)
        self.nodes = remaining
        self._index()

    def _remove_identities(self):
        """Bypasses Identity / inference-mode Dropout nodes."""
        alias = {}
        kept = []
        protected = set(self.outputs)
        for node in self.nodes:
            node.inputs = [alias.get(i, i) for i in node.inputs]
            bypass = (
                node.op_type in ("Identity", "Dropout")
                and node.outputs[0] not in protected
            )
            if bypass:
                alias[node.outputs[0]] = node.inputs[0]
                for extra in node.outputs[
                    1:
                ]:  # Dropout mask: unused at inference
                    alias.setdefault(extra, node.inputs[0])
            else:
                kept.append(node)
        self.nodes = kept
        self._index()

    def _toposort(self):
        available = set(self.initializers) | {v.name for v in self.inputs}
        indeg = {}
        waiting = defaultdict(list)
        for idx, node in enumerate(self.nodes):
            missing = {i for i in node.inputs if i and i not in available}
            indeg[idx] = len(missing)
            for m in missing:
                waiting[m].append(idx)
        ready = deque(i for i, d in indeg.items() if d == 0)
        order = []
        while ready:
            idx = ready.popleft()
            order.append(idx)
            for out in self.nodes[idx].outputs:
                for j in waiting.get(out, []):
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        ready.append(j)
        if len(order) != len(self.nodes):
            raise ValueError("ONNX graph has a cycle or an undefined tensor")
        self.nodes = [self.nodes[i] for i in order]
        self._index()

    # ----------------------------------------------------------- export ----
    def to_model(self, opset=None, linear_as_function=True):
        """Writes the (optimized) graph back as a standard ONNX ``ModelProto``.

        ``LinearLayer(x, W, b)`` (a ``MatMul`` with its bias) is written as a
        call of the ONNX local function ``quantanything::LinearLayer`` (one
        node in viewers, inlined by runtimes); with
        ``linear_as_function=False`` it is expanded to ``MatMul`` + ``Add``.
        ``FusedElementwise`` is expanded to its nodes, ``Silu`` to
        ``Sigmoid`` + ``Mul``.
        ``Gelu`` needs opset 20, so the model is written with
        ``max(opset, 20)`` when it is present. Requires opset >= 13.
        """
        if self.opset < 13:
            raise NotImplementedError("export needs a model with opset >= 13")
        nodes = []

        def emit(op, inputs, outputs, name, attrs=None):
            nodes.append(
                helper.make_node(
                    op, inputs, outputs, name=name, **_clean(attrs)
                )
            )

        consts = dict(self.initializers)
        uses_gelu = False
        uses_linear = False
        for node in self.nodes:
            op, attrs = node.op_type, dict(node.attrs)
            if op == "FusedElementwise":
                for sub in attrs["nodes"]:
                    emit(
                        sub.op_type,
                        sub.inputs,
                        sub.outputs,
                        sub.name,
                        sub.attrs,
                    )
                    uses_gelu |= sub.op_type == "Gelu"
                for k, v in attrs["consts"].items():
                    consts.setdefault(k, v)
            elif op == "LinearLayer":
                if linear_as_function:
                    uses_linear = True
                    nodes.append(
                        helper.make_node(
                            "LinearLayer",
                            node.inputs,
                            node.outputs,
                            name=node.name,
                            domain=LINEAR_DOMAIN,
                        )
                    )
                else:
                    mid = node.outputs[0] + "_nobias"
                    emit("MatMul", node.inputs[:2], [mid], node.name)
                    emit(
                        "Add",
                        [mid, node.inputs[2]],
                        node.outputs,
                        node.name + "_bias",
                    )
            elif op == "Silu":
                sig = node.outputs[0] + "_sigmoid"
                emit("Sigmoid", node.inputs, [sig], node.name + "_sigmoid")
                emit("Mul", [node.inputs[0], sig], node.outputs, node.name)
            elif (
                op == "Split"
                and len(node.inputs) == 1
                and "num_outputs" not in attrs
            ):
                attrs["num_outputs"] = len(node.outputs)
                emit(op, node.inputs, node.outputs, node.name, attrs)
            else:
                uses_gelu |= op == "Gelu"
                emit(op, node.inputs, node.outputs, node.name, attrs)

        version = max(opset or self.opset, 20 if uses_gelu else 0)
        if version >= 18:
            for n in nodes:
                if (
                    n.op_type == "Split"
                    and not any(a.name == "num_outputs" for a in n.attribute)
                    and len(n.input) == 1
                ):
                    n.attribute.append(
                        helper.make_attribute("num_outputs", len(n.output))
                    )
                if (
                    n.op_type == "ReduceMean"
                ):  # axes became an input (opset 18)
                    axes = [a for a in n.attribute if a.name == "axes"]
                    if axes:
                        name = n.name + "_axes"
                        consts[name] = np.array(axes[0].ints, dtype=np.int64)
                        n.input.append(name)
                        n.attribute.remove(axes[0])
        used = {i for n in nodes for i in n.input if i}
        inits = [
            numpy_helper.from_array(np.asarray(v), k)
            for k, v in consts.items()
            if k in used
        ]
        graph = helper.make_graph(
            nodes,
            "optimized",
            [
                helper.make_tensor_value_info(
                    v.name,
                    helper.np_dtype_to_tensor_dtype(np.dtype(v.dtype)),
                    v.shape,
                )
                for v in self.inputs
            ],
            [
                helper.make_tensor_value_info(o, TensorProto.FLOAT, None)
                for o in self.outputs
            ],
            inits,
        )
        imports = [helper.make_opsetid("", version)]
        functions = []
        if uses_linear:
            imports.append(helper.make_opsetid(LINEAR_DOMAIN, 1))
            functions.append(_linear_layer_function(version))
        model = helper.make_model(
            graph, opset_imports=imports, functions=functions
        )
        model.ir_version = max(8, min(model.ir_version, 10))
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        return model

    def save(self, path, opset=None, linear_as_function=True):
        """Saves the optimized graph as an ONNX file."""
        onnx.save(self.to_model(opset, linear_as_function), str(path))

    # ---------------------------------------------------------- helpers ----
    def is_constant(self, name):
        """True if ``name`` is an initializer / folded constant."""
        return name in self.initializers

    @property
    def input_names(self):
        """Names of the real graph inputs."""
        return [v.name for v in self.inputs]
