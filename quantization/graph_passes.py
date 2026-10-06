"""Graph rewrites that make transformer / SiLU patterns quantization friendly.

All passes are conservative: they only fire on an exact structural match and
leave the graph untouched otherwise.

* :func:`fuse_matmul_bias`: ``MatMul(x, W) + b`` -> ``MatMul(x, W, b)``
  (internal 3-input form, so the bias joins the accumulator),
* :func:`fuse_layernorm`: decomposed LayerNorm (ReduceMean / Sub / Pow /
  ReduceMean / Add / Sqrt / Div / Mul / Add) -> ``LayerNormalization``,
* :func:`fuse_elementwise`: any element-wise subgraph with a single dynamic
  input and scalar constants (SiLU, decomposed GELU, ``x * s + t``, ...) ->
  one ``FusedElementwise`` node that is quantized as a single lookup table.
"""

from collections import defaultdict

import numpy as np

from quantization.onnx_graph import Node

# Element-wise operators allowed inside a fused region.
FUSABLE = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Erf",
    "Sigmoid",
    "Tanh",
    "Relu",
    "Neg",
    "Abs",
    "Exp",
    "HardSwish",
    "HardSigmoid",
    "LeakyRelu",
    "Elu",
    "Softplus",
    "Gelu",
    "Clip",
    "Max",
    "Min",
}


def _consumers(graph):
    cons = defaultdict(list)
    for node in graph.nodes:
        for i in node.inputs:
            if i:
                cons[i].append(node)
    return cons


def _is_scalar_const(graph, name):
    return name in graph.initializers and graph.initializers[name].size == 1


# ------------------------------------------------------------ matmul bias ----
def fuse_matmul_bias(graph):
    """``MatMul(x, W) + b`` (``W`` 2-D, ``b`` a length-N constant)."""
    cons = _consumers(graph)
    remove, replace = set(), {}
    for node in graph.nodes:
        if node.op_type != "MatMul" or len(node.inputs) != 2:
            continue
        w = graph.initializers.get(node.inputs[1])
        if w is None or w.ndim != 2 or node.inputs[0] in graph.initializers:
            continue
        users = cons[node.outputs[0]]
        if len(users) != 1 or node.outputs[0] in graph.outputs:
            continue
        add = users[0]
        if add.op_type != "Add":
            continue
        other = [i for i in add.inputs if i != node.outputs[0]]
        if len(other) != 1 or other[0] not in graph.initializers:
            continue
        bias = graph.initializers[other[0]]
        if bias.size != w.shape[1] or bias.shape[-1:] != (w.shape[1],):
            continue
        replace[id(node)] = Node(
            node.name,
            "MatMul",
            [node.inputs[0], node.inputs[1], other[0]],
            list(add.outputs),
            dict(node.attrs),
        )
        remove.add(id(add))
    if replace:
        graph.nodes = [
            replace.get(id(n), n) for n in graph.nodes if id(n) not in remove
        ]
        graph._index()
    return len(replace)


# -------------------------------------------------------------- layernorm ----
def fuse_layernorm(graph):
    """Decomposed LayerNorm over the last axis -> ``LayerNormalization``."""
    cons = _consumers(graph)
    init = graph.initializers
    n_fused = 0

    def single(tensor):
        users = cons[tensor]
        return (
            users[0]
            if len(users) == 1 and tensor not in graph.outputs
            else None
        )

    def last_axis_mean(node):
        axes = node.attr("axes")
        if axes is None and len(node.inputs) > 1:
            axes = init.get(node.inputs[1])
        return (
            node.op_type == "ReduceMean"
            and node.attr("keepdims", 1) == 1
            and axes is not None
            and list(np.ravel(axes)) == [-1]
        )

    def other(node, tensor):
        rest = [i for i in node.inputs if i != tensor]
        return rest[0] if len(rest) == 1 else None

    replacements, removals = {}, set()
    for mean in graph.nodes:
        if not last_axis_mean(mean):
            continue
        x, m = mean.inputs[0], mean.outputs[0]
        sub = single(m)
        if not (sub and sub.op_type == "Sub" and sub.inputs == [x, m]):
            continue
        d = sub.outputs[0]
        # x may also feed a residual connection, but both LN reads must exist
        if not {id(mean), id(sub)} <= {id(u) for u in cons[x]}:
            continue
        users_d = cons[d]
        pw = [u for u in users_d if u.op_type in ("Pow", "Mul")]
        dv = [u for u in users_d if u.op_type == "Div"]
        if len(users_d) != 2 or len(pw) != 1 or len(dv) != 1:
            continue
        pw, div = pw[0], dv[0]
        if pw.op_type == "Pow":
            e = init.get(pw.inputs[1])
            if pw.inputs[0] != d or e is None or float(e) != 2.0:
                continue
        elif pw.inputs != [d, d]:
            continue
        mean2 = single(pw.outputs[0])
        if not (mean2 and last_axis_mean(mean2)):
            continue
        add_eps = single(mean2.outputs[0])
        if not (add_eps and add_eps.op_type == "Add"):
            continue
        eps_name = other(add_eps, mean2.outputs[0])
        if eps_name not in init or init[eps_name].size != 1:
            continue
        sqrt = single(add_eps.outputs[0])
        if not (sqrt and sqrt.op_type == "Sqrt"):
            continue
        if (
            div.inputs != [d, sqrt.outputs[0]]
            or single(sqrt.outputs[0]) is not div
        ):
            continue
        mul = single(div.outputs[0])
        if not (mul and mul.op_type == "Mul"):
            continue
        gamma = other(mul, div.outputs[0])
        add_b = single(mul.outputs[0])
        if not (add_b and add_b.op_type == "Add"):
            continue
        beta = other(add_b, mul.outputs[0])
        if gamma not in init or beta not in init:
            continue
        if init[gamma].ndim != 1 or init[beta].shape != init[gamma].shape:
            continue
        chain = [mean, sub, pw, mean2, add_eps, sqrt, div, mul, add_b]
        replacements[id(add_b)] = Node(
            add_b.name + "_layernorm",
            "LayerNormalization",
            [x, gamma, beta],
            list(add_b.outputs),
            {"axis": -1, "epsilon": float(init[eps_name])},
        )
        removals.update(id(n) for n in chain if n is not add_b)
        n_fused += 1
    if n_fused:
        graph.nodes = [
            replacements.get(id(n), n)
            for n in graph.nodes
            if id(n) not in removals
        ]
        graph._index()
    return n_fused


# -------------------------------------------------- element-wise fusion ----
def fuse_elementwise(graph, min_nodes=2):
    """Collapses single-input element-wise subgraphs into one node."""
    cons = _consumers(graph)
    init = graph.initializers
    dep = {}  # tensor -> the one dynamic tensor it is computed from
    group = defaultdict(list)
    for node in graph.nodes:
        dynamic = [i for i in node.inputs if i and i not in init]
        ok = (
            node.op_type in FUSABLE
            and len(node.outputs) == 1
            and dynamic
            and all(
                _is_scalar_const(graph, i)
                for i in node.inputs
                if i and i in init
            )
        )
        roots = {dep.get(i, i) for i in dynamic} if ok else set()
        if ok and len(roots) == 1:
            root = roots.pop()
            dep[node.outputs[0]] = root
            group[root].append(node)

    fused, removed = {}, set()
    for root, nodes in group.items():
        if len(nodes) < min_nodes:
            continue
        inside = {n.outputs[0] for n in nodes}
        used_inside = {i for n in nodes for i in n.inputs}
        terminals = [t for t in inside if t not in used_inside]
        if len(terminals) != 1:
            continue
        out = terminals[0]
        # every non-terminal tensor must stay private to the region
        private = all(
            all(id(u) in {id(n) for n in nodes} for u in cons[t])
            and t not in graph.outputs
            for t in inside
            if t != out
        )
        if not private:
            continue
        consts = {
            i: init[i] for n in nodes for i in n.inputs if i and i in init
        }
        last = max(graph.nodes.index(n) for n in nodes)
        fused[id(graph.nodes[last])] = Node(
            nodes[-1].name + "_fused",
            "FusedElementwise",
            [root],
            [out],
            {"nodes": list(nodes), "consts": consts},
        )
        removed.update(id(n) for n in nodes)
    if fused:
        out_nodes = []
        for n in graph.nodes:
            if id(n) in fused:
                out_nodes.append(fused[id(n)])
            elif id(n) not in removed:
                out_nodes.append(n)
        graph.nodes = out_nodes
        graph._index()
    return len(fused)


def run_passes(graph):
    """Applies every rewrite; returns ``{pass name: matches}``."""
    return {
        "matmul_bias": fuse_matmul_bias(graph),
        "layernorm": fuse_layernorm(graph),
        "elementwise": fuse_elementwise(graph),
    }
