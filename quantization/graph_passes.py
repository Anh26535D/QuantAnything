"""Graph rewrites that make transformer / SiLU patterns quantization friendly.

All passes are conservative: they only fire on an exact structural match and
leave the graph untouched otherwise.

* :func:`remove_noop_cast`, :func:`canonicalize_activations`,
  :func:`merge_constant_ops`, :func:`fold_mul_into_linear`: tidy-ups,
* :func:`fold_conv_bn`: ``Conv -> BatchNormalization`` -> ``Conv``,
* :func:`fuse_matmul_bias`: ``MatMul(x, W) + b`` -> ``MatMul(x, W, b)``
  (internal 3-input form, so the bias joins the accumulator),
* :func:`fuse_layernorm`: decomposed LayerNorm (ReduceMean / Sub / Pow /
  ReduceMean / Add / Sqrt / Div / Mul / Add) -> ``LayerNormalization``,
* :func:`fold_attention_scale`, :func:`fold_layernorm_affine`,
  :func:`hoist_gather_before_layernorm`: exact algebraic simplifications
  that remove quantization points and per-channel gains,
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


# ---------------------------------- exact algebraic simplification ----
def _rewire(graph, alias):
    """Redirects every use of the tensors in ``alias`` (old -> new)."""
    for node in graph.nodes:
        node.inputs = [alias.get(i, i) for i in node.inputs]
    graph.outputs = [alias.get(o, o) for o in graph.outputs]


def _new_const(graph, base, value):
    name, k = base, 0
    while name in graph.initializers:
        k += 1
        name = f"{base}_{k}"
    graph.initializers[name] = np.asarray(value)
    return name


def _trace_qkv(graph, cons, prod, tensor):
    """Walks back from a q / k tensor to the fused QKV projection.

    Accepts ``[Transpose(0,1,3,2)] <- Squeeze <- Split <- Transpose(2,0,3,1,4)
    <- Reshape [B,T,3,H,hd] <- MatMul(x, W, b)`` and returns
    ``(matmul node, slot, head width)`` (slot 0/1/2 = q/k/v) or ``None``.
    Every tensor on the way must have a single consumer.
    """
    init = graph.initializers

    def single(t):
        return len(cons[t]) == 1 and t not in graph.outputs

    node = prod.get(tensor)
    if (
        node
        and node.op_type == "Transpose"
        and list(node.attr("perm", [])) == [0, 1, 3, 2]
    ):
        if not single(node.inputs[0]):
            return None
        tensor = node.inputs[0]
        node = prod.get(tensor)
    if not (node and node.op_type == "Squeeze" and single(node.inputs[0])):
        return None
    sp = prod.get(node.inputs[0])
    if not (sp and sp.op_type == "Split" and len(sp.outputs) == 3):
        return None
    slot = sp.outputs.index(node.inputs[0])
    if not all(single(o) for o in sp.outputs):
        return None
    tr = prod.get(sp.inputs[0])
    if not (
        tr
        and tr.op_type == "Transpose"
        and list(tr.attr("perm", [])) == [2, 0, 3, 1, 4]
        and single(tr.outputs[0])
    ):
        return None
    rs = prod.get(tr.inputs[0])
    if not (
        rs
        and rs.op_type == "Reshape"
        and rs.inputs[1] in init
        and single(rs.outputs[0])
    ):
        return None
    shape = init[rs.inputs[1]].reshape(-1)
    mm = prod.get(rs.inputs[0])
    if not (
        shape.size == 5
        and shape[2] == 3
        and mm
        and mm.op_type == "MatMul"
        and len(mm.inputs) == 3
        and single(mm.outputs[0])
    ):
        return None
    w, b = init.get(mm.inputs[1]), init.get(mm.inputs[2])
    width = int(shape[3] * shape[4])
    if w is None or b is None or w.ndim != 2 or w.shape[1] != 3 * width:
        return None
    return mm, slot, width


def fold_attention_scale(graph):
    """Folds constant attention scalings into the QKV projection weights.

    Handles both exports of ``softmax(q k^T / sqrt(d))``:

    * a scalar ``Mul`` on the logits ``MatMul(q, k^T)`` (original ViT),
    * scalar ``Mul`` nodes on ``q`` and / or ``k^T`` themselves (timm).

    The matching columns of the fused ``MatMul`` weight and bias are multiplied
    by the scalar and the ``Mul`` disappears: fewer lossy quantization points
    and no per-head constant multiplications.
    """
    cons = _consumers(graph)
    prod = {o: n for n in graph.nodes for o in n.outputs}
    init = graph.initializers
    alias, removed = {}, set()
    # per projection: accumulated column scaling [q, k, v]
    scales = {}

    def single(t):
        return len(cons[t]) == 1 and t not in graph.outputs

    for mul in graph.nodes:
        if mul.op_type != "Mul" or len(mul.inputs) != 2:
            continue
        data, scale = mul.inputs
        if scale not in init or init[scale].size != 1:
            data, scale = scale, data
        if scale not in init or init[scale].size != 1 or data in init:
            continue
        if not single(data):
            continue
        target = data
        a = prod.get(data)
        if a and a.op_type == "MatMul" and len(a.inputs) == 2:
            target = a.inputs[0]  # scale on the logits -> into q
            if not single(target):
                continue
        traced = _trace_qkv(graph, cons, prod, target)
        if traced is None:
            continue
        mm, slot, width = traced
        s = float(init[scale].reshape(()))
        entry = scales.setdefault(id(mm), (mm, width, [1.0, 1.0, 1.0]))
        entry[2][slot] *= s
        alias[mul.outputs[0]] = data
        removed.add(id(mul))
    for mm, width, factors in scales.values():
        w = init[mm.inputs[1]].copy()
        b = init[mm.inputs[2]].reshape(-1).copy()
        for slot, f in enumerate(factors):
            if f != 1.0:
                sl = slice(slot * width, (slot + 1) * width)
                w[:, sl] *= np.float32(f)
                b[sl] *= np.float32(f)
        mm.inputs = [
            mm.inputs[0],
            _new_const(graph, mm.inputs[1] + "_qkscaled", w),
            _new_const(graph, mm.inputs[2] + "_qkscaled", b),
        ]
    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        _rewire(graph, alias)
        graph._index()
    return len(removed)


def fold_layernorm_affine(graph):
    """Moves a LayerNorm's ``gamma`` / ``beta`` into the next linear layers.

    ``(n * g + b) @ W + c == n @ (g[:, None] * W) + (b @ W + c)``: the
    normalized activation then has no per-channel gain to quantize. Applies
    when every consumer is a constant-weight ``MatMul`` / ``Gemm`` (alpha =
    beta = 1) and the LayerNorm output is not a graph output.
    """
    cons = _consumers(graph)
    init = graph.initializers
    n_fused = 0
    for ln in graph.nodes:
        if ln.op_type != "LayerNormalization" or len(ln.inputs) < 3:
            continue
        gamma, beta = init.get(ln.inputs[1]), init.get(ln.inputs[2])
        out = ln.outputs[0]
        users = cons[out]
        if gamma is None or beta is None or not users or out in graph.outputs:
            continue
        if np.all(gamma == 1) and np.all(beta == 0):
            continue

        def foldable(u):
            if u.inputs[0] != out or u.inputs[1] not in init:
                return False
            w = init[u.inputs[1]]
            if w.ndim != 2:
                return False
            if u.op_type == "MatMul":
                ok_k = w.shape[0] == gamma.size
            elif (
                u.op_type == "Gemm"
                and u.attr("alpha", 1.0) == 1.0
                and u.attr("beta", 1.0) == 1.0
                and not u.attr("transA", 0)
            ):
                ok_k = w.shape[1 if u.attr("transB", 0) else 0] == gamma.size
            else:
                return False
            return ok_k and all(i in init for i in u.inputs[2:] if i)

        if not all(foldable(u) for u in users):
            continue
        for u in users:
            w = init[u.inputs[1]].astype(np.float64)
            trans = u.op_type == "Gemm" and u.attr("transB", 0)
            kdim = 1 if trans else 0
            g_b = gamma.astype(np.float64).reshape(
                [-1, 1] if kdim == 0 else [1, -1]
            )
            w2 = w * g_b
            delta = beta.astype(np.float64) @ (w.T if trans else w)
            old = (
                init[u.inputs[2]].reshape(-1)
                if len(u.inputs) > 2 and u.inputs[2]
                else 0.0
            )
            u.inputs = [
                out,
                _new_const(
                    graph, u.inputs[1] + "_lnfold", w2.astype(np.float32)
                ),
                _new_const(
                    graph, u.name + "_lnbias", (old + delta).astype(np.float32)
                ),
            ]
        ln.inputs = [
            ln.inputs[0],
            _new_const(graph, ln.inputs[1] + "_one", np.ones_like(gamma)),
            _new_const(graph, ln.inputs[2] + "_zero", np.zeros_like(beta)),
        ]
        n_fused += 1
    return n_fused


def hoist_gather_before_layernorm(graph):
    """``Gather(LayerNorm(x), i, axis=1)`` -> ``LayerNorm(Gather(x, i))``.

    LayerNorm acts on the last axis only, so selecting tokens first is exact
    and normalizes just the selected (e.g. cls) token.
    """
    cons = _consumers(graph)
    n_fused = 0
    for ln in list(graph.nodes):
        if ln.op_type != "LayerNormalization" or ln.attr("axis", -1) != -1:
            continue
        users = cons[ln.outputs[0]]
        if len(users) != 1 or ln.outputs[0] in graph.outputs:
            continue
        g = users[0]
        if not (
            g.op_type == "Gather"
            and g.attr("axis", 0) == 1
            and g.inputs[0] == ln.outputs[0]
            and g.inputs[1] in graph.initializers
        ):
            continue
        mid = ln.outputs[0] + "_gathered"
        new_g = Node(
            g.name, "Gather", [ln.inputs[0], g.inputs[1]], [mid], dict(g.attrs)
        )
        new_ln = Node(
            ln.name,
            "LayerNormalization",
            [mid] + ln.inputs[1:],
            list(g.outputs),
            dict(ln.attrs),
        )
        graph.nodes = [x for x in graph.nodes if x is not g and x is not ln]
        graph.nodes += [new_g, new_ln]
        graph._index()
        cons = _consumers(graph)
        n_fused += 1
    return n_fused


def fold_conv_bn(graph):
    """``Conv -> BatchNormalization`` -> one ``Conv`` (inference folding)."""
    cons = _consumers(graph)
    init = graph.initializers
    replace, removed = {}, set()
    for conv in graph.nodes:
        if conv.op_type != "Conv" or len(conv.inputs) < 2:
            continue
        users = cons[conv.outputs[0]]
        if len(users) != 1 or conv.outputs[0] in graph.outputs:
            continue
        bn = users[0]
        if not (
            bn.op_type == "BatchNormalization"
            and bn.inputs[0] == conv.outputs[0]
            and all(i in init for i in bn.inputs[1:5])
            and conv.inputs[1] in init
            and all(i in init for i in conv.inputs[2:] if i)
        ):
            continue
        gamma, beta, mean, var = (
            init[i].astype(np.float64) for i in bn.inputs[1:5]
        )
        w = init[conv.inputs[1]].astype(np.float64)
        a = gamma / np.sqrt(var + bn.attr("epsilon", 1e-5))
        old = (
            init[conv.inputs[2]].astype(np.float64)
            if len(conv.inputs) > 2 and conv.inputs[2]
            else 0.0
        )
        w2 = (w * a.reshape(-1, 1, 1, 1)).astype(np.float32)
        b2 = (a * (old - mean) + beta).astype(np.float32)
        replace[id(conv)] = Node(
            conv.name,
            "Conv",
            [
                conv.inputs[0],
                _new_const(graph, conv.inputs[1] + "_bnfold", w2),
                _new_const(graph, conv.name + "_bnbias", b2),
            ],
            list(bn.outputs),
            dict(conv.attrs),
        )
        removed.add(id(bn))
    if replace:
        graph.nodes = [
            replace.get(id(n), n) for n in graph.nodes if id(n) not in removed
        ]
        graph._index()
    return len(replace)


# ----------------------------------------- canonical activations / casts ----
def canonicalize_activations(graph):
    """Names fused element-wise subgraphs that are a known activation.

    The fused function is probed numerically on a grid and compared with
    ``Gelu`` (exact / tanh), ``Silu``, ``Mish``, ``HardSwish``, ``Sigmoid``,
    ``Tanh`` ...; a match replaces the anonymous ``FusedElementwise`` by the
    named operator (cheaper to evaluate, readable, same lookup table).
    """
    from quantization import float_ops

    grid = np.concatenate(
        [np.linspace(-12, 12, 4801), np.linspace(-80, 80, 321)]
    ).astype(np.float32)
    catalog = [
        ("Gelu", {"approximate": "none"}),
        ("Gelu", {"approximate": "tanh"}),
        ("Silu", {}),
        ("Mish", {}),
        ("HardSwish", {}),
        ("Sigmoid", {}),
        ("Tanh", {}),
        ("Relu", {}),
        ("Softplus", {}),
    ]
    probes = {}
    for op, attrs in catalog:
        probe = Node("probe", op, ["x"], ["y"], attrs)
        probes[(op, tuple(attrs.items()))] = float_ops.run_node(probe, [grid])[
            0
        ].astype(np.float64)

    n_named = 0
    for idx, node in enumerate(graph.nodes):
        if node.op_type != "FusedElementwise":
            continue
        y = float_ops.run_node(node, [grid])[0].astype(np.float64)
        tol = 2e-5 * np.maximum(1.0, np.abs(y))
        for (op, attrs), ref in probes.items():
            if np.all(np.abs(y - ref) <= tol):
                graph.nodes[idx] = Node(
                    node.name,
                    op,
                    list(node.inputs),
                    list(node.outputs),
                    dict(attrs),
                )
                n_named += 1
                break
    if n_named:
        graph._index()
    return n_named


def remove_noop_cast(graph):
    """Drops ``Cast`` nodes whose input already has the target type."""
    from onnx import helper

    dtypes = dict(getattr(graph, "_dtypes", {}))
    dtypes.update({k: v.dtype for k, v in graph.initializers.items()})
    alias, kept = {}, []
    for node in graph.nodes:
        node.inputs = [alias.get(i, i) for i in node.inputs]
        if node.op_type == "Cast" and node.outputs[0] not in graph.outputs:
            to = helper.tensor_dtype_to_np_dtype(node.attr("to"))
            src = dtypes.get(node.inputs[0])
            if src is not None and np.dtype(src) == np.dtype(to):
                alias[node.outputs[0]] = node.inputs[0]
                continue
        for out in node.outputs:
            if node.op_type == "Cast":
                dtypes[out] = helper.tensor_dtype_to_np_dtype(node.attr("to"))
        kept.append(node)
    if alias:
        graph.nodes = kept
        _rewire(graph, alias)
        graph._index()
    return len(alias)


# ----------------------------------------------- constant multiplications ----
def _const_operand(graph, node):
    """``(dynamic input, const array)`` of a binary node with one constant."""
    a, b = node.inputs[:2]
    init = graph.initializers
    if b in init and a not in init:
        return a, init[b]
    if a in init and b not in init:
        return b, init[a]
    return None


def merge_constant_ops(graph):
    """Merges ``(x*a)*b`` and ``(x+a)+b``; drops ``x*1`` and ``x+0``."""
    cons = _consumers(graph)
    prod = {o: n for n in graph.nodes for o in n.outputs}
    alias, removed = {}, set()
    for node in graph.nodes:
        if node.op_type not in ("Mul", "Add") or id(node) in removed:
            continue
        found = _const_operand(graph, node)
        if found is None:
            continue
        x, c = found
        neutral = 1.0 if node.op_type == "Mul" else 0.0
        if (
            c.size == 1
            and float(c.reshape(())) == neutral
            and node.outputs[0] not in graph.outputs
        ):
            alias[node.outputs[0]] = x
            removed.add(id(node))
            continue
        inner = prod.get(x)
        if not (
            inner
            and inner.op_type == node.op_type
            and id(inner) not in removed
            and len(cons[x]) == 1
            and x not in graph.outputs
        ):
            continue
        found_in = _const_operand(graph, inner)
        if found_in is None:
            continue
        x0, c0 = found_in
        merged = c0 * c if node.op_type == "Mul" else c0 + c
        node.inputs = [
            x0,
            _new_const(
                graph,
                node.name + "_merged",
                merged.astype(np.result_type(c0.dtype, c.dtype)),
            ),
        ]
        removed.add(id(inner))
    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        _rewire(graph, alias)
        graph._index()
    return len(removed)


def fold_mul_into_linear(graph):
    """``Mul(Conv/MatMul/Gemm(x, W, b), c)`` -> scaled ``W`` and ``b``.

    ``c`` must be a scalar or vary only along the output channels.
    """
    cons = _consumers(graph)
    prod = {o: n for n in graph.nodes for o in n.outputs}
    init = graph.initializers
    alias, removed = {}, set()
    for mul in graph.nodes:
        if mul.op_type != "Mul":
            continue
        found = _const_operand(graph, mul)
        if found is None:
            continue
        y, c = found
        lin = prod.get(y)
        if not (
            lin
            and lin.op_type in ("Conv", "MatMul", "Gemm")
            and len(cons[y]) == 1
            and y not in graph.outputs
            and id(lin) not in removed
            and lin.inputs[1] in init
            and all(i in init for i in lin.inputs[2:] if i)
        ):
            continue
        w = init[lin.inputs[1]]
        trans = lin.op_type == "Gemm" and lin.attr("transB", 0)
        if lin.op_type == "Conv":
            out_ch, axis = w.shape[0], 0
            ok = c.size == 1 or c.shape in (
                (out_ch,),
                (out_ch, 1, 1),
                (1, out_ch, 1, 1),
            )
        else:
            if w.ndim != 2 or (
                lin.op_type == "Gemm" and (lin.attr("transA", 0))
            ):
                continue
            out_ch = w.shape[0] if trans else w.shape[1]
            axis = 0 if trans else 1
            ok = c.size == 1 or c.shape in ((out_ch,), (1, out_ch))
        if not ok:
            continue
        vec = np.broadcast_to(c.reshape(-1), (out_ch,)).astype(np.float64)
        shape = [1] * w.ndim
        shape[axis] = -1
        w2 = (w.astype(np.float64) * vec.reshape(shape)).astype(np.float32)
        ins = [
            lin.inputs[0],
            _new_const(graph, lin.inputs[1] + "_mulfold", w2),
        ]
        if len(lin.inputs) > 2 and lin.inputs[2]:
            b = init[lin.inputs[2]].astype(np.float64)
            b2 = (np.broadcast_to(b.reshape(-1), (out_ch,)) * vec).astype(
                np.float32
            )
            ins.append(_new_const(graph, lin.inputs[2] + "_mulfold", b2))
        lin.inputs = ins
        alias[mul.outputs[0]] = y
        removed.add(id(mul))
    if removed:
        graph.nodes = [n for n in graph.nodes if id(n) not in removed]
        _rewire(graph, alias)
        graph._index()
    return len(removed)


def run_passes(graph):
    """Applies every rewrite; returns ``{pass name: matches}``."""
    return {
        "cast": remove_noop_cast(graph),
        "conv_bn": fold_conv_bn(graph),
        "matmul_bias": fuse_matmul_bias(graph),
        "layernorm": fuse_layernorm(graph),
        "elementwise": fuse_elementwise(graph),
        "activation": canonicalize_activations(graph),
        "const_ops": merge_constant_ops(graph),
        "mul_into_linear": fold_mul_into_linear(graph),
        "attention_scale": fold_attention_scale(graph),
        "hoist_gather": hoist_gather_before_layernorm(graph),
        "layernorm_affine": fold_layernorm_affine(graph),
    }
