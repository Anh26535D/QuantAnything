"""Writes an optimized copy of an ONNX model (the graph the quantizer sees).

All passes of ``quantization.graph_passes`` are applied (constant folding,
Cast / Shape / scale chains removed, LayerNorm and GELU recognised,
Conv+BN and constant Mul folded into weights, ...) and the result is saved
as a standard ONNX file that runs in any runtime.

``MatMul + bias`` is written as one ``LinearLayer`` node (an ONNX local
function in domain ``quantanything``; ``--expand-linear`` writes plain
``MatMul`` + ``Add`` for tools that do not support functions).

Usage:
    uv run python optimize_onnx.py vit_tiny.onnx -o vit_tiny_opt.onnx
"""

import argparse
import collections

import numpy as np
import onnx

from quantization import OnnxGraph


def summarize(path_or_model):
    """``(node count, {op: count})`` of an ONNX file / model (no Constant)."""
    model = (
        onnx.load(path_or_model)
        if isinstance(path_or_model, str)
        else path_or_model
    )
    ops = collections.Counter(
        n.op_type for n in model.graph.node if n.op_type != "Constant"
    )
    return sum(ops.values()), dict(sorted(ops.items()))


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument(
        "--expand-linear",
        action="store_true",
        help="write MatMul + Add instead of LinearLayer functions",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="compare outputs with ONNX Runtime on random input",
    )
    args = p.parse_args(argv)
    out = args.output or args.input.replace(".onnx", "_opt.onnx")

    graph = OnnxGraph.from_model(args.input)
    graph.save(out, linear_as_function=not args.expand_linear)
    n0, _ = summarize(args.input)
    n1, ops = summarize(out)
    print(f"{args.input}: {n0} nodes -> {out}: {n1} nodes")
    print(f"passes: { {k: v for k, v in graph.fusions.items() if v} }")
    print(f"operators: {ops}")

    if args.check:
        import onnxruntime as ort

        a = ort.InferenceSession(args.input)
        b = ort.InferenceSession(out)
        inp = a.get_inputs()[0]
        x = (
            np.random.default_rng(0)
            .standard_normal(inp.shape)
            .astype(np.float32)
        )
        ya = a.run(None, {inp.name: x})[0]
        yb = b.run(None, {inp.name: x})[0]
        print(f"max |original - optimized| = {np.abs(ya - yb).max():.3g}")
    return out


if __name__ == "__main__":
    main()
