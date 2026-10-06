"""Downloads YOLOv8n-cls and exports it to ONNX (the only model format used).

Needs the optional ``ultralytics`` dependency: ``uv sync --extra export``.
"""

import argparse

import onnx

from quantization import OnnxGraph


def main(argv=None):
    """Exports the model and prints a short summary of the ONNX graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n-cls.pt")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args(argv)

    from quantization.evaluator import export_yolov8_cls

    print(f"--- Exporting {args.weights} to ONNX ---")
    onnx_path = export_yolov8_cls(args.weights, args.imgsz)
    print(f"ONNX model saved at: {onnx_path}")

    onnx.checker.check_model(onnx.load(onnx_path))
    graph = OnnxGraph.from_model(onnx_path)
    ops = {}
    for node in graph.nodes:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"Inputs: {[(v.name, v.shape) for v in graph.inputs]}")
    print(f"Operators: {dict(sorted(ops.items()))}")


if __name__ == "__main__":
    main()
