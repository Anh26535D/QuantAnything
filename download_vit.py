"""Gets a Vision Transformer as a static-batch ONNX file for this project.

Two sources:

* ``--source hf`` (default): downloads a pre-exported ONNX file from the
  Hugging Face Hub. Needs only ``huggingface_hub``
  (``uv pip install huggingface_hub``).
* ``--source timm``: exports a pretrained ``timm`` model with PyTorch. Needs
  ``torch`` and ``timm``.

The quantizer folds ``Shape -> Gather -> Expand`` chains only when the batch
size is static, so the batch dimension is pinned (``--batch``, default 1).

Examples:
    uv run python download_vit.py
    uv run python download_vit.py --source timm --model vit_tiny_patch16_224
    uv run python -m quantization.evaluator --onnx vit.onnx --mode integer

Note: ViTs expect normalized input (e.g. mean=std=0.5 for the Google / timm
augreg checkpoints). ``quantization.evaluator.load_dataset`` only scales to
[0, 1], so normalize the calibration data accordingly for real accuracy.
"""

import argparse

import onnx
from onnx.tools import update_model_dims


def pin_batch(model, batch):
    """Returns ``model`` with the batch dimension of inputs/outputs fixed."""

    def dims(values):
        out = {}
        for v in values:
            shape = [
                d.dim_value if d.HasField("dim_value") else (d.dim_param or 1)
                for d in v.type.tensor_type.shape.dim
            ]
            shape[0] = batch
            out[v.name] = shape
        return out

    inputs = [
        v
        for v in model.graph.input
        if v.name not in {i.name for i in model.graph.initializer}
    ]
    new = update_model_dims.update_inputs_outputs_dims(
        model, dims(inputs), dims(model.graph.output)
    )
    # stale (dynamic) intermediate shapes would block shape inference
    del new.graph.value_info[:]
    return onnx.shape_inference.infer_shapes(new)


def download_hf(repo_id, filename):
    """Downloads an ONNX file from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def export_timm(model_name, imgsz, batch, path, opset=17):
    """Exports a pretrained timm model to a static-shape ONNX file."""
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=True).eval()
    dummy = torch.randn(batch, 3, imgsz, imgsz)
    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=opset,
        input_names=["images"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    return path


def main(argv=None):
    """Downloads / exports the model and prints what the quantizer sees."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", choices=["hf", "timm"], default="hf")
    parser.add_argument(
        "--repo",
        default="Xenova/vit-base-patch16-224",
        help="Hugging Face repo (--source hf)",
    )
    parser.add_argument(
        "--filename",
        default="onnx/model.onnx",
        help="file inside the repo (--source hf)",
    )
    parser.add_argument(
        "--model",
        default="vit_tiny_patch16_224",
        help="timm model name (--source timm)",
    )
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", default="vit.onnx")
    args = parser.parse_args(argv)

    if args.source == "hf":
        print(f"Downloading {args.repo}/{args.filename} ...")
        model = onnx.load(download_hf(args.repo, args.filename))
        model = pin_batch(model, args.batch)
        onnx.save(model, args.output)
    else:
        print(f"Exporting timm model {args.model} ...")
        export_timm(args.model, args.imgsz, args.batch, args.output)
    print(f"Saved {args.output}")

    from quantization import OnnxGraph

    graph = OnnxGraph.from_model(args.output)
    ops = {}
    for node in graph.nodes:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(f"Inputs: {[(v.name, v.shape) for v in graph.inputs]}")
    print(f"Fusions applied: {graph.fusions}")
    print(f"Operators: {dict(sorted(ops.items()))}")


if __name__ == "__main__":
    main()
