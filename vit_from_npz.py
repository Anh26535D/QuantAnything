"""Builds an ONNX Vision Transformer from Google's official ViT weights.

No PyTorch / timm / Hugging Face needed: the pre-trained ``augreg`` checkpoints
(ImageNet-21k pre-trained, ImageNet-1k fine-tuned, 224x224) are public on
Google Cloud Storage as ``.npz`` files. They are downloaded and written as a
static-batch ONNX graph (``opset 20``: ``LayerNormalization`` + ``Gelu``,
tanh approximation as in the original JAX code).

Usage:
    uv run python vit_from_npz.py --variant ti16                # classifier
    uv run python vit_from_npz.py --variant ti16 --embedding \\
        --output vit_ti16_emb.onnx                               # for ReID

Input is NCHW float32 normalized with mean = std = 0.5 (``--norm half`` in
``reid_zero_shot.py``).
"""

import argparse
import os
import urllib.request

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

BASE = "https://storage.googleapis.com/vit_models/augreg/"
VARIANTS = {
    "ti16": "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--"
    "imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "s16": "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--"
    "imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "b16": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--"
    "imagenet2012-steps_20k-lr_0.03-res_224.npz",
}


def download(variant, cache_dir="."):
    """Downloads a checkpoint (once) and returns its local path."""
    name = VARIANTS[variant]
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        print(f"Downloading {BASE}{name} ...")
        urllib.request.urlretrieve(BASE + name, path + ".part")
        os.replace(path + ".part", path)
    return path


class _Graph:
    """Minimal ONNX graph accumulator."""

    def __init__(self):
        self.nodes, self.inits, self._n = [], [], 0

    def const(self, value, name=None):
        self._n += 1
        name = name or f"const_{self._n}"
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, op_type, inputs, n_out=1, **attrs):
        self._n += 1
        outs = [f"{op_type.lower()}_{self._n}_{i}" for i in range(n_out)]
        self.nodes.append(
            helper.make_node(
                op_type, inputs, outs, name=f"{op_type}_{self._n}", **attrs
            )
        )
        return outs[0] if n_out == 1 else outs


def build_onnx(weights, batch=1, embedding=False):
    """Converts the flax parameter dict ``weights`` into an ONNX model."""
    w = {k: weights[k] for k in weights.files}
    g = _Graph()
    f32 = np.float32
    dim = w["cls"].shape[-1]
    tokens = w["Transformer/posembed_input/pos_embedding"].shape[1]
    attn = "MultiHeadDotProductAttention_1"
    heads, hd = w[f"Transformer/encoderblock_0/{attn}/query/kernel"].shape[1:]
    depth = 1 + max(
        int(k.split("encoderblock_")[1].split("/")[0])
        for k in w
        if "encoderblock_" in k
    )
    patch = w["embedding/kernel"].shape[0]
    img = patch * int(round((tokens - 1) ** 0.5))

    def layer_norm(x, prefix):
        return g.op(
            "LayerNormalization",
            [
                x,
                g.const(w[prefix + "/scale"].astype(f32)),
                g.const(w[prefix + "/bias"].astype(f32)),
            ],
            axis=-1,
            epsilon=1e-6,
        )

    def linear(x, kernel, bias):
        y = g.op("MatMul", [x, g.const(kernel.astype(f32))])
        return g.op("Add", [y, g.const(bias.astype(f32))])

    x = g.op(
        "Conv",
        [
            "images",
            g.const(w["embedding/kernel"].transpose(3, 2, 0, 1).astype(f32)),
            g.const(w["embedding/bias"].astype(f32)),
        ],
        kernel_shape=[patch, patch],
        strides=[patch, patch],
    )
    x = g.op("Reshape", [x, g.const(np.array([batch, dim, -1], np.int64))])
    x = g.op("Transpose", [x], perm=[0, 2, 1])
    cls = g.const(np.tile(w["cls"].astype(f32), (batch, 1, 1)))
    x = g.op("Concat", [cls, x], axis=1)
    pos = w["Transformer/posembed_input/pos_embedding"].astype(f32)
    x = g.op("Add", [x, g.const(pos)])

    for i in range(depth):
        p = f"Transformer/encoderblock_{i}/"
        a = p + attn + "/"
        h = layer_norm(x, p + "LayerNorm_0")
        # query / key / value kernels are [dim, heads, head_dim]
        qkv_w = np.concatenate(
            [
                w[a + n + "/kernel"].reshape(dim, -1)
                for n in ("query", "key", "value")
            ],
            axis=1,
        )
        qkv_b = np.concatenate(
            [w[a + n + "/bias"].reshape(-1) for n in ("query", "key", "value")]
        )
        qkv = linear(h, qkv_w, qkv_b)
        qkv = g.op(
            "Reshape",
            [qkv, g.const(np.array([batch, tokens, 3, heads, hd], np.int64))],
        )
        qkv = g.op("Transpose", [qkv], perm=[2, 0, 3, 1, 4])
        q, k, v = [
            g.op("Squeeze", [t, g.const(np.array([0], np.int64))])
            for t in g.op("Split", [qkv], n_out=3, axis=0, num_outputs=3)
        ]
        att = g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])])
        att = g.op("Mul", [att, g.const(f32(hd**-0.5))])
        att = g.op("Softmax", [att], axis=-1)
        o = g.op("Transpose", [g.op("MatMul", [att, v])], perm=[0, 2, 1, 3])
        o = g.op(
            "Reshape", [o, g.const(np.array([batch, tokens, dim], np.int64))]
        )
        x = g.op(
            "Add",
            [
                x,
                linear(
                    o,
                    w[a + "out/kernel"].reshape(-1, dim),
                    w[a + "out/bias"],
                ),
            ],
        )

        h = layer_norm(x, p + "LayerNorm_2")
        m = p + "MlpBlock_3/"
        h = linear(h, w[m + "Dense_0/kernel"], w[m + "Dense_0/bias"])
        h = g.op("Gelu", [h], approximate="tanh")
        x = g.op(
            "Add",
            [x, linear(h, w[m + "Dense_1/kernel"], w[m + "Dense_1/bias"])],
        )

    x = layer_norm(x, "Transformer/encoder_norm")
    x = g.op("Gather", [x, g.const(np.array(0, np.int64))], axis=1)
    if embedding:
        out, out_shape = x, [batch, dim]
    else:
        classes = w["head/bias"].shape[0]
        out = g.op(
            "Gemm",
            [
                x,
                g.const(w["head/kernel"].T.astype(f32)),
                g.const(w["head/bias"].astype(f32)),
            ],
            transB=1,
        )
        out_shape = [batch, classes]

    graph = helper.make_graph(
        g.nodes,
        "vit",
        [
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [batch, 3, img, img]
            )
        ],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT, out_shape)],
        g.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)]
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--variant", choices=sorted(VARIANTS), default="ti16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--embedding",
        action="store_true",
        help="output the cls feature instead of ImageNet logits",
    )
    p.add_argument("--weights", default=None, help="use a local .npz")
    p.add_argument("--cache-dir", default=".")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    path = args.weights or download(args.variant, args.cache_dir)
    model = build_onnx(np.load(path), args.batch, args.embedding)
    out = (
        args.output
        or f"vit_{args.variant}{'_emb' if args.embedding else ''}.onnx"
    )
    onnx.save(model, out)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()
