"""Zero-shot person re-identification with a (quantized) ONNX backbone.

The backbone is any image model whose embedding (e.g. the ViT cls token) is
compared by cosine distance; no ReID fine-tuning is involved. Datasets follow
the Market-1501 convention: ``query/`` and ``bounding_box_test/`` folders with
files named ``<pid>_c<cam>s<seq>_<frame>.jpg`` (``pid`` -1 = junk, 0 =
distractor). DukeMTMC-reID uses the same naming.

The protocol is the standard one: a gallery image is ignored for a query if
it has the same identity *and* camera, or if it is junk (``pid == -1``).
"""

import os
import re

import numpy as np
import onnx
from onnx import TensorProto, helper

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
_NAME = re.compile(r"^(-?\d+)_c(\d+)")

# (mean, std) presets for the usual pretrained checkpoints
NORMALIZATION = {
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


# ---------------------------------------------------------------- dataset ----
def parse_name(path):
    """``(pid, camid)`` from a Market-1501 style file name."""
    match = _NAME.match(os.path.basename(path))
    if match is None:
        raise ValueError(f"not a Market-1501 style file name: {path}")
    return int(match.group(1)), int(match.group(2))


def list_images(directory):
    """Sorted image paths in ``directory``."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )


def find_split_dirs(root):
    """Locates ``(query_dir, gallery_dir)`` under a dataset root."""
    for sub in [""] + sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ):
        base = os.path.join(root, sub)
        q = os.path.join(base, "query")
        g = os.path.join(base, "bounding_box_test")
        if os.path.isdir(q) and os.path.isdir(g):
            return q, g
    raise FileNotFoundError(
        f"no 'query' + 'bounding_box_test' folders found under {root}"
    )


def load_split(root, max_ids=None, distractors=0, seed=0):
    """Reads query / gallery lists, optionally subsampled by identity.

    Args:
        root: dataset root (or the folder holding ``query`` and
            ``bounding_box_test``).
        max_ids: keep only this many randomly chosen query identities (all of
            their query and gallery images); ``None`` keeps everything.
        distractors: extra gallery images of *other* identities to keep when
            subsampling.

    Returns:
        ``(query, gallery)``, each a ``(paths, pids, camids)`` tuple of
        arrays.
    """
    q_dir, g_dir = find_split_dirs(root)

    def read(directory):
        paths = list_images(directory)
        ids = np.array([parse_name(p) for p in paths]).reshape(-1, 2)
        return np.array(paths), ids[:, 0], ids[:, 1]

    query, gallery = read(q_dir), read(g_dir)
    if max_ids is not None:
        rng = np.random.default_rng(seed)
        pids = np.unique(query[1][query[1] >= 0])
        chosen = set(rng.choice(pids, min(max_ids, len(pids)), replace=False))
        q_keep = np.array([p in chosen for p in query[1]])
        g_in = np.array([p in chosen for p in gallery[1]])
        others = np.flatnonzero(~g_in & (gallery[1] >= 0))
        extra = rng.choice(
            others, min(distractors, len(others)), replace=False
        )
        g_keep = g_in.copy()
        g_keep[extra] = True
        query = tuple(a[q_keep] for a in query)
        gallery = tuple(a[g_keep] for a in gallery)
    return query, gallery


def load_image(path, size, mean, std):
    """Image -> float32 CHW, resized to ``size = (height, width)``."""
    from PIL import Image

    h, w = size
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
    return arr.transpose(2, 0, 1)


# --------------------------------------------------------------- features ----
def l2_normalize(x, eps=1e-12):
    """Row-wise L2 normalization."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def extract_features(infer, paths, batch, size, mean, std, progress=None):
    """Runs ``infer`` over images in fixed-size batches.

    Args:
        infer: ``f(array[batch, 3, H, W]) -> array[batch, D]``.
        batch: batch size the model was exported with; the last batch is
            padded by repeating its final image.
        progress: optional iterator wrapper such as ``tqdm``.

    Returns:
        ``[len(paths), D]`` float32 features (not normalized).
    """
    starts = range(0, len(paths), batch)
    if progress is not None:
        starts = progress(starts)
    out = []
    for s in starts:
        chunk = list(paths[s : s + batch])
        n = len(chunk)
        chunk += [chunk[-1]] * (batch - n)
        x = np.stack([load_image(p, size, mean, std) for p in chunk])
        out.append(
            np.asarray(infer(x), dtype=np.float32).reshape(batch, -1)[:n]
        )
    return np.concatenate(out)


def embedding_model(model, tensor=None):
    """Re-targets an ONNX model to output an intermediate tensor.

    With ``tensor=None`` the classification head is removed automatically:
    trailing ``Softmax`` / ``Sigmoid`` are skipped, and if the producer is a
    ``Gemm`` / ``MatMul`` its (first) input is used, e.g. the ViT cls
    embedding after the final LayerNorm. Nodes that are no longer needed are
    pruned.
    """
    graph = model.graph
    producers = {o: n for n in graph.node for o in n.output}
    if tensor is None:
        tensor = graph.output[0].name
        node = producers.get(tensor)
        while node is not None and node.op_type in ("Softmax", "Sigmoid"):
            tensor = node.input[0]
            node = producers.get(tensor)
        if node is not None and node.op_type in (
            "Gemm",
            "MatMul",
            "LinearLayer",
        ):
            tensor = node.input[0]
    if tensor not in producers and tensor not in {i.name for i in graph.input}:
        raise ValueError(f"tensor '{tensor}' not found in the model")

    needed, keep = {tensor}, []
    for node in reversed(graph.node):
        if any(o in needed for o in node.output):
            keep.append(node)
            needed.update(i for i in node.input if i)
    keep.reverse()
    inits = [t for t in graph.initializer if t.name in needed]
    del graph.node[:], graph.initializer[:], graph.output[:]
    del graph.value_info[:]
    graph.node.extend(keep)
    graph.initializer.extend(inits)
    graph.output.append(
        helper.make_tensor_value_info(tensor, TensorProto.FLOAT, None)
    )
    # fill in the (inferred) shape so the model passes onnx.checker
    inferred = onnx.shape_inference.infer_shapes(model)
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        if vi.name == tensor and vi.type.tensor_type.HasField("shape"):
            graph.output[0].type.CopyFrom(vi.type)
    return model, tensor


# ------------------------------------------------------------- evaluation ----
def cosine_distance(query_feats, gallery_feats):
    """``1 - cos`` distance matrix ``[num_query, num_gallery]``."""
    return 1.0 - l2_normalize(query_feats) @ l2_normalize(gallery_feats).T


def evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Market-1501 CMC / mAP protocol.

    Returns:
        ``{"mAP", "rank1", "rank5", "rank10", "num_valid_queries"}``; queries
        without any valid gallery match are skipped.
    """
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    order = np.argsort(distmat, axis=1, kind="stable")
    matches = (g_pids[order] == q_pids[:, None]).astype(np.int32)

    all_cmc, aps = [], []
    for i in range(num_q):
        g_order = order[i]
        # drop same-identity-same-camera and junk gallery images
        keep = ~(
            (
                (g_pids[g_order] == q_pids[i])
                & (g_camids[g_order] == q_camids[i])
            )
            | (g_pids[g_order] == -1)
        )
        m = matches[i][keep]
        if not m.any():
            continue
        cmc = m.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        precision = m.cumsum() / (np.arange(m.size) + 1.0)
        aps.append((precision * m).sum() / m.sum())
    if not all_cmc:
        raise ValueError("no query has a valid gallery match")
    cmc = np.mean(all_cmc, axis=0)
    top = lambda k: float(cmc[min(k, len(cmc)) - 1])  # noqa: E731
    return {
        "mAP": float(np.mean(aps)),
        "rank1": top(1),
        "rank5": top(5),
        "rank10": top(10),
        "num_valid_queries": len(all_cmc),
    }
