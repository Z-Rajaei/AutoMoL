"""
Regenerate transformed_*.pt graphs with the DSL-declared feature encodings.

Pass 1: FeatureEncoder.fit() builds global vocabularies and trains word2vec.
Pass 2: every transformed_*.json is re-encoded and written to --out-dir with
the same basename; the fitted encoder is saved to <out-dir>/encoder/.

For deterministic word2vec training run with PYTHONHASHSEED=0, e.g.:
    PYTHONHASHSEED=0 python regenerate_features.py --json-dir manualDomains2/pyg_torch_data \
        --out-dir manualDomains2/pyg_torch_data_v2 --config structure_only_config.yaml
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")  # effective only if set before start
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from automol.core.feature_encoder import FeatureEncoder


def feature_specs_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    specs = []
    for mm in cfg.get("configuration", {}).get("adaptations", {}).get("metamodels", []):
        for pkg in mm.get("packages", []):
            for cls in pkg.get("classes", []):
                for spec in cls.get("features", []) or []:
                    if spec not in specs:
                        specs.append(spec)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--w2v-dim", type=int, default=64)
    args = ap.parse_args()

    specs = feature_specs_from_config(args.config)
    print("feature specs:", specs)
    json_paths = sorted(glob.glob(os.path.join(args.json_dir, "transformed_*.json")))
    print(f"pass 1: fitting encoder on {len(json_paths)} JSON files")
    enc = FeatureEncoder(specs, w2v_dim=args.w2v_dim).fit(json_paths)
    print("feature_dim:", enc.feature_dim,
          "| node types:", len(enc.node_type_vocab),
          "| edge types:", len(enc.edge_type_vocab),
          "| oneHot vocabs:", {k: len(v) for k, v in enc.onehot_vocabs.items()})

    os.makedirs(args.out_dir, exist_ok=True)
    dims = set()
    n = 0
    for p in json_paths:
        data = enc.encode_graph(p)
        dims.add(data.x.size(1))
        torch.save(data, os.path.join(
            args.out_dir, os.path.basename(p).replace(".json", ".pt")))
        n += 1
    assert dims == {enc.feature_dim}, f"inconsistent feature dims: {dims}"
    print(f"pass 2: wrote {n} graphs, all dims == {enc.feature_dim}")
    enc.save(os.path.join(args.out_dir, "encoder"))
    print("encoder saved to", os.path.join(args.out_dir, "encoder"))

    # sanity: identical type strings map to identical columns across graphs
    for p in json_paths[:3]:
        d = torch.load(os.path.join(
            args.out_dir, os.path.basename(p).replace(".json", ".pt")))
        import json
        model = json.load(open(p, encoding="utf-8"))
        for i, node in enumerate(model["nodes"][:3]):
            t = node.get("type", "Unknown")
            col = d.x[i, : len(enc.node_type_vocab) + 1].argmax().item()
            exp = enc.node_type_vocab.index(t) if t in enc.node_type_vocab else len(enc.node_type_vocab)
            print(f"  {os.path.basename(p)[:40]} node{i} type={t} -> col {col} (expected {exp})")
            assert col == exp


if __name__ == "__main__":
    main()
