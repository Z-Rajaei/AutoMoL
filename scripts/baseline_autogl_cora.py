"""Plain-AutoGL baseline on Cora: same NN config and seeds as the AutoMol
element-classification runs, but no AutoMol wrapper. Appends records to
results/replication_results.jsonl with task="element_autogl_plain".

Usage:
    python scripts/baseline_autogl_cora.py
"""
import os, sys, json, time, logging
os.environ["AUTOGL_BACKEND"] = "pyg"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
import yaml
from torch_geometric.datasets import Planetoid
from autogl.solver import AutoNodeClassifier

JSONL = os.path.join("results", "replication_results.jsonl")

for seed in [42, 0, 1, 2, 3]:
    torch.manual_seed(seed)
    dataset = Planetoid(root="data/Cora", name="Cora")
    cfg = yaml.safe_load(open("configs/cora_reduced_config.yaml"))
    cfg.setdefault("hpo", {})["max_evals"] = 10
    entry = {"task": "element_autogl_plain", "seed": seed, "evals": 10,
             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        solver = AutoNodeClassifier.from_config(cfg)
        t0 = time.time()
        solver.fit(dataset, seed=seed)
        elapsed = time.time() - t0
        acc = solver.evaluate(dataset, metric="acc")
        entry.update(metric="acc", value=acc, elapsed_s=elapsed, status="ok")
    except Exception:
        import traceback
        entry.update(metric=None, value=None, status="failed",
                     traceback=traceback.format_exc())
    with open(JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("RESULT:", json.dumps({k: entry.get(k) for k in
          ("task", "seed", "evals", "metric", "value", "status")}))
