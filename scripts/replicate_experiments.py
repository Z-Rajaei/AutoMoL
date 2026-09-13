#!/usr/bin/env python3
"""
Replication of the three AutoMol experiments reported in the paper, using the
refactored components (metrics come from AutoGL's own test-split evaluation).

Usage (from rep3/, inside .venv):
    python replicate_experiments.py --task model  --evals 30 --seeds 42
    python replicate_experiments.py --task link   --evals 25 --seeds 42
    python replicate_experiments.py --task element --evals 10 --seeds 42
    python replicate_experiments.py --task all

Every run appends one JSON line to results/replication_results.jsonl and writes
the full log to results/<task>_evals<E>_seed<S>.log.  Nothing is ever defaulted:
if a run fails, the failure (traceback) is what gets recorded.
"""
import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
import traceback

os.environ["AUTOGL_BACKEND"] = "pyg"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
JSONL = os.path.join(RESULTS_DIR, "replication_results.jsonl")


def setup_log(name):
    path = os.path.join(RESULTS_DIR, name + ".log")
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return path


def load_config(path, evals):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if evals is not None and "hpo" in cfg:
        cfg["hpo"]["max_evals"] = evals
    return cfg


def leaderboard_rows(component):
    try:
        lb = component.get_leaderboard()
        df = lb.perform_dict
        return json.loads(df.to_json(orient="records"))
    except Exception as exc:  # noqa: BLE001
        return f"leaderboard unavailable: {exc}"


def record(entry):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    print("RESULT:", json.dumps({k: entry[k] for k in ("task", "seed", "evals", "metric", "value", "status")}))


# --------------------------------------------------------------------------- tasks
def run_model_classification(evals, seed, pyg_dir="pyg_torch_data"):
    from automol.datasets.model_dataset import ModelDataset
    from automol.classifiers import AutoModelClassifier

    dataset = ModelDataset("data/manualDomains2", pyg_subdir=pyg_dir)
    labels = [int(g.y) for g in dataset]
    dist = {c: labels.count(c) for c in sorted(set(labels))}
    logging.info("ModelDataset: %d graphs, class distribution %s", len(dataset), dist)
    cfg = load_config("configs/model_classification_nn.yaml", evals)
    clf = AutoModelClassifier(cfg)
    t0 = time.time()
    res = clf.fit(dataset, train_split=0.8, val_split=0.1, seed=seed)
    elapsed = time.time() - t0
    path = os.path.join("saved_models", f"repl_model_classifier_seed{seed}.pt")
    clf.save(path)
    reloaded = AutoModelClassifier.load(path).evaluate(dataset)
    return {
        "metric": "acc", "value": res["acc"], "reloaded_value": reloaded, "elapsed_s": elapsed,
        "n_graphs": len(dataset), "class_distribution": dist, "config": cfg,
        "leaderboard": leaderboard_rows(clf), "saved_model": path, "pyg_dir": pyg_dir,
    }


def run_link_prediction(evals, seed, max_graphs, link_dir="data/Movies/pyg_torch_data"):
    from automol.classifiers import AutoModelLinkPredictor

    files = sorted(glob.glob(os.path.join(link_dir, "*.pt")))
    graphs = []
    for fp in files:
        g = torch.load(fp, weights_only=False)
        if hasattr(g, "x") and hasattr(g, "edge_index") and g.x is not None:
            graphs.append(g)
    logging.info("Movies: %d pyg files, %d valid graphs; using first %s", len(files), len(graphs), max_graphs)
    if max_graphs:
        graphs = graphs[:max_graphs]
    n_nodes = sum(int(g.num_nodes) for g in graphs)
    n_edges = sum(int(g.edge_index.shape[1]) for g in graphs)
    logging.info("Total nodes %d, total edges %d", n_nodes, n_edges)
    cfg = load_config("configs/link_prediction_nn.yaml", evals)
    lp = AutoModelLinkPredictor(cfg)
    t0 = time.time()
    res = lp.fit(graphs, train_split=0.8, val_split=0.1, seed=seed)
    elapsed = time.time() - t0
    path = os.path.join("saved_models", f"repl_link_predictor_seed{seed}.pt")
    lp.save(path)
    reloaded = AutoModelLinkPredictor.load(path).evaluate()
    return {
        "metric": "auc", "value": res["auc"], "reloaded_value": reloaded, "elapsed_s": elapsed,
        "n_graphs_used": len(graphs), "n_nodes": n_nodes, "n_edges": n_edges, "config": cfg,
        "leaderboard": leaderboard_rows(lp), "saved_model": path, "link_dir": link_dir,
    }


def run_element_classification(evals, seed):
    from torch_geometric.datasets import Planetoid
    from automol.classifiers import AutoModelElementClassifier

    dataset = Planetoid(root="data/Cora", name="Cora")
    cfg = load_config("configs/cora_reduced_config.yaml", evals)
    clf = AutoModelElementClassifier(cfg)
    t0 = time.time()
    res = clf.fit(dataset, seed=seed)
    elapsed = time.time() - t0
    path = os.path.join("saved_models", f"repl_element_classifier_seed{seed}.pt")
    clf.save(path)
    reloaded = AutoModelElementClassifier.load(path).evaluate(dataset)
    return {
        "metric": "acc", "value": res["acc"], "reloaded_value": reloaded, "elapsed_s": elapsed,
        "n_nodes": int(dataset[0].num_nodes), "config": cfg,
        "leaderboard": leaderboard_rows(clf), "saved_model": path,
    }


TASKS = {
    "model": lambda e, s, a: run_model_classification(e, s, a.pyg_dir),
    "link": lambda e, s, a: run_link_prediction(e, s, a.max_graphs, a.link_dir),
    "element": lambda e, s, a: run_element_classification(e, s),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--evals", type=int, default=None, help="override hpo.max_evals")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-graphs", type=int, default=50, help="link prediction: graphs to combine (0 = all)")
    ap.add_argument("--pyg-dir", default="pyg_torch_data", help="model classification: graph subdir under manualDomains2")
    ap.add_argument("--link-dir", default="data/Movies/pyg_torch_data", help="link prediction: directory with combined .pt graphs")
    args = ap.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    for task in tasks:
        for seed in args.seeds:
            name = f"{task}_evals{args.evals}_seed{seed}"
            log_path = setup_log(name)
            entry = {"task": task, "seed": seed, "evals": args.evals, "log": log_path,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                torch.manual_seed(seed)
                entry.update(TASKS[task](args.evals, seed, args))
                entry["status"] = "ok"
            except Exception:  # noqa: BLE001
                entry.update({"status": "failed", "metric": None, "value": None,
                              "traceback": traceback.format_exc()})
                logging.exception("%s failed", name)
            record(entry)


if __name__ == "__main__":
    main()
