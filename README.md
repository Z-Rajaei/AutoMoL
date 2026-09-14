# AutoMol

Configuration-driven automated graph learning for Model-Driven Engineering (MDE) tasks.

AutoMol lets MDE practitioners apply graph neural networks to models and metamodels
through two YAML files - a *structure configuration* that says how model elements are
encoded as graphs, and a *neural-network configuration* that is handed to
[AutoGL](https://github.com/THUMNLab/AutoGL) for architecture search, hyper-parameter
optimisation and training. Three components are provided:

| Component | Task | Granularity |
|---|---|---|
| `AutoModelClassifier` | model classification | whole model (graph level) |
| `AutoModelLinkPredictor` | link prediction | relationship (edge level) |
| `AutoModelElementClassifier` | model element classification | element (node level) |

This repository is the replication package of the paper
*AutoMol: Configuration-Driven Automated Graph Learning for Model-Driven Engineering Tasks*
(submitted to Software and Systems Modeling).

## Repository layout

```
automol/                 the Python package
  core/                  element extraction, DSL-driven feature encoder, graph creation
  datasets/              PyG datasets for the three tasks
  classifiers/           the three learning components (+ shared base: predict/save/load)
configs/                 structure and neural-network configurations used in the paper
scripts/
  regenerate_features.py   encode transformed models into PyG graphs (pass 1: fit vocabularies
                           and word2vec; pass 2: write graphs; the encoder is saved for inference)
  replicate_experiments.py run the three experiments and append results to results/*.jsonl
  diag_link_baselines.py  no-learning link-prediction baselines (type-pair prior etc.)
  baseline_autogl_cora.py plain-AutoGL Cora baseline (same config and seeds)
data/
  manualDomains2/        555 Ecore metamodels (Nguyen et al.), transformed JSON and domain labels
  Movies/                movie-domain XMI models (Miranda et al.) and their transformed JSON
results/                 logs and JSON records of the runs reported in the paper
```

## Installation (tested on Windows 10, Python 3.10, CPU-only)

```
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that were verified to work together
(AutoGL 0.4.0 needs `torch 2.2.x`, `torch_geometric 2.2.0`, `numpy < 2`, `pandas 1.5`,
`networkx 2.8`). The torch / torch_scatter / torch_sparse wheels come from the
PyTorch and PyG wheel indexes listed at the top of that file.

## Reproducing the experiments

1. Encode the models into graphs (once per dataset):

```
python scripts/regenerate_features.py --json-dir data/manualDomains2/pyg_torch_data \
    --out-dir data/manualDomains2/pyg_torch_data_v2 --config configs/structure_only_config.yaml
python scripts/regenerate_features.py --json-dir data/Movies/pyg_torch_data \
    --out-dir data/Movies/pyg_torch_data_v2 --config configs/structure_config_usermovies.yaml
```

2. Run the experiments (five seeds as in the paper):

```
python scripts/replicate_experiments.py --task model   --evals 30 --seeds 42 0 1 2 3 --pyg-dir pyg_torch_data_v2
python scripts/replicate_experiments.py --task link    --evals 25 --seeds 42 0 1 2 3 --link-dir data/Movies/pyg_torch_data_v2
python scripts/replicate_experiments.py --task element --evals 10 --seeds 42 0 1 2 3
```

Each run appends one JSON record (test metric, metric after save/load round-trip,
elapsed time, AutoGL leaderboard, configuration) to `results/replication_results.jsonl`
and writes a full log to `results/<task>_evals<E>_seed<S>.log`. The Cora dataset used for
element classification is downloaded automatically by PyTorch Geometric.

Two helper scripts reproduce the additional baselines reported in the paper:

- `python scripts/diag_link_baselines.py` — no-learning link-prediction baselines
  (feature cosine, common neighbours, and the type-pair prior of Section 5.5).
- `python scripts/baseline_autogl_cora.py` — plain-AutoGL Cora baseline with the same
  configuration and seeds as the AutoMol element-classification runs.

## Using a trained component

```python
from automol.classifiers import AutoModelClassifier

clf = AutoModelClassifier.from_config("configs/model_classification_nn.yaml",
                                      structure_config_path="configs/structure_only_config.yaml")
clf.fit(dataset)                     # returns {"acc": <test accuracy>}
clf.save("saved_models/domain_classifier.pt")

clf = AutoModelClassifier.load("saved_models/domain_classifier.pt")
clf.predict(new_dataset)             # new models must be encoded with the same structure config
```

`save()` stores the trained AutoGL solver together with both configurations; `load()`
restores it for inference or re-evaluation. Metrics returned by `fit()` and `evaluate()`
are always computed by AutoGL's evaluator on the held-out test split.

## Citation

Rajaei, Z., Kolahdouz-Rahimi, S., Tisi, M.: AutoMol: Configuration-Driven Automated Graph
Learning for Model-Driven Engineering Tasks. Software and Systems Modeling (under review).
