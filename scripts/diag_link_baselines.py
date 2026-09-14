"""No-learning baselines for the link-prediction experiment.

Combines the encoded movie-domain graphs exactly like
:class:`AutoModelLinkPredictor` and scores test edges with three cheap
heuristics: feature cosine similarity, common neighbours, and the
type-pair prior reported in the paper (the probability that a pair of
node types is connected in the training edges).

Usage:
    python scripts/diag_link_baselines.py [link_dir] [max_graphs]
    # defaults: data/Movies/pyg_torch_data_v2, all graphs
"""
import os, sys, glob
os.environ["AUTOGL_BACKEND"] = "pyg"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, degree
from sklearn.metrics import roc_auc_score
import networkx as nx

link_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Movies/pyg_torch_data_v2"
print("link_dir:", link_dir)
max_g = int(sys.argv[2]) if len(sys.argv) > 2 else 0
files = sorted(glob.glob(os.path.join(link_dir, "*.pt")))[:max_g or None]
graphs = [torch.load(f, weights_only=False) for f in files]
graphs = [g for g in graphs if hasattr(g, "x") and g.x is not None]

# combine exactly like _combine_graphs / fit
all_x, all_ei, off = [], [], 0
for g in graphs:
    all_x.append(g.x)
    all_ei.append(g.edge_index + off)
    off += g.num_nodes
x = torch.cat(all_x, 0)
ei = torch.cat(all_ei, 1)
print(f"combined: {x.shape[0]} nodes, {ei.shape[1]} directed edges")

# bidirectionality
es = set(map(tuple, ei.t().tolist()))
both = sum(1 for (u, v) in es if (v, u) in es)
print(f"edges with reverse also present: {both}/{len(es)}")

G = nx.Graph(); G.add_nodes_from(range(x.shape[0])); G.add_edges_from(es)
cc = list(nx.connected_components(G))
print(f"connected components: {len(cc)} (sizes {[len(c) for c in sorted(cc, key=len, reverse=True)][:10]})")
deg = degree(ei[0], num_nodes=x.shape[0]) + degree(ei[1], num_nodes=x.shape[0])
d = deg.numpy()
print(f"degree min/median/max: {d.min()}/{np.median(d)}/{d.max()}")
print(f"x shape {tuple(x.shape)}; distinct feature rows: {len(torch.unique(x, dim=0))} / {x.shape[0]}")
for a in ("node_type", "node_type_names"):
    if hasattr(graphs[0], a):
        print(f"{a}: {getattr(graphs[0], a)}")

# undirected combined graph for split
ei_u = to_undirected(ei)
data = Data(x=x, edge_index=ei_u)
data.num_nodes = x.shape[0]
split = RandomLinkSplit(num_val=0.0, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=True, neg_sampling_ratio=1.0)
tr, va, te = split(data)
pos = te.edge_label_index[:, te.edge_label == 1]
neg = te.edge_label_index[:, te.edge_label == 0]
print(f"test: {pos.shape[1]} pos, {neg.shape[1]} neg")

# (a) cosine similarity baseline
xn = torch.nn.functional.normalize(x.float(), dim=1)
def cos_scores(eidx):
    return (xn[eidx[0]] * xn[eidx[1]]).sum(1).numpy()
s = np.concatenate([cos_scores(pos), cos_scores(neg)])
y = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
print("AUC cosine(features):", roc_auc_score(y, s))

# (b) common neighbours baseline (on train graph structure)
etr = set(map(tuple, to_undirected(tr.edge_index).t().tolist()))
adj = {}
for u, v in etr:
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)
def cn(eidx):
    return np.array([len(adj.get(int(u), set()) & adj.get(int(v), set())) for u, v in eidx.t().tolist()])
s2 = np.concatenate([cn(pos), cn(neg)])
print("AUC common-neighbours:", roc_auc_score(y, s2))

# --- (c) type-pair prior baseline ---
# carry node_type through the same combination
all_nt, off = [], 0
for g in graphs:
    if hasattr(g, "node_type"):
        all_nt.append(g.node_type)
    else:
        # v2 graphs: node type = argmax of the leading type one-hot block
        # (vocab size read from the saved encoder state if available)
        try:
            import pickle
            st = pickle.load(open(os.path.join(link_dir, "encoder", "encoder_state.pkl"), "rb"))
            k = len(st["node_type_vocab"]) + 1
        except Exception:
            k = 6
        all_nt.append(g.x[:, :k].argmax(dim=1))
nt = torch.cat(all_nt)

def pair_key(eidx):
    us = nt[eidx[0]].tolist(); vs = nt[eidx[1]].tolist()
    return [tuple(sorted((int(a), int(b)))) for a, b in zip(us, vs)]

tr_pos = tr.edge_label_index[:, tr.edge_label == 1]
tr_neg = tr.edge_label_index[:, tr.edge_label == 0]
from collections import Counter
pos_ct = Counter(pair_key(tr_pos)); neg_ct = Counter(pair_key(tr_neg))
print("train pos type-pairs:", dict(pos_ct))
print("train neg type-pairs:", dict(neg_ct))

pairs = set(pos_ct) | set(neg_ct)
table = {p: pos_ct.get(p, 0) / (pos_ct.get(p, 0) + neg_ct.get(p, 0)) for p in pairs}
s3 = np.array([table.get(k, 0.5) for k in pair_key(te.edge_label_index)])
print("AUC type-pair prior:", roc_auc_score(y, s3))
print("test pos type-pairs:", dict(Counter(pair_key(pos))))
print("test neg type-pairs:", dict(Counter(pair_key(neg))))

# --- (d) occurs-at-all rule ---
s4 = np.array([1.0 if k in pos_ct else 0.0 for k in pair_key(te.edge_label_index)])
print("AUC occurs-in-train rule:", roc_auc_score(y, s4))
