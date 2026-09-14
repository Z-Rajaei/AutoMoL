"""
Feature Encoder for AutoMol Framework

Encodes node/edge attributes of transformed model JSONs into fixed-width
feature vectors according to the ``features`` section of the DSL
configuration. Vocabularies and the word2vec model are fitted once over the
whole dataset (``fit``) so every graph shares one feature layout and the
encoder can be persisted for identical re-encoding at inference time.
"""

import os
import re
import json
import pickle
import logging

import torch
import numpy as np
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

_CAMEL_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(value):
    """Split an identifier into lowercase tokens (camelCase/snake_case/digits/punctuation)."""
    s = _NON_ALNUM.sub(" ", str(value))
    s = _CAMEL_1.sub(r"\1 \2", s)
    s = _CAMEL_2.sub(r"\1 \2", s)
    return [t for t in s.lower().split() if t]


def _to_float(value):
    s = str(value).strip().lower()
    if s == "true":
        return 1.0
    if s == "false":
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


class FeatureEncoder:
    """
    Configuration-driven feature encoder.

    Parameters
    ----------
    feature_specs : list of dict
        Entries like ``{"feature": "name", "encoding": "word2vec"}`` taken
        from the DSL configuration. Encodings: ``word2vec``, ``oneHot``,
        ``numerical``.
    w2v_dim : int
        Dimension of the word2vec embeddings.
    seed : int
        Seed for word2vec training (use ``workers=1`` + fixed
        ``PYTHONHASHSEED`` for full determinism).
    """

    SUPPORTED_ENCODINGS = ("word2vec", "oneHot", "numerical")

    def __init__(self, feature_specs, w2v_dim=64, seed=42):
        self.feature_specs = list(feature_specs or [])
        for s in self.feature_specs:
            enc = s.get("encoding")
            if enc not in self.SUPPORTED_ENCODINGS:
                raise ValueError(
                    f"Unknown encoding '{enc}' for feature '{s.get('feature')}'; "
                    f"supported: {', '.join(self.SUPPORTED_ENCODINGS)}"
                )
        self.w2v_dim = w2v_dim
        self.seed = seed
        self.node_type_vocab = []
        self.edge_type_vocab = []
        self.onehot_vocabs = {}
        self.w2v = None
        self.attribute_coverage = {}
        self._fitted = False

    # ------------------------------------------------------------------ fit

    @staticmethod
    def _iter_models(json_paths):
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                yield json.load(f)

    def fit(self, json_paths):
        """Pass 1 over all transformed_*.json files: build global vocabularies
        and train word2vec on every word2vec-declared attribute value."""
        node_types, edge_types = set(), set()
        onehot_values = {
            s["feature"]: set()
            for s in self.feature_specs
            if s.get("encoding") == "oneHot"
        }
        sentences = []
        w2v_feats = [
            s["feature"] for s in self.feature_specs if s.get("encoding") == "word2vec"
        ]
        coverage = {s["feature"]: 0 for s in self.feature_specs}
        numerical_bad = {
            s["feature"]: 0
            for s in self.feature_specs
            if s.get("encoding") == "numerical"
        }

        for model in self._iter_models(json_paths):
            for node in model.get("nodes", []):
                node_types.add(node.get("type", "Unknown"))
                attrs = node.get("attributes", {})
                for feat in coverage:
                    if feat in attrs:
                        coverage[feat] += 1
                for feat in onehot_values:
                    if feat in attrs:
                        onehot_values[feat].add(str(attrs[feat]))
                for feat in w2v_feats:
                    if feat in attrs:
                        sentences.append(tokenize(attrs[feat]))
                for feat in numerical_bad:
                    if feat in attrs:
                        try:
                            float(str(attrs[feat]).strip())
                        except (TypeError, ValueError):
                            if str(attrs[feat]).strip().lower() not in ("true", "false"):
                                numerical_bad[feat] += 1
            for edge in model.get("edges", []):
                edge_types.add(edge.get("type", "Unknown"))

        self.attribute_coverage = coverage
        n_models = len(json_paths)
        for s in self.feature_specs:
            feat, enc = s["feature"], s.get("encoding")
            if coverage[feat] == 0:
                logger.warning(
                    "Feature '%s' (%s) is declared in the configuration but never "
                    "occurs in the %d models", feat, enc, n_models,
                )
            elif enc == "numerical" and numerical_bad.get(feat, 0) > coverage[feat] / 2:
                logger.warning(
                    "Feature '%s' (numerical): %d of %d observed values could not "
                    "be parsed as numbers and were encoded as 0",
                    feat, numerical_bad[feat], coverage[feat],
                )

        self.node_type_vocab = sorted(node_types)
        self.edge_type_vocab = sorted(edge_types)
        self.onehot_vocabs = {k: sorted(v) for k, v in onehot_values.items()}

        if sentences and w2v_feats:
            from gensim.models import Word2Vec

            self.w2v = Word2Vec(
                sentences=sentences,
                vector_size=self.w2v_dim,
                min_count=1,
                window=3,
                sg=1,
                seed=self.seed,
                workers=1,
                epochs=20,
            )
            logger.info(
                "Word2Vec trained: %d sentences, vocab %d, dim %d",
                len(sentences), len(self.w2v.wv), self.w2v_dim,
            )
        self._fitted = True
        return self

    # -------------------------------------------------------------- encode

    @property
    def feature_dim(self):
        dim = len(self.node_type_vocab) + 1
        for s in self.feature_specs:
            enc = s.get("encoding")
            if enc == "word2vec":
                dim += self.w2v_dim
            elif enc == "oneHot":
                dim += len(self.onehot_vocabs.get(s["feature"], [])) + 1
            elif enc == "numerical":
                dim += 1
        return dim

    def _one_hot(self, value, vocab):
        vec = [0.0] * (len(vocab) + 1)
        try:
            vec[vocab.index(value)] = 1.0
        except ValueError:
            vec[-1] = 1.0
        return vec

    def _w2v_encode(self, value):
        tokens = tokenize(value)
        if self.w2v is None or not tokens:
            return [0.0] * self.w2v_dim
        vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        if not vecs:
            return [0.0] * self.w2v_dim
        return np.mean(vecs, axis=0).astype(np.float32).tolist()

    def _node_features(self, node):
        attrs = node.get("attributes", {})
        feats = self._one_hot(node.get("type", "Unknown"), self.node_type_vocab)
        for s in self.feature_specs:
            feat, enc = s["feature"], s.get("encoding")
            present = feat in attrs
            if enc == "word2vec":
                feats += self._w2v_encode(attrs[feat]) if present else [0.0] * self.w2v_dim
            elif enc == "oneHot":
                vocab = self.onehot_vocabs.get(feat, [])
                feats += self._one_hot(str(attrs[feat]), vocab) if present else [0.0] * (len(vocab) + 1)
            elif enc == "numerical":
                feats.append(_to_float(attrs[feat]) if present else 0.0)
        return feats

    def encode_graph(self, model_json):
        """Encode one transformed model into a PyG ``Data`` object.

        ``model_json`` may be a path to a transformed_*.json file or an
        already-loaded dict. Edge direction is kept as stored in the JSON.
        """
        if not self._fitted:
            raise RuntimeError("FeatureEncoder must be fit() or load()ed first")
        if isinstance(model_json, (str, os.PathLike)):
            with open(model_json, "r", encoding="utf-8") as f:
                model = json.load(f)
        else:
            model = model_json

        id_to_idx = {}
        feats = []
        for i, node in enumerate(model.get("nodes", [])):
            id_to_idx[node.get("id")] = i
            feats.append(self._node_features(node))

        edge_index, edge_attr = [], []
        for edge in model.get("edges", []):
            s, t = edge.get("source"), edge.get("target")
            if s in id_to_idx and t in id_to_idx:
                edge_index.append([id_to_idx[s], id_to_idx[t]])
                edge_attr.append(
                    self._one_hot(edge.get("type", "Unknown"), self.edge_type_vocab)
                )

        x = (
            torch.tensor(feats, dtype=torch.float)
            if feats
            else torch.zeros((1, self.feature_dim), dtype=torch.float)
        )
        ei = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index
            else torch.zeros((2, 0), dtype=torch.long)
        )
        ea = (
            torch.tensor(edge_attr, dtype=torch.float)
            if edge_attr
            else torch.zeros((0, len(self.edge_type_vocab) + 1), dtype=torch.float)
        )
        return Data(x=x, edge_index=ei, edge_attr=ea)

    # ----------------------------------------------------------- persist

    def save(self, dir_path):
        """Save vocabularies and the word2vec model under ``dir_path``."""
        os.makedirs(dir_path, exist_ok=True)
        state = {
            "feature_specs": self.feature_specs,
            "w2v_dim": self.w2v_dim,
            "seed": self.seed,
            "node_type_vocab": self.node_type_vocab,
            "edge_type_vocab": self.edge_type_vocab,
            "onehot_vocabs": self.onehot_vocabs,
            "attribute_coverage": self.attribute_coverage,
            "has_w2v": self.w2v is not None,
        }
        with open(os.path.join(dir_path, "encoder_state.pkl"), "wb") as f:
            pickle.dump(state, f)
        if self.w2v is not None:
            self.w2v.save(os.path.join(dir_path, "word2vec.model"))

    @classmethod
    def load(cls, dir_path):
        with open(os.path.join(dir_path, "encoder_state.pkl"), "rb") as f:
            state = pickle.load(f)
        enc = cls(state["feature_specs"], state["w2v_dim"], state["seed"])
        enc.node_type_vocab = state["node_type_vocab"]
        enc.edge_type_vocab = state["edge_type_vocab"]
        enc.onehot_vocabs = state["onehot_vocabs"]
        enc.attribute_coverage = state.get("attribute_coverage", {})
        if state["has_w2v"]:
            from gensim.models import Word2Vec

            enc.w2v = Word2Vec.load(os.path.join(dir_path, "word2vec.model"))
        enc._fitted = True
        return enc
