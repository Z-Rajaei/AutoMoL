"""
Common base class for AutoMol learning components.

Provides the shared training-result bookkeeping, prediction, and
persistence API (save / load) for the three AutoMol components.
"""

import logging
import os

import torch
import yaml

logger = logging.getLogger(__name__)


class AutoModelLearnerBase:
    """Shared behaviour of AutoModelClassifier, AutoModelElementClassifier and AutoModelLinkPredictor.

    Subclasses must set ``solver_cls`` (the AutoGL solver class) and ``metric``
    (the name of the primary evaluation metric reported by ``fit``).
    """

    solver_cls = None
    metric = "acc"

    def __init__(self, config=None, structure_config=None):
        self.config = config
        # Encoding (structure) configuration used to build the graphs; stored with the
        # trained network so that inference can re-encode new models identically.
        self.structure_config = structure_config
        self.solver = None
        self.is_trained = False
        self.metrics = {}

    @classmethod
    def from_config(cls, config_path, structure_config_path=None):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        structure_config = None
        if structure_config_path is not None:
            with open(structure_config_path, "r", encoding="utf-8") as f:
                structure_config = yaml.safe_load(f)
        return cls(config, structure_config=structure_config)

    # ------------------------------------------------------------------ training
    def _build_solver(self):
        if self.solver_cls is None:
            raise NotImplementedError("solver_cls must be set by the subclass")
        return self.solver_cls.from_config(self.config)

    def _finish_fit(self, dataset=None):
        """Evaluate the fitted solver on the held-out test split and record the result.

        Raises RuntimeError if the metric cannot be computed; no default value is ever substituted.
        """
        try:
            value = self.solver.evaluate(dataset=dataset, metric=self.metric)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Training finished but the test-set {self.metric} could not be computed: {exc}"
            ) from exc
        if isinstance(value, (list, tuple)):
            value = value[0]
        self.metrics = {self.metric: float(value)}
        self.is_trained = True
        logger.info("Training completed; test %s = %.4f", self.metric, self.metrics[self.metric])
        return dict(self.metrics)

    def _require_trained(self):
        if not self.is_trained or self.solver is None:
            raise RuntimeError("The component must be trained (fit) or loaded (load) before use")

    # ----------------------------------------------------------------- inference
    def predict(self, dataset=None, **kwargs):
        """Predict labels (or link existence) for ``dataset`` with the trained network."""
        self._require_trained()
        return self.solver.predict(dataset=dataset, **kwargs)

    def predict_proba(self, dataset=None, **kwargs):
        """Predict class / link probabilities for ``dataset`` with the trained network."""
        self._require_trained()
        return self.solver.predict_proba(dataset=dataset, **kwargs)

    def evaluate(self, dataset=None, metric=None, **kwargs):
        """Evaluate the trained network on ``dataset`` (default: the test split of the fitted dataset)."""
        self._require_trained()
        value = self.solver.evaluate(dataset=dataset, metric=metric or self.metric, **kwargs)
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)

    def get_leaderboard(self):
        """Expose AutoGL's HPO/NAS leaderboard (validation scores of the explored models)."""
        self._require_trained()
        return self.solver.get_leaderboard()

    # --------------------------------------------------------------- persistence
    def save(self, path):
        """Persist the trained solver together with the configurations that produced it."""
        self._require_trained()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "component": type(self).__name__,
                "solver": self.solver,
                "config": self.config,
                "structure_config": self.structure_config,
                "metrics": self.metrics,
            },
            path,
        )
        logger.info("Trained %s saved to %s", type(self).__name__, path)

    @classmethod
    def load(cls, path, map_location=None):
        """Reload a component saved with :meth:`save` for inference or re-evaluation."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("component") != cls.__name__:
            raise ValueError(
                f"{path} contains a {payload.get('component')}, not a {cls.__name__}"
            )
        obj = cls(payload.get("config"), structure_config=payload.get("structure_config"))
        obj.solver = payload["solver"]
        obj.metrics = payload.get("metrics") or {}
        obj.is_trained = True
        logger.info("Trained %s loaded from %s", cls.__name__, path)
        return obj
