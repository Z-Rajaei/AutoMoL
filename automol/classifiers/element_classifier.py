"""
Element Classifier for AutoMol Framework

This module provides AutoGL-based node (element) classification functionality.
"""

import os
import logging

os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.solver import AutoNodeClassifier

from .base import AutoModelLearnerBase

logger = logging.getLogger(__name__)

class AutoModelElementClassifier(AutoModelLearnerBase):
    """AutoGL-based node classifier for model element classification."""

    solver_cls = AutoNodeClassifier
    metric = "acc"

    @property
    def classifier(self):
        """Backward-compatible alias for the underlying AutoGL solver."""
        return self.solver

    def fit(self, dataset, train_split=0.8, val_split=0.1, seed=42):
        """Train the node classifier on the dataset."""
        logger.info(f"Training node classifier on dataset")

        # Create AutoGL node classifier with config
        self.solver = self._build_solver()

        # Train the classifier
        # For node classification, check if dataset has train/val masks
        if hasattr(dataset, 'data') and hasattr(dataset.data, 'train_mask'):
            # Dataset has pre-defined splits (like Cora)
            self.solver.fit(dataset, seed=seed)
        else:
            # Dataset needs manual split
            self.solver.fit(dataset, train_split=train_split, val_split=val_split, seed=seed)

        # Report the test-set accuracy computed by the solver
        return self._finish_fit(dataset)

    def predict_nodes(self, dataset):
        """Predict node classes."""
        return self.predict(dataset)

    def evaluate_nodes(self, dataset, metric="acc"):
        """Evaluate node classification performance."""
        return self.evaluate(dataset, metric=metric)
