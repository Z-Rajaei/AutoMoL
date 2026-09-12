"""
Model Classifier for AutoMol Framework

This module provides AutoGL-based model classification functionality.
"""

import os
import logging

# Set AutoGL backend
os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.solver import AutoGraphClassifier

from .base import AutoModelLearnerBase

logger = logging.getLogger(__name__)

class AutoModelClassifier(AutoModelLearnerBase):
    """AutoGL-based model classifier for model graph classification."""

    solver_cls = AutoGraphClassifier
    metric = "acc"

    @property
    def classifier(self):
        """Backward-compatible alias for the underlying AutoGL solver."""
        return self.solver

    def fit(self, dataset, train_split=0.8, val_split=0.1, seed=42):
        """Train the classifier on the dataset."""
        logger.info(f"Training on dataset with {len(dataset)} graphs")

        # Create AutoGL classifier with config
        self.solver = self._build_solver()

        # Train the classifier
        # AutoGL expects the dataset directly
        self.solver.fit(dataset, train_split=train_split, val_split=val_split, seed=seed)

        # Report the test-set accuracy computed by the solver
        return self._finish_fit()
