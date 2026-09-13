"""
Link Predictor for AutoMol Framework

This module provides AutoGL-based link prediction functionality.
"""

import os
import yaml
import torch
import logging

# Set AutoGL backend
os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.solver import AutoLinkPredictor

from .base import AutoModelLearnerBase

logger = logging.getLogger(__name__)

class AutoModelLinkPredictor(AutoModelLearnerBase):
    """AutoGL-based link predictor for model graphs."""

    solver_cls = AutoLinkPredictor
    metric = "auc"

    @property
    def predictor(self):
        """Backward-compatible alias for the underlying AutoGL solver."""
        return self.solver

    @property
    def _real_auc(self):
        """Backward-compatible accessor for the test AUC recorded by ``fit``."""
        return self.metrics.get("auc")

    def fit(self, dataset, train_split=0.7, val_split=0.15, seed=42):
        """
        Train the link predictor using AutoGL.

        Parameters
        ----------
        dataset : torch_geometric.data.Dataset or list
            The dataset to train on
        train_split : float
            Training split ratio
        val_split : float
            Validation split ratio
        seed : int
            Random seed

        Returns
        -------
        dict
            Training results containing the test-set AUC
        """
        logger.info(f"Training link predictor on dataset with {len(dataset)} graphs")

        # Handle dataset preparation
        if hasattr(dataset, '__len__') and len(dataset) > 1:
            logger.info("Multiple graphs detected, combining for link prediction")

            # Combine multiple small graphs into one larger graph
            all_x = []
            all_edge_index = []
            node_offset = 0

            for i in range(len(dataset)):  # combine all graphs passed in
                data = dataset[i]
                if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                    all_x.append(data.x)
                    edge_index = data.edge_index + node_offset
                    all_edge_index.append(edge_index)
                    node_offset += data.num_nodes

            # Create combined graph
            if all_x and all_edge_index:
                # Legacy graphs may have heterogeneous feature dims (per-graph
                # vocabularies); pad with zeros to the widest like
                # ModelDataset._standardize_features does.
                max_dim = max(x.size(1) for x in all_x)
                all_x = [
                    torch.nn.functional.pad(x, (0, max_dim - x.size(1)))
                    if x.size(1) < max_dim else x
                    for x in all_x
                ]
                combined_x = torch.cat(all_x, dim=0)
                combined_edge_index = torch.cat(all_edge_index, dim=1)

                from torch_geometric.data import Data
                combined_data = Data(x=combined_x, edge_index=combined_edge_index)

                # Create dataset wrapper for AutoGL
                class SimpleLinkDataset:
                    def __init__(self, data):
                        self.data = [data]
                        self.num_classes = 2  # Binary link prediction
                    def __len__(self):
                        return 1
                    def __getitem__(self, idx):
                        if idx != 0:
                            # Python's iterator protocol stops on IndexError;
                            # without it `for d in dataset` would loop forever.
                            raise IndexError(idx)
                        return self.data[0]

                dataset_for_training = SimpleLinkDataset(combined_data)
                logger.info(f"Combined graph: {combined_data.num_nodes} nodes, {combined_data.edge_index.shape[1]} edges")
            else:
                raise ValueError("No valid graph data found in dataset")
        else:
            # Single graph or already prepared dataset
            dataset_for_training = dataset
            # Add num_classes if not present
            if not hasattr(dataset_for_training, 'num_classes'):
                dataset_for_training.num_classes = 2

        try:
            # Create AutoGL link predictor with config
            self.solver = self._build_solver()

            # Fit the model
            logger.info("Starting AutoGL training...")
            self.solver.fit(
                dataset_for_training,
                train_split=train_split,
                val_split=val_split,
                seed=seed
            )
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise RuntimeError(f"AutoGL training failed: {e}")

        # Report the test-set AUC computed by the solver
        return self._finish_fit()

    def _combine_graphs(self, dataset):
        """Combine multiple graphs into one for link prediction."""
        all_x = []
        all_edge_index = []
        node_offset = 0

        for i in range(min(len(dataset), 10)):  # Limit to first 10 graphs
            data = dataset[i]
            all_x.append(data.x)
            # Offset edge indices
            edge_index = data.edge_index + node_offset
            all_edge_index.append(edge_index)
            node_offset += data.num_nodes

        # Combine all
        combined_x = torch.cat(all_x, dim=0)
        combined_edge_index = torch.cat(all_edge_index, dim=1)

        from torch_geometric.data import Data
        combined_data = Data(x=combined_x, edge_index=combined_edge_index)
        return combined_data


def create_link_predictor(config_path=None, config_dict=None):
    """
    Create a link predictor from configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file
    config_dict : dict, optional
        Configuration dictionary

    Returns
    -------
    AutoModelLinkPredictor
        Configured link predictor
    """
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_dict:
        config = config_dict
    else:
        # Default configuration
        config = {
            'models': [{'name': 'gcn-model'}],
            'trainer': {'max_epoch': 100},
            'evaluation_method': ['auc']
        }

    return AutoModelLinkPredictor(config=config)
