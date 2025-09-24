"""
Link Predictor for AutoMol Framework

This module provides AutoGL-based link prediction functionality.
"""

import os
import yaml
import json
import torch
import numpy as np
import logging
from torch_geometric.transforms import RandomLinkSplit

# Set AutoGL backend
os.environ["AUTOGL_BACKEND"] = "pyg"

from autogl.solver import AutoLinkPredictor
from autogl.module.train.evaluation import Auc
from autogl.datasets import build_dataset_from_name
from torch_geometric.data import DataLoader

logger = logging.getLogger(__name__)

class AutoModelLinkPredictor:
    """AutoGL-based link predictor for model graphs."""
    def __init__(self, config=None):
        self.config = config
        self.predictor = None
        self.is_trained = False
        self._real_auc = None

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

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
            Training results containing AUC
        """
        logger.info(f"Training link predictor on dataset with {len(dataset)} graphs")
        
        # Handle dataset preparation
        if hasattr(dataset, '__len__') and len(dataset) > 1:
            logger.info("Multiple graphs detected, combining for link prediction")
            
            # Combine multiple small graphs into one larger graph
            all_x = []
            all_edge_index = []
            node_offset = 0
            
            for i in range(min(len(dataset), 50)):  # Limit to prevent memory issues
                data = dataset[i]
                if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                    all_x.append(data.x)
                    edge_index = data.edge_index + node_offset
                    all_edge_index.append(edge_index)
                    node_offset += data.num_nodes
            
            # Create combined graph
            if all_x and all_edge_index:
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
            self.predictor = AutoLinkPredictor.from_config(self.config)
            
            # Fit the model
            logger.info("Starting AutoGL training...")
            self.predictor.fit(
                dataset_for_training,
                train_split=train_split,
                val_split=val_split,
                seed=seed
            )
            
            # Method 1: Try to get data from leaderboard attributes
            try:
                leaderboard = self.predictor.get_leaderboard()
                logger.info(f"Leaderboard type: {type(leaderboard)}")
                
                if hasattr(leaderboard, 'perform_dict'):
                    perform_dict = leaderboard.perform_dict
                    logger.info(f"Perform dict available")
                    
                    # Try to get from DataFrame inside perform_dict
                    if hasattr(perform_dict, 'iloc'):
                        best_auc = perform_dict.iloc[0, -1]  # First row, last column
                        logger.info(f"✅ Extracted real AUC from leaderboard DataFrame: {best_auc}")
                        self._real_auc = float(best_auc)
                        self.is_trained = True
                        return {"auc": float(best_auc), "method": "leaderboard_dataframe"}
                    
                    # Get AUC from perform_dict if it's a regular dict
                    if isinstance(perform_dict, dict) and 'auc' in perform_dict:
                        auc_values = perform_dict['auc']
                        if isinstance(auc_values, list) and len(auc_values) > 0:
                            best_auc = max(auc_values)
                            logger.info(f"✅ Extracted real AUC from perform_dict list: {best_auc}")
                            self._real_auc = float(best_auc)
                            self.is_trained = True
                            return {"auc": float(best_auc), "method": "leaderboard_dict"}
                            
            except Exception as e:
                logger.warning(f"Could not get AUC from leaderboard: {e}")
            
            # Method 2: Try evaluation method
            try:
                if hasattr(self.predictor, 'evaluate'):
                    logger.info("Trying evaluate method...")
                    eval_result = self.predictor.evaluate()
                    logger.info(f"Evaluation result: {eval_result}")
                    
                    if isinstance(eval_result, (float, int)):
                        auc = float(eval_result)
                        if 0.0 <= auc <= 1.0:  # Validate AUC range
                            logger.info(f"✅ Extracted real AUC from evaluate method: {auc}")
                            self._real_auc = auc
                            self.is_trained = True
                            return {"auc": auc, "method": "evaluate"}
                        
            except Exception as e:
                logger.warning(f"Could not get AUC from evaluate method: {e}")
            
            # If we reach here, we couldn't extract real AUC
            raise RuntimeError("Could not extract real AUC from AutoGL - all extraction methods failed")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise RuntimeError(f"AutoGL training failed: {e}")

    def predict(self, data):
        """Make predictions on new data."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        if self.predictor is None:
            raise RuntimeError("No trained predictor available")
        
        return self.predictor.predict(data)

    def evaluate(self, test_data=None):
        """Evaluate the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        if self._real_auc is not None:
            return {"auc": self._real_auc}
        
        if self.predictor is not None and hasattr(self.predictor, 'evaluate'):
            return {"auc": self.predictor.evaluate(test_data)}
        
        raise RuntimeError("No evaluation method available")

    # Method removed: _extract_auc_from_output - was unused and unnecessary
    # We use direct AutoGL API methods instead: leaderboard.perform_dict
        
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
    
    def get_real_auc(self):
        """Get the real AUC from training."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        if self._real_auc is None:
            raise ValueError("No real AUC found!")
        return self._real_auc

    def save(self, path):
        """Save trained model and hyperparameters."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if hasattr(self, 'predictor') and self.predictor is not None:
                # Save the entire predictor object
                torch.save({
                    'predictor_state': self.predictor,
                    'config': self.config,
                    'real_auc': self._real_auc
                }, path)
                logger.info(f"Model saved to {path}")
            else:
                raise ValueError("No predictor to save")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise 

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