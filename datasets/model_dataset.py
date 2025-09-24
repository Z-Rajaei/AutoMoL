"""
Model Dataset for AutoMol Framework

This module provides PyTorch Geometric dataset classes for model classification tasks.
"""

import os
import glob
import json
import torch
import yaml
import logging
import numpy as np
from torch_geometric.data import Dataset, Data
from pathlib import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelDataset(Dataset):
    """
    A PyTorch Geometric Dataset for loading model graphs for classification.
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mapping_file=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        transform : callable, optional
            A function/transform that takes in a `torch_geometric.data.Data` 
            object and returns a transformed version.
        pre_transform : callable, optional
            A function/transform that takes in a `torch_geometric.data.Data` 
            object and returns a transformed version. The transformation is 
            applied before the data is saved to disk.
        pre_filter : callable, optional
            A function that takes in a `torch_geometric.data.Data` object and 
            returns a boolean value, indicating whether the data object should 
            be included in the dataset.
        mapping_file : str, optional
            Path to the mapping file that maps graph IDs to class labels.
        """
        self.root = root
        self.pyg_dir = os.path.join(root, 'pyg_torch_data')
        self._data_files = None
        self._class_map = None
        self._label_mapping = None
        self._feature_dim = None
        
        # Set mapping file path
        if mapping_file is None:
            self.mapping_file = os.path.join(root, 'models', 'graph_cluster_mapping.json')
        else:
            self.mapping_file = mapping_file
            
        super(ModelDataset, self).__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def data_files(self):
        """Get all .pt files in the pyg_torch_data directory"""
        if self._data_files is None:
            self._data_files = []
            if os.path.exists(self.pyg_dir):
                self._data_files = glob.glob(os.path.join(self.pyg_dir, 'transformed_*.pt'))
                logger.info(f"Found {len(self._data_files)} .pt files in {self.pyg_dir}")
            else:
                logger.warning(f"PyG data directory {self.pyg_dir} does not exist")
        return self._data_files
    
    @property
    def feature_dim(self):
        """Get the maximum feature dimension across all graphs"""
        if self._feature_dim is None:
            # Check feature dimensions across graphs
            dims = []
            for file_path in self.data_files[:10]:  # Check first 10 for speed
                try:
                    data = torch.load(file_path)
                    if hasattr(data, 'x') and data.x is not None and data.x.dim() > 1:
                        dims.append(data.x.size(1))
                except Exception as e:
                    logger.warning(f"Error loading graph from {file_path}: {e}")
            
            if dims:
                self._feature_dim = max(dims)
                logger.info(f"Maximum feature dimension: {self._feature_dim}")
            else:
                self._feature_dim = 100  # Default
                logger.warning(f"Could not determine feature dimension, using default: {self._feature_dim}")
        
        return self._feature_dim
    
    @property
    def class_map(self):
        """Get mapping from model names to class labels"""
        if self._class_map is None:
            self._class_map = self._create_class_map()
        return self._class_map
    
    @property
    def raw_file_names(self):
        """Required by PyTorch Geometric Dataset"""
        return []
    
    @property
    def processed_file_names(self):
        """Required by PyTorch Geometric Dataset"""
        return [os.path.basename(f) for f in self.data_files]
    
    def download(self):
        """Required by PyTorch Geometric Dataset"""
        pass
    
    def process(self):
        """Required by PyTorch Geometric Dataset"""
        pass
    
    def _create_class_map(self):
        """
        Create a mapping from model names to class labels.
        """
        # Extract model names from file paths
        model_names = set()
        for file_path in self.data_files:
            file_name = os.path.basename(file_path)
            # Extract model name from file name (format: transformed_XXX_YYY_ZZZ...)
            parts = file_name.split('_')
            if len(parts) > 1:
                model_name = parts[1]  # Use the first part after 'transformed_'
                model_names.add(model_name)
        
        # Create mapping from model names to class labels
        class_map = {name: i for i, name in enumerate(sorted(model_names))}
        logger.info(f"Created class map with {len(class_map)} classes")
        return class_map
    
    def _extract_graph_id(self, file_name):
        """
        Extract graph ID from file name.
        """
        # Remove prefix and suffix
        name_without_prefix = file_name.replace("transformed_", "")
        name_without_suffix = name_without_prefix.replace(".ecore.pt", "")
        
        # Extract parts
        parts = name_without_suffix.split('_')
        
        # Graph ID is the first part
        if len(parts) > 0:
            return parts[0]
        
        return "0"
    
    def _standardize_features(self, data):
        """
        Standardize node feature dimensions.
        """
        # Ensure node features exist
        if not hasattr(data, 'x') or data.x is None or data.x.numel() == 0:
            # Create default features
            data.x = torch.zeros((data.num_nodes, self.feature_dim), dtype=torch.float)
            logger.warning(f"Graph has no node features, created default features with dim {self.feature_dim}")
        
        # Standardize feature dimensions
        if data.x.size(1) != self.feature_dim:
            # Create new tensor with standard dimensions
            x_new = torch.zeros((data.x.size(0), self.feature_dim), dtype=torch.float)
            
            # Copy existing data
            min_dim = min(data.x.size(1), self.feature_dim)
            x_new[:, :min_dim] = data.x[:, :min_dim]
            
            # Replace old features with new
            data.x = x_new
            logger.debug(f"Standardized node features from {data.x.size(1)} to {self.feature_dim}")
        
        # Remove extra metadata that might cause errors
        for attr in ['node_ids', 'node_types', 'attribute_names', 'node_type_mapping']:
            if hasattr(data, attr):
                delattr(data, attr)
        
        return data
    
    def len(self):
        """
        Return the number of graphs in the dataset.
        """
        return len(self.data_files)
    
    def get(self, idx):
        """
        Get the graph at the given index.
        """
        file_path = self.data_files[idx]
        file_name = os.path.basename(file_path)
        
        try:
            # Load graph from file
            data = torch.load(file_path)
            
            # Standardize feature dimensions
            data = self._standardize_features(data)
            
            # Determine graph label
            graph_id = self._extract_graph_id(file_name)
            
            # Use class mapping to get label
            parts = file_name.split('_')
            if len(parts) > 1:
                model_name = parts[1]  # Use the first part after 'transformed_'
                
                if model_name in self.class_map:
                    class_label = self.class_map[model_name]
                    logger.debug(f"Using label {class_label} from class map for model {model_name}")
                else:
                    logger.warning(f"Model name {model_name} not found in class map")
                    class_label = 0  # Default label
            else:
                logger.warning(f"Could not extract model name from {file_name}")
                class_label = 0  # Default label
            
            # Convert label to tensor
            data.y = torch.tensor([int(class_label)], dtype=torch.long)
            
            return data
        
        except Exception as e:
            logger.error(f"Error loading graph from {file_path}: {e}")
            # Return a dummy graph in case of error
            return Data(
                x=torch.tensor([[0.0] * self.feature_dim], dtype=torch.float),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                y=torch.tensor([0], dtype=torch.long)
            )
    
    @classmethod
    def from_config(cls, config_path, dataset_path, mapping_file=None):
        """
        Create a dataset from a configuration file.
        """
        # Load configuration
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            config = config_path
            logger.info("Using provided configuration dictionary")
        
        # Create dataset
        dataset = cls(dataset_path, mapping_file=mapping_file)
        logger.info(f"Created dataset with {len(dataset)} graphs")
        
        return dataset


class SplitDataset:
    """Dataset splitter for train/val/test splits."""
    
    def __init__(self, dataset, seed, train_split, val_split):
        # Store the full dataset
        self.dataset = dataset
        # Calculate indices for each split
        num_graphs = len(dataset)
        indices = list(range(num_graphs))
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Calculate number of samples in each split
        train_size = int(train_split * num_graphs) + 1
        val_size = int(val_split * num_graphs) + 1
        
        # Split indices
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:train_size + val_size]
        self.test_indices = indices[train_size + val_size:]
        
        # Create dataset subsets
        self.train_split = SubsetDataset(dataset, self.train_indices)
        self.val_split = SubsetDataset(dataset, self.val_indices)
        self.test_split = SubsetDataset(dataset, self.test_indices)
        
        logger.info(f"Split dataset: train={len(self.train_split)}, val={len(self.val_split)}, test={len(self.test_split)}")
        
        # Add index attributes for compatibility with AutoGL
        self.train_index = self.train_indices
        self.val_index = self.val_indices
        self.test_index = self.test_indices
        
        # Add data property for AutoGL compatibility
        self.data = dataset.data if hasattr(dataset, 'data') else None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class SubsetDataset:
    """Subset of a dataset."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.dataset[self.indices[idx]]
        else:
            return [self.dataset[self.indices[i]] for i in idx] 