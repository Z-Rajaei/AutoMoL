"""
Model Node Dataset for AutoMol Framework

This module provides PyTorch Geometric dataset for node classification on model graphs.
This is a generic dataset that can handle any PyG graphs for node classification.
"""

import os
import glob
import torch
import logging
import numpy as np
from torch_geometric.data import Dataset, Data, InMemoryDataset
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class ModelNodeDataset(InMemoryDataset):
    """
    A generic PyTorch Geometric Dataset for node classification.
    
    This dataset can load PyG graphs from a directory and prepare them for node classification.
    It expects graphs to have node features (x) and will automatically create labels if not present.
    """
    
    def __init__(self, root, num_classes=7, transform=None, pre_transform=None, pre_filter=None):
        """Initialize the dataset."""
        self.root = root
        self.pyg_dir = os.path.join(root, 'pyg_torch_data')
        self._num_classes = num_classes
        
        # Also check for raw PyG files directly in root
        if not os.path.exists(self.pyg_dir):
            self.pyg_dir = root
            
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        """Return raw file names."""
        # Look for any .pt files in the directory
        pattern = os.path.join(self.pyg_dir, '*.pt')
        files = glob.glob(pattern)
        
        if not files:
            # Also check for subdirectories
            pattern = os.path.join(self.pyg_dir, '**', '*.pt')
            files = glob.glob(pattern, recursive=True)
            
        return [os.path.basename(f) for f in files] if files else ['dummy.pt']
    
    @property
    def processed_file_names(self):
        """Return processed file names."""
        return ['node_classification_data.pt']
    
    def download(self):
        """Download is not needed."""
        pass
    
    def process(self):
        """Process the raw data and save."""
        data_list = []
        
        # Find all PyG files
        pyg_files = glob.glob(os.path.join(self.pyg_dir, '*.pt'))
        if not pyg_files:
            # Check subdirectories
            pyg_files = glob.glob(os.path.join(self.pyg_dir, '**', '*.pt'), recursive=True)
        
        logger.info(f"Found {len(pyg_files)} PyG files in {self.pyg_dir}")
        
        if len(pyg_files) == 0:
            raise ValueError(f"No PyG files found in {self.pyg_dir}. "
                           "Please ensure graphs have been saved as .pt files.")
        
        # Load all graphs
        all_graphs = []
        for file_path in pyg_files:
            try:
                # Load the file
                data = torch.load(file_path, weights_only=False)
                
                # Handle different formats
                if isinstance(data, list):
                    # If it's a list of graphs
                    for graph in data:
                        if self._validate_graph(graph):
                            all_graphs.append(graph)
                elif self._validate_graph_basic(data):
                    # Single graph - only check basic requirements
                    all_graphs.append(data)
                else:
                    logger.warning(f"Invalid data format in {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if len(all_graphs) == 0:
            raise ValueError("No valid graphs found. Graphs must have 'x' (features) and 'edge_index'.")
        
        logger.info(f"Loaded {len(all_graphs)} valid graphs")
        
        # Process based on number of graphs
        if len(all_graphs) == 1:
            # Single graph - typical for node classification
            graph = all_graphs[0]
            
            # Create labels if not present
            if not hasattr(graph, 'y') or graph.y is None:
                logger.info("No labels found. Creating labels using node clustering...")
                graph = self._create_node_labels(graph)
            else:
                logger.info(f"Labels already present with {graph.y.max().item() + 1} classes")
            
            # Ensure masks exist
            if not hasattr(graph, 'train_mask'):
                self._create_masks(graph)
                
            data_list = [graph]
            logger.info(f"Single graph with {graph.num_nodes} nodes, "
                       f"{graph.edge_index.size(1)} edges, "
                       f"{graph.y.max().item() + 1} classes")
            
        else:
            # Multiple graphs - need to decide strategy
            logger.info(f"Multiple graphs found. Combining for node classification...")
            
            # Option 1: Use each graph separately (for graph-level tasks)
            # Option 2: Combine into one large graph (for node-level tasks)
            # Here we combine them for node classification
            combined_data = self._combine_graphs_for_node_classification(all_graphs)
            data_list = [combined_data]
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        logger.info(f"Saved {len(data_list)} graphs for node classification")
    
    def _validate_graph_basic(self, graph):
        """Check if graph has basic requirements (features and structure)."""
        if not hasattr(graph, 'x') or graph.x is None:
            return False
        if not hasattr(graph, 'edge_index') or graph.edge_index is None:
            return False
        return True
    
    def _validate_graph(self, graph):
        """Check if graph has required attributes for node classification."""
        if not self._validate_graph_basic(graph):
            return False
        if not hasattr(graph, 'y') or graph.y is None:
            return False
        return True
    
    def _create_node_labels(self, data):
        """Create node labels using clustering on node features."""
        logger.info(f"Creating {self._num_classes} node labels using K-means clustering...")
        
        # Get node features
        features = data.x.numpy()
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self._num_classes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Convert to torch tensor
        data.y = torch.tensor(cluster_labels, dtype=torch.long)
        
        # Log cluster distribution
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        logger.info("Created cluster labels:")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Cluster {label}: {count} nodes")
        
        return data
    
    def _create_masks(self, data):
        """Create train/val/test masks for a graph."""
        num_nodes = data.num_nodes
        
        # Create random split
        indices = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        logger.info(f"Created masks: train={train_mask.sum()}, "
                   f"val={val_mask.sum()}, test={test_mask.sum()}")
    
    def _combine_graphs_for_node_classification(self, graphs):
        """Combine multiple graphs into one for node classification."""
        # Initialize lists to collect data
        all_x = []
        all_edge_index = []
        all_y = []
        node_offset = 0
        
        for graph in graphs:
            # Node features
            all_x.append(graph.x)
            
            # Edge indices with offset
            edge_index = graph.edge_index + node_offset
            all_edge_index.append(edge_index)
            
            # Node labels - create if not present
            if not hasattr(graph, 'y') or graph.y is None:
                logger.info(f"Creating labels for graph with {graph.num_nodes} nodes...")
                graph = self._create_node_labels(graph)
            
            all_y.append(graph.y)
            node_offset += graph.num_nodes
        
        # Combine all data
        combined_x = torch.cat(all_x, dim=0)
        combined_edge_index = torch.cat(all_edge_index, dim=1)
        combined_y = torch.cat(all_y, dim=0)
        
        # Create combined data object
        combined_data = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            y=combined_y
        )
        
        # Create masks
        self._create_masks(combined_data)
        
        logger.info(f"Combined {len(graphs)} graphs into one with "
                   f"{combined_data.num_nodes} nodes, "
                   f"{combined_edge_index.size(1)} edges")
        
        return combined_data
    
    def len(self):
        """Return number of graphs."""
        return len(self.slices['x']) - 1 if self.slices is not None else 1
    
    def get(self, idx):
        """Get graph by index."""
        return super().get(idx) 