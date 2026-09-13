"""
Link Prediction Dataset Creation Module

This module provides functions to create link prediction datasets from graphs.
"""

import os
import glob
import torch
import numpy as np
import logging
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops, remove_self_loops, degree
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
import networkx as nx

logger = logging.getLogger(__name__)

def create_link_dataset_from_graphs(graphs, combine_small_graphs=True, min_nodes=100):
    """
    Create a link prediction dataset from a list of PyG graphs.
    
    Parameters
    ----------
    graphs : list
        List of PyTorch Geometric Data objects
    combine_small_graphs : bool
        Whether to combine small graphs into larger ones
    min_nodes : int
        Minimum number of nodes for a graph to be used independently
        
    Returns
    -------
    LinkDataset
        Dataset suitable for link prediction
    """
    logger.info(f"Creating link dataset from {len(graphs)} graphs")
    
    # Filter valid graphs
    valid_graphs = []
    for i, graph in enumerate(graphs):
        if graph.num_nodes > 0 and graph.edge_index.shape[1] > 0:
            valid_graphs.append(graph)
        else:
            logger.warning(f"Skipping graph {i}: no nodes or edges")
    
    logger.info(f"Valid graphs: {len(valid_graphs)}")
    
    # Check if we need to combine graphs
    if combine_small_graphs:
        large_graphs = []
        small_graphs = []
        
        for graph in valid_graphs:
            if graph.num_nodes >= min_nodes:
                large_graphs.append(graph)
            else:
                small_graphs.append(graph)
        
        logger.info(f"Large graphs: {len(large_graphs)}, Small graphs: {len(small_graphs)}")
        
        # Combine small graphs if needed
        if len(small_graphs) > 0:
            combined = combine_graphs(small_graphs)
            if combined is not None:
                large_graphs.append(combined)
                logger.info(f"Combined {len(small_graphs)} small graphs into 1 large graph")
        
        final_graphs = large_graphs
    else:
        final_graphs = valid_graphs
    
    # Create dataset
    dataset = LinkDataset(final_graphs)
    logger.info(f"Created link dataset with {len(dataset)} graphs")
    
    return dataset

def combine_graphs(graphs):
    """
    Combine multiple graphs into one large graph.
    
    Parameters
    ----------
    graphs : list
        List of PyTorch Geometric Data objects
        
    Returns
    -------
    Data
        Combined graph
    """
    if len(graphs) == 0:
        return None
        
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    node_offset = 0
    
    for graph in graphs:
        # Node features
        if hasattr(graph, 'x') and graph.x is not None:
            all_x.append(graph.x)
        else:
            # Create dummy features if not present
            all_x.append(torch.ones(graph.num_nodes, 1))
        
        # Edge indices with offset
        edge_index = graph.edge_index + node_offset
        all_edge_index.append(edge_index)
        
        # Edge attributes if present
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            all_edge_attr.append(graph.edge_attr)
        
        node_offset += graph.num_nodes
    
    # Combine all
    combined_x = torch.cat(all_x, dim=0)
    combined_edge_index = torch.cat(all_edge_index, dim=1)
    
    combined_data = Data(x=combined_x, edge_index=combined_edge_index)
    
    if len(all_edge_attr) > 0:
        combined_data.edge_attr = torch.cat(all_edge_attr, dim=0)
    
    logger.info(f"Combined graph: {combined_data.num_nodes} nodes, {combined_data.edge_index.shape[1]} edges")
    
    return combined_data

class LinkDataset(InMemoryDataset):
    """
    In-memory dataset for link prediction.
    """
    def __init__(self, data_list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list
        
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def get_data(self):
        """For compatibility with AutoGL."""
        return self.data_list

class ModelLinkDataset(InMemoryDataset):
    """
    A PyTorch Geometric Dataset for link prediction on model graphs.
    This dataset creates a synthetic graph suitable for link prediction with meaningful structure.
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, logger=None):
        """Initialize the link prediction dataset."""
        self.root = root
        self.pyg_dir = os.path.join(root, 'pyg_torch_data')
        self.logger = logger or logging.getLogger(__name__)
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Load all data into memory
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        """Return raw file names."""
        pattern = os.path.join(self.pyg_dir, '*.pt')
        files = glob.glob(pattern)
        return [os.path.basename(f) for f in files] if files else ['dummy.pt']
        
    @property
    def processed_file_names(self):
        """Return processed file names."""
        return ['link_prediction_data.pt']
        
    def download(self):
        """Download is not needed as we create synthetic data."""
        pass
        
    def process(self):
        """Create a synthetic graph suitable for link prediction."""
        self.logger.info("Creating synthetic graph for link prediction")
        
        # Create a large synthetic graph with community structure
        # This will be more suitable for link prediction
        
        # Parameters for synthetic graph
        num_communities = 8
        nodes_per_community = 100
        p_intra = 0.15  # Probability of edge within community
        p_inter = 0.01  # Probability of edge between communities
        
        # Create stochastic block model graph
        sizes = [nodes_per_community] * num_communities
        p_matrix = np.full((num_communities, num_communities), p_inter)
        np.fill_diagonal(p_matrix, p_intra)
        
        # Use NetworkX to create the graph
        G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
        
        # Convert to PyG format
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_index = to_undirected(edge_index)
        
        # Create node features based on community structure and degree
        num_nodes = G.number_of_nodes()
        
        # Feature 1: Community assignment (one-hot encoded)
        community_features = torch.zeros(num_nodes, num_communities)
        for i, community_nodes in enumerate(G.graph['partition']):
            for node in community_nodes:
                community_features[node, i] = 1.0
        
        # Feature 2: Node degree (normalized)
        degrees = degree(edge_index[0], num_nodes=num_nodes)
        degree_features = degrees.view(-1, 1) / degrees.max()
        
        # Feature 3: Random features for diversity
        random_features = torch.randn(num_nodes, 16)
        
        # Combine all features
        x = torch.cat([community_features, degree_features, random_features], dim=1)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        
        # Add some graph-level statistics as additional features
        clustering_coef = nx.average_clustering(G)
        data.clustering = torch.tensor([clustering_coef])
        
        self.logger.info(f"Created synthetic graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        self.logger.info(f"Node features shape: {data.x.shape}")
        
        # Apply RandomLinkSplit for proper train/val/test split
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
            split_labels=True
        )
        
        # This creates train_data, val_data, test_data with proper positive/negative edges
        train_data, val_data, test_data = transform(data)
        
        # Log split information - use the correct attribute names
        if hasattr(train_data, 'pos_edge_label_index'):
            self.logger.info(f"Train edges: {train_data.pos_edge_label_index.size(1)} positive")
        if hasattr(val_data, 'pos_edge_label_index'):
            self.logger.info(f"Val edges: {val_data.pos_edge_label_index.size(1)} positive")
        if hasattr(test_data, 'pos_edge_label_index'):
            self.logger.info(f"Test edges: {test_data.pos_edge_label_index.size(1)} positive")
        
        # For AutoGL compatibility, we need to add edge_label_index and edge_label
        # which combines positive and negative edges
        for data_split in [train_data, val_data, test_data]:
            if hasattr(data_split, 'pos_edge_label_index') and hasattr(data_split, 'neg_edge_label_index'):
                # Combine positive and negative edges
                data_split.edge_label_index = torch.cat([
                    data_split.pos_edge_label_index,
                    data_split.neg_edge_label_index
                ], dim=1)
                
                # Create labels (1 for positive, 0 for negative)
                num_pos = data_split.pos_edge_label_index.size(1)
                num_neg = data_split.neg_edge_label_index.size(1)
                data_split.edge_label = torch.cat([
                    torch.ones(num_pos),
                    torch.zeros(num_neg)
                ])
        
        # Store all three splits
        data_list = [train_data, val_data, test_data]
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        self.logger.info("Link prediction dataset processing complete")
        
    def len(self):
        """Return the number of graphs (we have train/val/test splits)."""
        return 3  # train, val, test
        
    def get(self, idx):
        """Get a data split (0=train, 1=val, 2=test)."""
        if idx >= 3:
            # If requesting beyond our 3 splits, return the full training data
            return super().get(0)
        return super().get(idx) 