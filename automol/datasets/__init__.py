"""
Datasets module for AutoMol framework.

Contains PyTorch Geometric dataset classes for different tasks:
- ModelDataset: For model classification tasks
- ModelLinkDataset: For link prediction tasks  
- ModelNodeDataset: For node classification tasks
"""

from .model_dataset import ModelDataset
from .link_dataset import ModelLinkDataset, create_link_dataset_from_graphs, LinkDataset
from .model_node_dataset import ModelNodeDataset

__all__ = ["ModelDataset", "ModelLinkDataset", "ModelNodeDataset", 
           "create_link_dataset_from_graphs", "LinkDataset"] 