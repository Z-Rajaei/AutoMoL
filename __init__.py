"""
AutoMol: Automated Machine Learning for Software Models

A framework for applying Graph Neural Networks to software model analysis tasks
including model classification, node classification, and link prediction.

Main Components:
- Element extraction from XMI/Ecore models
- Graph creation and transformation
- PyTorch Geometric dataset creation
- AutoGL-based classifiers and predictors
"""

__version__ = "1.0.0"
__author__ = "AutoMol Team"

from .core import ElementExtractor, GraphCreator
from .datasets import ModelDataset, ModelNodeDataset
from .classifiers import AutoModelClassifier, AutoModelElementClassifier, AutoModelLinkPredictor

__all__ = [
    "ElementExtractor",
    "GraphCreator", 
    "ModelDataset",
    "ModelLinkDataset",
    "NodeDataset",
    "AutoModelClassifier",
    "AutoModelElementClassifier", 
    "AutoModelLinkPredictor"
] 