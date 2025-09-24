"""
Core components for AutoMol framework.

Contains the fundamental classes for processing software models:
- ElementExtractor: Extracts elements from XMI/Ecore models
- GraphCreator: Creates PyTorch Geometric graphs from extracted elements
"""

from .element_extractor import ElementExtractor
from .graph_creator import GraphCreator

__all__ = ["ElementExtractor", "GraphCreator"] 