"""
AutoMol Classifiers Package

This package provides graph-based classifiers extending AutoGL functionality.
"""

from .model_classifier import AutoModelClassifier
from .element_classifier import AutoModelElementClassifier
from .link_predictor import AutoModelLinkPredictor

__all__ = [
    'AutoModelClassifier',
    'AutoModelElementClassifier', 
    'AutoModelLinkPredictor'
]