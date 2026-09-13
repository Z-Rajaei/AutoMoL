"""
Graph Creator for AutoMol Framework

This module handles the second stage of the AutoMol pipeline: converting extracted
JSON model data into PyTorch Geometric graphs with proper feature encoding and
filtering based on configuration.
"""

import os
import json
import yaml
import logging
import torch
import numpy as np
import re
from pathlib import Path
from torch_geometric.data import Data

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GraphCreator:
    """
    A class for creating PyTorch Geometric graphs from JSON model data.
    
    This class applies transformations, filters, and feature encodings specified
    in the configuration file to convert raw model data into structured graphs
    suitable for machine learning tasks.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the GraphCreator with configuration from a file.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file, by default None
        """
        self.config = None
        self.model_cache_dir = None
        self.pyg_data_dir = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a file.
        
        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
            
            # Set model cache directory
            self.model_cache_dir = os.path.join(self.config['input']['modelspath'], 'model_cache')
            logger.info(f"Model cache path: {self.model_cache_dir}")
            
            # Set PyG data directory
            if 'output' in self.config and 'path' in self.config['output']:
                self.pyg_data_dir = self.config['output']['path']
            else:
                self.pyg_data_dir = os.path.join(self.config['input']['modelspath'], 'pyg_torch_data')
            logger.info(f"PyG data path: {self.pyg_data_dir}")
            
            # Create directories if they don't exist
            os.makedirs(self.model_cache_dir, exist_ok=True)
            os.makedirs(self.pyg_data_dir, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error in configuration: {e}")
            raise
    
    def transform_all_models(self):
        """
        Apply transformations and filters from config to all JSON files in cache directory.
        
        Returns
        -------
        int
            Number of successfully transformed files
        """
        # Check if cache directory exists
        if not os.path.exists(self.model_cache_dir):
            logger.error(f"Cache directory {self.model_cache_dir} does not exist")
            return 0
        
        # Find all JSON files in cache directory
        json_files = [f for f in os.listdir(self.model_cache_dir) if f.endswith('.json')]
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Apply transformations to each file
        successful_count = 0
        for json_file in json_files:
            json_path = os.path.join(self.model_cache_dir, json_file)
            output_path = os.path.join(self.pyg_data_dir, f"transformed_{json_file}")
            if self.transform_model(json_path, output_path) is not None:
                successful_count += 1
        
        logger.info(f"Applied transformations to {successful_count}/{len(json_files)} files")
        return successful_count
    
    def transform_model(self, json_path, output_path=None):
        """
        Apply transformations and filters from config to a JSON file.
        
        Parameters
        ----------
        json_path : str
            Path to the input JSON file
        output_path : str, optional
            Path for the transformed output file, by default None
            
        Returns
        -------
        dict or None
            Transformed model data or None if transformation fails
        """
        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                logger.info(f"Data loaded from {json_path}")
            
            # Apply transformations and filters
            transformed_data = self._apply_transformations(model_data)
            
            # Determine output path
            if output_path is None:
                base_name = os.path.basename(json_path)
                output_path = os.path.join(self.pyg_data_dir, f"transformed_{base_name}")
            
            # Save transformed data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Transformed data saved to {output_path}")
            
            return transformed_data
        
        except Exception as e:
            logger.error(f"Error in transformation: {e}")
            return None
    
    def _apply_transformations(self, model_data):
        """
        Apply transformations and filters from config to model data.
        
        Parameters
        ----------
        model_data : dict
            Raw model data from JSON
            
        Returns
        -------
        dict
            Transformed model data
        """
        # Create a copy of the data to avoid modifying the original
        transformed_data = {
            "nodes": [],
            "edges": model_data.get("edges", []),
            "stats": model_data.get("stats", {})
        }
        
        # If no config, return original data
        if not self.config:
            return model_data
        
        # Load filter settings from config file
        config_adaptations = self.config.get('configuration', {}).get('adaptations', {})
        metamodels = config_adaptations.get('metamodels', [])
        
        # Global settings for all classes
        global_include_all_attrs = config_adaptations.get('includeAllAttributes', False)
        global_exclude_all_attrs = config_adaptations.get('excludeAllAttributes', False)
        global_include_attrs = config_adaptations.get('includeAttributes', [])
        global_exclude_attrs = config_adaptations.get('excludeAttributes', [])
        global_include_classes = config_adaptations.get('includeClasses', [])
        global_exclude_classes = config_adaptations.get('excludeClasses', [])
        
        # Create a dictionary for quick access to class configurations
        class_configs = {}
        
        for metamodel in metamodels:
            for package in metamodel.get('packages', []):
                for class_config in package.get('classes', []):
                    class_name = class_config.get('class', class_config.get('name', ''))
                    if class_name:
                        class_configs[class_name] = class_config
        
        # Process nodes
        for node in model_data.get("nodes", []):
            node_id = node.get("id")
            node_type = node.get("type")
            attributes = node.get("attributes", {}).copy()
            
            # Check if this class should be included
            if global_exclude_classes and node_type in global_exclude_classes:
                continue
                
            if global_include_classes and len(global_include_classes) > 0 and node_type not in global_include_classes:
                continue
            
            # Apply class-specific settings
            if node_type in class_configs:
                class_config = class_configs[node_type]
                
                # Check class renaming
                element_renaming = class_config.get('renaming', '')
                if element_renaming and element_renaming.strip():
                    node_type = element_renaming
                
                # Apply attribute filtering
                include_all_attrs = class_config.get('includeAllAttributes', global_include_all_attrs)
                exclude_all_attrs = class_config.get('excludeAllAttributes', global_exclude_all_attrs)
                include_attrs = class_config.get('include', global_include_attrs)
                exclude_attrs = class_config.get('exclude', global_exclude_attrs)
                
                if exclude_all_attrs:
                    attributes = {}
                elif not include_all_attrs and include_attrs:
                    filtered_attrs = {}
                    for attr_name, attr_value in attributes.items():
                        if attr_name in include_attrs:
                            filtered_attrs[attr_name] = attr_value
                    attributes = filtered_attrs
                
                # Remove excluded attributes
                for attr_name in exclude_attrs:
                    if attr_name in attributes:
                        del attributes[attr_name]
            
            # Create transformed node
            transformed_node = {
                "id": node_id,
                "type": node_type,
                "attributes": attributes
            }
            
            transformed_data["nodes"].append(transformed_node)
        
        return transformed_data
    
    def convert_to_pyg(self, json_path):
        """
        Convert JSON file to PyTorch Geometric graph.
        
        Parameters
        ----------
        json_path : str
            Path to the transformed JSON file
            
        Returns
        -------
        torch_geometric.data.Data or None
            PyTorch Geometric graph object or None if conversion fails
        """
        try:
            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Extract nodes and create mappings
            node_id_to_index = {}
            node_features = []
            node_types = []
            
            # Create simple features (one-hot encoding for node types)
            unique_types = list(set(node.get("type", "Unknown") for node in model_data.get("nodes", [])))
            type_to_index = {t: i for i, t in enumerate(unique_types)}
            
            for i, node in enumerate(model_data.get("nodes", [])):
                node_id = node.get("id")
                node_type = node.get("type", "Unknown")
                
                node_id_to_index[node_id] = i
                node_types.append(node_type)
                
                # Create one-hot encoding for node type
                type_encoding = [0.0] * len(unique_types)
                if node_type in type_to_index:
                    type_encoding[type_to_index[node_type]] = 1.0
                
                # Add attribute features (simplified)
                attributes = node.get("attributes", {})
                attr_features = []
                for attr_name in ["name", "type"]:  # Common attributes
                    if attr_name in attributes:
                        # Simple hash-based encoding
                        attr_hash = hash(str(attributes[attr_name])) % 100
                        attr_features.append(float(attr_hash) / 100.0)
                    else:
                        attr_features.append(0.0)
                
                node_features.append(type_encoding + attr_features)
            
            # Create edge index
            edge_index = []
            for edge in model_data.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                
                if source in node_id_to_index and target in node_id_to_index:
                    source_idx = node_id_to_index[source]
                    target_idx = node_id_to_index[target]
                    edge_index.append([source_idx, target_idx])
            
            # Convert to tensors
            if node_features:
                x = torch.tensor(node_features, dtype=torch.float)
            else:
                x = torch.zeros((1, 1), dtype=torch.float)
            
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index)
            return data
        
        except Exception as e:
            logger.error(f"Error converting to PyG: {e}")
            return None
    
    def convert_all_models(self, encoder=None, out_dir=None):
        """
        Convert all transformed JSON files to PyTorch Geometric format.

        Parameters
        ----------
        encoder : FeatureEncoder, optional
            When given, graphs are encoded with ``encoder.encode_graph``
            (global vocabularies, DSL-declared encodings) instead of the
            legacy per-graph ``convert_to_pyg``.
        out_dir : str, optional
            Output directory for the .pt files; defaults to
            ``self.pyg_data_dir``.

        Returns
        -------
        int
            Number of successfully converted files
        """
        if not os.path.exists(self.pyg_data_dir):
            logger.error(f"PyG data directory {self.pyg_data_dir} does not exist")
            return 0

        if out_dir is None:
            out_dir = self.pyg_data_dir
        os.makedirs(out_dir, exist_ok=True)

        # Find all transformed JSON files
        json_files = [f for f in os.listdir(self.pyg_data_dir)
                     if f.startswith('transformed_') and f.endswith('.json')]
        logger.info(f"Found {len(json_files)} transformed JSON files")

        # Convert each file to PyG format
        successful_count = 0
        for json_file in json_files:
            json_path = os.path.join(self.pyg_data_dir, json_file)
            if encoder is not None:
                pyg_data = encoder.encode_graph(json_path)
            else:
                pyg_data = self.convert_to_pyg(json_path)

            if pyg_data is not None:
                # Save PyG data
                output_path = os.path.join(
                    out_dir, os.path.basename(json_path).replace('.json', '.pt'))
                torch.save(pyg_data, output_path)
                logger.info(f"Saved PyG data to {output_path}")
                successful_count += 1

        logger.info(f"Converted {successful_count}/{len(json_files)} files to PyG format")
        return successful_count


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PyTorch Geometric graphs from JSON model data')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--transform-only', action='store_true', help='Only apply transformations, do not convert to PyG')
    
    args = parser.parse_args()
    
    graph_creator = GraphCreator(args.config)
    
    if args.transform_only:
        count = graph_creator.transform_all_models()
        logger.info(f"Transformed {count} models")
    else:
        # Full pipeline: transform and convert
        transform_count = graph_creator.transform_all_models()
        convert_count = graph_creator.convert_all_models()
        logger.info(f"Transformed {transform_count} models, converted {convert_count} to PyG format")

if __name__ == '__main__':
    main() 