"""
Element Extractor for Software Models

This module provides functionality to extract elements from XMI and Ecore model files
and convert them to JSON format for further processing.
"""

import os
import json
import yaml
import logging
import traceback
import glob
import xml.etree.ElementTree as ET
from pyecore.resources import ResourceSet, URI
import networkx as nx
from pathlib import Path

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ElementExtractor:
    """
    A class for extracting model elements from XMI and Ecore files and saving them as JSON.
    
    This class handles the first stage of the AutoMol pipeline: converting software models
    from their native formats (XMI/Ecore) into a standardized JSON representation that
    preserves the graph structure and element attributes.
    """
    
    def __init__(self, config_file, debug=False):
        """
        Initialize the extractor with a configuration file.
        
        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file containing extraction settings
        debug : bool, optional
            Enable debug logging, by default False
        """
        self.config_file = config_file
        self.debug = debug
        
        # Configure logger
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Load configuration
        logger.info(f"Starting extraction with configuration: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_file}")
        
        # Set up paths
        self.input_format = self.config.get('input', {}).get('format', 'xmi')
        self.metamodel_path = self.config.get('input', {}).get('metamodelpath', '')
        self.models_path = self.config.get('input', {}).get('modelspath', '')
        
        # Create required directories
        self.model_cache_dir = os.path.join(self.models_path, 'model_cache')
        self.pyg_data_dir = os.path.join(self.models_path, 'pyg_torch_data')
        
        os.makedirs(self.model_cache_dir, exist_ok=True)
        logger.info(f"Cache directory created: {self.model_cache_dir}")
        
        os.makedirs(self.pyg_data_dir, exist_ok=True)
        logger.info(f"PyG output directory created: {self.pyg_data_dir}")
        
        # Create ResourceSet for loading Ecore metamodels
        self.rset = ResourceSet()
        
        # Load metamodel
        self.load_metamodel()
    
    def load_metamodel(self):
        """Load the Ecore metamodel if specified in configuration."""
        if 'input' in self.config and 'metamodelpath' in self.config['input']:
            metamodel_path = self.config['input']['metamodelpath']
            if metamodel_path and os.path.exists(metamodel_path):
                logger.info(f"Loading metamodel from: {metamodel_path}")
                
                try:
                    resource = self.rset.get_resource(URI(metamodel_path))
                    mm_root = resource.contents[0]
                    self.rset.metamodel_registry[mm_root.nsURI] = mm_root
                    logger.info(f"Metamodel registered with URI: {mm_root.nsURI}")
                except Exception as e:
                    logger.error(f"Error loading metamodel: {e}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"Metamodel path not found or empty: {metamodel_path}")
    
    def get_model_files(self, limit=None):
        """
        Get all model files from the specified directory.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of files to process, by default None (no limit)
            
        Returns
        -------
        list
            List of model file paths
        """
        if not self.models_path:
            logger.warning("Model path not specified in configuration")
            return []
        
        model_files = []
        model_format = self.config.get('input', {}).get('format', 'ecore')
        
        if os.path.isdir(self.models_path):
            # Search for files with the specified extension
            if model_format.lower() == 'ecore':
                pattern = "*.ecore"
            elif model_format.lower() == 'xmi':
                pattern = "*.xmi"
            else:
                pattern = f"*.{model_format}"
            
            # Search in main directory
            model_files = glob.glob(os.path.join(self.models_path, pattern))
            logger.info(f"Found {len(model_files)} {pattern} files")
            
            # If no files found, search in Models subdirectory
            if not model_files:
                models_dir = os.path.join(self.models_path, 'Models')
                if os.path.exists(models_dir):
                    model_files = glob.glob(os.path.join(models_dir, pattern))
                    logger.info(f"Found {len(model_files)} {pattern} files in Models directory")
        elif os.path.isfile(self.models_path):
            model_files = [self.models_path]
        
        # Limit the number of files if specified
        if limit and isinstance(limit, int) and limit > 0:
            model_files = model_files[:limit]
        
        return model_files
    
    def extract_all_models(self, limit=None):
        """
        Extract all models and save them to cache.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of models to process, by default None
            
        Returns
        -------
        int
            Number of successfully processed models
        """
        model_files = self.get_model_files(limit)
        logger.info(f"Found {len(model_files)} model files in total")
        
        processed_count = 0
        
        for i, model_file in enumerate(model_files):
            logger.info(f"Processing model {i+1}/{len(model_files)}: {model_file}")
            
            # Extract graph from model
            if self.extract_model(model_file) is not None:
                processed_count += 1
        
        logger.info(f"Successfully processed {processed_count}/{len(model_files)} models")
        return processed_count
    
    def is_cached(self, model_path):
        """
        Check if a model exists in cache.
        
        Parameters
        ----------
        model_path : str
            Path to the model file
            
        Returns
        -------
        bool
            True if model is cached, False otherwise
        """
        cache_file = os.path.join(self.model_cache_dir, os.path.basename(model_path) + '.json')
        return os.path.exists(cache_file)
    
    def load_from_cache(self, model_path):
        """
        Load model data from cache.
        
        Parameters
        ----------
        model_path : str
            Path to the original model file
            
        Returns
        -------
        dict or None
            Cached model data or None if loading fails
        """
        cache_file = os.path.join(self.model_cache_dir, os.path.basename(model_path) + '.json')
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Model data loaded from cache: {cache_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None
    
    def save_to_cache(self, G, model_path):
        """
        Save graph to cache file.
        
        Parameters
        ----------
        G : networkx.Graph
            The graph to save
        model_path : str
            Path to the original model file
            
        Returns
        -------
        dict or None
            Model data in JSON format or None if saving fails
        """
        try:
            # Convert graph to JSON
            model_data = self.graph_to_json(G, model_path)
            
            # Save to cache file
            cache_file = os.path.join(self.model_cache_dir, os.path.basename(model_path) + '.json')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Graph saved to cache: {cache_file}")
            return model_data
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def extract_model(self, model_path):
        """
        Extract graph from model file.
        
        Parameters
        ----------
        model_path : str
            Path to the model file
            
        Returns
        -------
        networkx.Graph or None
            Extracted graph or None if extraction fails
        """
        try:
            # Use direct XML parser for loading model
            logger.debug(f"Loading model from path: {model_path}")
            
            # Load XML file
            tree = ET.parse(model_path)
            root = tree.getroot()
            logger.debug(f"XML file loaded successfully")
            
            # Create graph
            G = nx.MultiDiGraph()
            node_id_counter = 0
            node_map = {}
            
            # Recursive function to process elements - store raw information without applying filters
            def process_element(element, parent_id=None):
                nonlocal node_id_counter
                
                # Create an ID for this element
                current_id = node_id_counter
                node_id_counter += 1
                
                # Extract element type from tag name
                element_tag = element.tag
                if '}' in element_tag:
                    element_type = element_tag.split('}')[-1]  # Remove XML namespace
                else:
                    element_type = element_tag
                
                logger.debug(f"Processing element {element_type} with ID {current_id}")
                
                # Extract attributes - store all attributes without filtering
                attributes = {}
                for attr_name, attr_value in element.attrib.items():
                    # Remove XML namespace from attribute name
                    clean_attr_name = attr_name.split('}')[-1] if '}' in attr_name else attr_name
                    attributes[clean_attr_name] = attr_value
                
                # Add node to graph with raw information
                G.add_node(current_id, type=element_type, attributes=attributes)
                logger.debug(f"Node {current_id} added to graph with raw attributes")
                
                # Store ID for future references
                if 'name' in attributes:
                    node_map[attributes['name']] = current_id
                
                # If has parent, create an edge
                if parent_id is not None:
                    # Use target node type as edge type
                    G.add_edge(parent_id, current_id, type=element_type)
                
                # Process children
                for child in element:
                    process_element(child, current_id)
                
                return current_id
            
            # Start processing from root
            root_id = process_element(root)
            
            # Display graph statistics
            logger.info(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            
            # Save graph to cache
            self.save_to_cache(G, model_path)
            
            return G
        
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def graph_to_json(self, G, model_path):
        """
        Convert NetworkX graph to JSON format.
        
        Parameters
        ----------
        G : networkx.Graph
            The graph to convert
        model_path : str
            Path to the original model file
            
        Returns
        -------
        dict
            Graph data in JSON format
        """
        output_data = {
            "model_info": {
                "name": os.path.basename(model_path),
                "path": model_path,
                "type": self.input_format
            },
            "nodes": [],
            "edges": [],
            "node_types": {},
            "edge_types": {},
            "stats": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "node_type_count": 0,
                "edge_type_count": 0
            }
        }
        
        # Process nodes
        node_types = {}
        for node_id, node_data in G.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            
            # Count node types
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types[node_type] = 1
            
            # Create node information for JSON output
            node_info = {
                "id": str(node_id),
                "type": node_type,
                "attributes": node_data.get('attributes', {})
            }
            
            output_data["nodes"].append(node_info)
        
        # Process edges
        edge_types = {}
        for source, target, edge_data in G.edges(data=True):
            edge_type = edge_data.get('type', 'Unknown')
            
            # Count edge types
            if edge_type in edge_types:
                edge_types[edge_type] += 1
            else:
                edge_types[edge_type] = 1
            
            # Create edge information for JSON output
            edge_info = {
                "source": str(source),
                "target": str(target),
                "type": edge_type
            }
            
            output_data["edges"].append(edge_info)
        
        # Add statistics to output
        output_data["node_types"] = node_types
        output_data["edge_types"] = edge_types
        output_data["stats"]["node_type_count"] = len(node_types)
        output_data["stats"]["edge_type_count"] = len(edge_types)
        
        return output_data


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract model elements from XMI and Ecore files')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of models to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    extractor = ElementExtractor(args.config, debug=args.debug)
    extractor.extract_all_models(limit=args.limit)

if __name__ == '__main__':
    main() 