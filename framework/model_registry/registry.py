"""
ModelRegistry: Register and manage AI/ML models, storing hashes and metadata on blockchain.
"""
import json
import hashlib
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime


class ModelRegistry:
    """
    Registry for AI/ML models with blockchain-based storage and versioning.
    
    This class handles model registration, versioning, and metadata storage
    on the blockchain for audit and provenance tracking.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the model registry.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain = blockchain_connector
        self._logger = logging.getLogger(__name__)
    
    def register_model(self, model_name: str, model_path: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Register a new AI/ML model on the blockchain.
        
        Args:
            model_name: Unique name for the model
            model_path: Path to the model file
            metadata: Model metadata (accuracy, version, training date, etc.)
            
        Returns:
            Model hash if successful, None if failed
        """
        try:
            # Validate inputs
            if not model_name or not model_path:
                raise ValueError("Model name and path are required")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Calculate model hash
            model_hash = self._calculate_file_hash(model_path)
            self._logger.info(f"Calculated hash for model {model_name}: {model_hash}")
            
            # Check if model already exists
            existing_model = self.get_model(model_name)
            if existing_model and existing_model.get('hash') == model_hash:
                self._logger.warning(f"Model {model_name} with same hash already exists")
                return model_hash
            
            # Prepare metadata
            full_metadata = {
                'name': model_name,
                'file_path': model_path,
                'file_size': os.path.getsize(model_path),
                'registration_date': datetime.now().isoformat(),
                'version': self._generate_version(model_name),
                **metadata
            }
            
            # Store model on blockchain
            response = self.blockchain.invoke_chaincode(
                "registerModel",
                [model_name, model_hash, json.dumps(full_metadata)]
            )
            
            if response.success:
                self._logger.info(f"Successfully registered model {model_name} with hash {model_hash}")
                return model_hash
            else:
                self._logger.error(f"Failed to register model on blockchain: {response.data}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error registering model {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model information from blockchain.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = self.blockchain.query_chaincode("getModel", [model_name])
            
            if response.success and response.data:
                return response.data
            else:
                self._logger.debug(f"Model {model_name} not found")
                return None
                
        except Exception as e:
            self._logger.error(f"Error retrieving model {model_name}: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self.blockchain.query_chaincode("getAllModels", [])
            
            if response.success and response.data:
                return response.data
            else:
                return []
                
        except Exception as e:
            self._logger.error(f"Error listing models: {e}")
            return []
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Base name of the model
            
        Returns:
            List of model versions
        """
        try:
            all_models = self.list_models()
            versions = [m for m in all_models if m.get('name', '').startswith(model_name)]
            
            # Sort by version number
            versions.sort(key=lambda x: x.get('version', '0'), reverse=True)
            return versions
            
        except Exception as e:
            self._logger.error(f"Error retrieving model versions for {model_name}: {e}")
            return []
    
    def verify_model_integrity(self, model_name: str, model_path: str) -> bool:
        """
        Verify that a model file matches its registered hash.
        
        Args:
            model_name: Name of the registered model
            model_path: Path to the model file to verify
            
        Returns:
            True if integrity check passes, False otherwise
        """
        try:
            # Get registered model info
            model_info = self.get_model(model_name)
            if not model_info:
                self._logger.error(f"Model {model_name} not found in registry")
                return False
            
            # Calculate current file hash
            if not os.path.exists(model_path):
                self._logger.error(f"Model file not found: {model_path}")
                return False
            
            current_hash = self._calculate_file_hash(model_path)
            registered_hash = model_info.get('hash')
            
            if current_hash == registered_hash:
                self._logger.info(f"Model {model_name} integrity verified")
                return True
            else:
                self._logger.warning(f"Model {model_name} integrity check failed. "
                                   f"Expected: {registered_hash}, Got: {current_hash}")
                return False
                
        except Exception as e:
            self._logger.error(f"Error verifying model integrity: {e}")
            return False
    
    def update_model_metadata(self, model_name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing model.
        
        Args:
            model_name: Name of the model to update
            metadata: New metadata to add/update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing model
            existing_model = self.get_model(model_name)
            if not existing_model:
                self._logger.error(f"Model {model_name} not found")
                return False
            
            # Merge metadata
            updated_metadata = {**existing_model.get('metadata', {}), **metadata}
            updated_metadata['last_updated'] = datetime.now().isoformat()
            
            # Re-register with updated metadata
            response = self.blockchain.invoke_chaincode(
                "registerModel",
                [model_name, existing_model['hash'], json.dumps(updated_metadata)]
            )
            
            if response.success:
                self._logger.info(f"Successfully updated metadata for model {model_name}")
                return True
            else:
                self._logger.error(f"Failed to update model metadata: {response.data}")
                return False
                
        except Exception as e:
            self._logger.error(f"Error updating model metadata: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _generate_version(self, model_name: str) -> str:
        """
        Generate version number for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version string (e.g., "1.0", "1.1", etc.)
        """
        try:
            versions = self.get_model_versions(model_name)
            if not versions:
                return "1.0"
            
            # Get highest version number
            max_version = 0.0
            for version_info in versions:
                try:
                    version_str = version_info.get('version', '0.0')
                    version_num = float(version_str)
                    max_version = max(max_version, version_num)
                except (ValueError, TypeError):
                    continue
            
            # Increment minor version
            new_version = max_version + 0.1
            return f"{new_version:.1f}"
            
        except Exception as e:
            self._logger.error(f"Error generating version: {e}")
            return "1.0"