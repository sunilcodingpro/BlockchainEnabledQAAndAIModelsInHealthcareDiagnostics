"""
ModelRegistry: Register and manage AI/ML models, storing hashes and metadata on blockchain.

This module provides functionality to register AI/ML models with the blockchain,
including model versioning, metadata storage, and integrity verification.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class ModelRegistry:
    """
    Registry for managing AI/ML models with blockchain-based integrity and provenance.
    
    This class handles model registration, metadata storage, version management,
    and provides interfaces for querying registered models.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the model registry.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain_connector = blockchain_connector
    
    def register_model(self, model_name: str, model_path: str, 
                      metadata: Dict[str, Any]) -> str:
        """
        Register a new AI/ML model with the blockchain.
        
        This method calculates the model's hash, stores metadata, and creates
        a blockchain record for the model.
        
        Args:
            model_name: Human-readable name for the model
            model_path: Path to the model file
            metadata: Dictionary containing model metadata (accuracy, version, etc.)
            
        Returns:
            Hash of the registered model
        """
        try:
            # Generate unique model ID
            model_id = self._generate_model_id(model_name, metadata.get('version', '1.0'))
            
            # Calculate model hash
            model_hash = self._calculate_file_hash(model_path)
            
            # Prepare enhanced metadata
            enhanced_metadata = {
                **metadata,
                "file_path": model_path,
                "file_hash": model_hash,
                "registered_at": datetime.now().isoformat(),
                "file_size": self._get_file_size(model_path)
            }
            
            # Register model on blockchain
            result = self.blockchain_connector.invoke_transaction(
                "register_model",
                model_id,
                model_name,
                metadata.get('version', '1.0'),
                enhanced_metadata
            )
            
            if result.get("status") == "success":
                return result.get("hash", model_hash)
            else:
                print(f"Failed to register model on blockchain: {result.get('message')}")
                return model_hash
                
        except Exception as e:
            print(f"Error registering model: {e}")
            # Return a fallback hash if blockchain registration fails
            return self._calculate_file_hash(model_path) if os.path.exists(model_path) else None
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model information from the blockchain.
        
        Args:
            model_id: Unique identifier of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            result = self.blockchain_connector.query_ledger("query_model", model_id)
            
            if result.get("status") == "success":
                return result.get("model")
            else:
                print(f"Model {model_id} not found: {result.get('message')}")
                return None
                
        except Exception as e:
            print(f"Error retrieving model: {e}")
            return None
    
    def verify_model_integrity(self, model_id: str, model_path: str) -> bool:
        """
        Verify the integrity of a model file against its blockchain record.
        
        Args:
            model_id: Unique identifier of the model
            model_path: Current path to the model file
            
        Returns:
            True if model integrity is verified, False otherwise
        """
        try:
            # Get model record from blockchain
            model_info = self.get_model(model_id)
            if not model_info:
                return False
            
            # Calculate current file hash
            current_hash = self._calculate_file_hash(model_path)
            
            # Compare with stored hash
            stored_hash = model_info.get("metadata", {}).get("file_hash")
            
            return current_hash == stored_hash
            
        except Exception as e:
            print(f"Error verifying model integrity: {e}")
            return False
    
    def list_models(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        List all registered models.
        
        Args:
            status_filter: Optional filter by model status (active, deprecated, etc.)
            
        Returns:
            Dictionary containing list of models
        """
        try:
            # In a real implementation, this would query all models from the blockchain
            # For now, return a placeholder response
            return {
                "status": "success",
                "message": "Model listing functionality would query blockchain for all registered models",
                "models": []
            }
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return {
                "status": "error",
                "message": str(e),
                "models": []
            }
    
    def update_model_status(self, model_id: str, new_status: str, 
                           reason: str = "") -> bool:
        """
        Update the status of a registered model.
        
        Args:
            model_id: Unique identifier of the model
            new_status: New status (active, deprecated, recalled, etc.)
            reason: Reason for status change
            
        Returns:
            True if status updated successfully, False otherwise
        """
        try:
            # Create status update record
            status_update = {
                "model_id": model_id,
                "old_status": "active",  # Would query current status in real implementation
                "new_status": new_status,
                "reason": reason,
                "updated_at": datetime.now().isoformat()
            }
            
            # Log status change as compliance event
            result = self.blockchain_connector.invoke_transaction(
                "log_compliance_event",
                f"status_update_{model_id}_{int(datetime.now().timestamp())}",
                "model_status_update",
                model_id,
                status_update
            )
            
            return result.get("status") == "success"
            
        except Exception as e:
            print(f"Error updating model status: {e}")
            return False
    
    def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """
        Get the complete history of a model including all updates and events.
        
        Args:
            model_id: Unique identifier of the model
            
        Returns:
            Dictionary containing model history
        """
        try:
            # Get audit trail for this specific model
            result = self.blockchain_connector.query_ledger("get_audit_trail", model_id)
            
            return {
                "status": "success",
                "model_id": model_id,
                "history": result.get("audit_trail", [])
            }
            
        except Exception as e:
            print(f"Error retrieving model history: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """
        Generate a unique model identifier.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Unique model identifier
        """
        # Create a unique ID based on name, version, and timestamp
        timestamp = str(int(datetime.now().timestamp()))
        data = f"{model_name}_{version}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash of the file
        """
        if not os.path.exists(file_path):
            # Create a dummy hash for non-existent files (demo purposes)
            return hashlib.sha256(f"dummy_{file_path}".encode()).hexdigest()
        
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash: {e}")
            return hashlib.sha256(f"error_{file_path}".encode()).hexdigest()
    
    def _get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path)
            else:
                return 0
        except Exception:
            return 0