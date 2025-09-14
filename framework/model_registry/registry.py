"""
ModelRegistry: Register and manage AI/ML models, storing hashes and metadata on blockchain.

Provides comprehensive model lifecycle management including:
- Model registration with blockchain verification
- Version control and metadata tracking
- Regulatory status management
- Performance monitoring and drift detection
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import os


class ModelRegistry:
    """
    Comprehensive AI/ML model registry with blockchain integration
    
    Manages the complete lifecycle of healthcare AI models including
    registration, versioning, compliance tracking, and performance monitoring.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize model registry with blockchain connector
        
        Args:
            blockchain_connector: HyperledgerConnector instance
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger(__name__)
        
        # Local cache for model metadata (optional optimization)
        self.model_cache = {}
        self.cache_ttl = timedelta(minutes=30)
    
    async def register_model(self, model_name: str, model_path: str, metadata: Dict[str, Any]) -> str:
        """
        Register a new AI/ML model in the blockchain registry
        
        Args:
            model_name: Unique identifier for the model
            model_path: Path to the model file (for integrity verification)
            metadata: Model metadata including accuracy, algorithm, validation metrics
            
        Returns:
            Model hash for verification and reference
        """
        try:
            self.logger.info(f"Registering model: {model_name}")
            
            # Validate inputs
            self._validate_model_metadata(metadata)
            
            # Calculate model integrity hash
            model_hash = await self._calculate_model_integrity(model_path)
            
            # Prepare enhanced metadata
            enhanced_metadata = self._prepare_model_metadata(model_name, metadata, model_hash)
            
            # Register in blockchain
            tx_id = await self.blockchain.register_model(model_name, model_path, enhanced_metadata)
            
            # Update local cache
            model_id = f"{model_name}_{metadata.get('version', '1.0')}"
            self._update_cache(model_id, enhanced_metadata)
            
            self.logger.info(f"Model {model_name} registered successfully with hash: {model_hash}")
            return model_hash
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {str(e)}")
            raise
    
    async def get_model(self, model_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by ID
        
        Args:
            model_id: Model identifier
            use_cache: Whether to use local cache for faster retrieval
            
        Returns:
            Model metadata dictionary or None if not found
        """
        try:
            # Check cache first if enabled
            if use_cache and model_id in self.model_cache:
                cache_entry = self.model_cache[model_id]
                if datetime.utcnow() - cache_entry['cached_at'] < self.cache_ttl:
                    return cache_entry['data']
            
            # Query blockchain
            model_data = await self.blockchain.get_model(model_id)
            
            # Update cache
            if model_data:
                self._update_cache(model_id, model_data)
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id}: {str(e)}")
            return None
    
    async def update_model_status(self, model_id: str, new_status: str, notes: str = "") -> bool:
        """
        Update model regulatory/approval status
        
        Args:
            model_id: Model identifier
            new_status: New status (pending, approved, rejected, deprecated)
            notes: Optional status change notes
            
        Returns:
            Success status
        """
        try:
            valid_statuses = ['pending', 'approved', 'conditionally_approved', 'rejected', 'deprecated', 'under_review']
            
            if new_status not in valid_statuses:
                raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
            
            # Update in blockchain
            tx_id = await self.blockchain.update_model_status(model_id, new_status, notes)
            
            # Invalidate cache
            if model_id in self.model_cache:
                del self.model_cache[model_id]
            
            self.logger.info(f"Model {model_id} status updated to {new_status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model status: {str(e)}")
            return False
    
    async def list_models(self, organization: str = "", status: str = "") -> List[Dict[str, Any]]:
        """
        List models with optional filtering
        
        Args:
            organization: Filter by organization
            status: Filter by regulatory status
            
        Returns:
            List of model metadata dictionaries
        """
        try:
            # This would query the blockchain using composite key indexes
            # For now, return mock data structure
            models = []
            
            # In production, this would use blockchain queries:
            # models = await self.blockchain.query_models_by_org(organization)
            # or await self.blockchain.query_models_by_status(status)
            
            self.logger.info(f"Listed {len(models)} models")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def get_model_performance_metrics(self, model_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for a model over specified period
        
        Args:
            model_id: Model identifier
            days: Number of days to look back
            
        Returns:
            Performance metrics including accuracy trends, usage stats
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get audit trail to analyze performance
            audit_data = await self.blockchain.get_audit_trail(
                model_id, 
                start_date.isoformat(), 
                end_date.isoformat()
            )
            
            # Analyze diagnostics for performance metrics
            diagnostics = audit_data.get('diagnostics', [])
            
            if not diagnostics:
                return {'error': 'No diagnostic data available for the specified period'}
            
            # Calculate metrics
            total_predictions = len(diagnostics)
            confidence_scores = [d.get('confidence_score', 0) for d in diagnostics]
            
            metrics = {
                'model_id': model_id,
                'period_days': days,
                'total_predictions': total_predictions,
                'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'min_confidence': min(confidence_scores) if confidence_scores else 0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0,
                'low_confidence_predictions': len([s for s in confidence_scores if s < 0.7]),
                'predictions_per_day': total_predictions / days,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Add drift detection
            drift_indicators = self._analyze_drift(diagnostics)
            metrics.update(drift_indicators)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def check_model_compliance(self, model_id: str) -> Dict[str, Any]:
        """
        Check model compliance status and generate recommendations
        
        Args:
            model_id: Model identifier
            
        Returns:
            Compliance status and recommendations
        """
        try:
            # Get model info
            model_data = await self.get_model(model_id)
            if not model_data:
                return {'error': 'Model not found'}
            
            # Get compliance report
            compliance_report = await self.blockchain.generate_compliance_report(model_id)
            
            # Analyze compliance
            compliance_score = compliance_report.get('summary', {}).get('compliance_score', 0)
            unresolved_events = compliance_report.get('summary', {}).get('unresolved_events', 0)
            critical_events = compliance_report.get('summary', {}).get('critical_events', 0)
            
            # Generate recommendations
            recommendations = []
            
            if compliance_score < 90:
                recommendations.append("Compliance score below 90%. Review and resolve outstanding issues.")
            
            if unresolved_events > 5:
                recommendations.append("High number of unresolved compliance events. Prioritize resolution.")
            
            if critical_events > 0:
                recommendations.append("Critical compliance events require immediate attention.")
            
            regulatory_status = model_data.get('regulatory_status', 'unknown')
            if regulatory_status == 'under_review':
                recommendations.append("Model is under regulatory review. Monitor status regularly.")
            
            return {
                'model_id': model_id,
                'compliance_score': compliance_score,
                'regulatory_status': regulatory_status,
                'unresolved_events': unresolved_events,
                'critical_events': critical_events,
                'recommendations': recommendations,
                'overall_status': self._determine_compliance_status(compliance_score, critical_events),
                'checked_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check model compliance: {str(e)}")
            return {'error': str(e)}
    
    # === Private Helper Methods ===
    
    def _validate_model_metadata(self, metadata: Dict[str, Any]):
        """Validate required model metadata fields"""
        required_fields = ['accuracy', 'algorithm']
        
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        if not isinstance(metadata['accuracy'], (int, float)) or not 0 <= metadata['accuracy'] <= 1:
            raise ValueError("Accuracy must be a number between 0 and 1")
    
    async def _calculate_model_integrity(self, model_path: str) -> str:
        """Calculate model file integrity hash"""
        try:
            if not os.path.exists(model_path):
                # For demo purposes, generate deterministic hash from path
                return hashlib.sha256(f"demo_model_{model_path}".encode()).hexdigest()
            
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Could not calculate model hash: {str(e)}")
            return hashlib.sha256(f"fallback_{model_path}_{time.time()}".encode()).hexdigest()
    
    def _prepare_model_metadata(self, model_name: str, metadata: Dict[str, Any], model_hash: str) -> Dict[str, Any]:
        """Prepare enhanced metadata for blockchain storage"""
        enhanced = {
            'name': model_name,
            'version': metadata.get('version', '1.0'),
            'algorithm': metadata['algorithm'],
            'accuracy': metadata['accuracy'],
            'date': metadata.get('date', datetime.utcnow().isoformat()),
            'model_hash': model_hash,
            'created_at': datetime.utcnow().isoformat(),
            'created_by': getattr(self.blockchain, 'user_name', 'unknown'),
            'organization': getattr(self.blockchain, 'org_name', 'unknown')
        }
        
        # Add optional fields
        optional_fields = ['metrics', 'training_dataset', 'validation_dataset', 'description', 'tags']
        for field in optional_fields:
            if field in metadata:
                enhanced[field] = metadata[field]
        
        return enhanced
    
    def _update_cache(self, model_id: str, model_data: Dict[str, Any]):
        """Update local model cache"""
        self.model_cache[model_id] = {
            'data': model_data,
            'cached_at': datetime.utcnow()
        }
        
        # Cleanup old cache entries (simple LRU)
        if len(self.model_cache) > 100:
            oldest_key = min(self.model_cache.keys(), 
                           key=lambda k: self.model_cache[k]['cached_at'])
            del self.model_cache[oldest_key]
    
    def _analyze_drift(self, diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze diagnostics for drift indicators"""
        if len(diagnostics) < 10:  # Need sufficient data for drift analysis
            return {'drift_status': 'insufficient_data'}
        
        # Sort by timestamp
        sorted_diagnostics = sorted(diagnostics, key=lambda d: d.get('timestamp', ''))
        
        # Split into recent vs historical
        split_point = len(sorted_diagnostics) // 2
        historical = sorted_diagnostics[:split_point]
        recent = sorted_diagnostics[split_point:]
        
        # Compare confidence distributions
        hist_confidence = [d.get('confidence_score', 0) for d in historical]
        recent_confidence = [d.get('confidence_score', 0) for d in recent]
        
        hist_avg = sum(hist_confidence) / len(hist_confidence) if hist_confidence else 0
        recent_avg = sum(recent_confidence) / len(recent_confidence) if recent_confidence else 0
        
        confidence_drift = abs(hist_avg - recent_avg)
        
        return {
            'drift_status': 'detected' if confidence_drift > 0.1 else 'stable',
            'confidence_drift': confidence_drift,
            'historical_avg_confidence': hist_avg,
            'recent_avg_confidence': recent_avg,
            'drift_severity': 'high' if confidence_drift > 0.2 else 'medium' if confidence_drift > 0.1 else 'low'
        }
    
    def _determine_compliance_status(self, compliance_score: float, critical_events: int) -> str:
        """Determine overall compliance status"""
        if critical_events > 0:
            return 'critical'
        elif compliance_score >= 95:
            return 'excellent'
        elif compliance_score >= 90:
            return 'good'
        elif compliance_score >= 75:
            return 'acceptable'
        else:
            return 'poor'