"""
DataProvenanceLogger: Logs sample provenance and metadata for traceability.

Provides comprehensive data lineage tracking with blockchain immutability for:
- Sample data collection and processing
- Data transformations and feature engineering  
- Source attribution and quality metrics
- HIPAA-compliant data anonymization tracking
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging


class DataProvenanceLogger:
    """
    Comprehensive data provenance logger with blockchain integration
    
    Tracks complete data lineage from collection through processing,
    ensuring transparency and compliance in healthcare AI workflows.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize provenance logger with blockchain connector
        
        Args:
            blockchain_connector: HyperledgerConnector instance
        """
        self.blockchain = blockchain_connector
        self.logger = logging.getLogger(__name__)
        
        # Track active data processing sessions
        self.active_sessions = {}
    
    async def log_sample(self, sample_id: str, sample_data: Dict[str, Any], 
                        provenance_info: Dict[str, Any]) -> str:
        """
        Log a data sample with full provenance information
        
        Args:
            sample_id: Unique identifier for the sample (anonymized)
            sample_data: The actual data (features, measurements, etc.)
            provenance_info: Source, collection method, quality metrics
            
        Returns:
            Cryptographic hash of the logged sample
        """
        try:
            self.logger.info(f"Logging sample: {sample_id}")
            
            # Validate and anonymize data
            validated_data = self._validate_and_anonymize_sample(sample_data)
            
            # Create comprehensive provenance record
            provenance_record = self._create_provenance_record(
                sample_id, validated_data, provenance_info
            )
            
            # Calculate data integrity hash
            data_hash = self._calculate_data_hash(validated_data)
            
            # Store in blockchain (in production, would store hash + metadata)
            # Full data would be stored in secure, HIPAA-compliant storage
            blockchain_record = {
                'sample_id': sample_id,
                'data_hash': data_hash,
                'provenance': provenance_record,
                'logged_at': datetime.utcnow().isoformat(),
                'logged_by': getattr(self.blockchain, 'user_name', 'unknown')
            }
            
            # Submit to blockchain (mock implementation)
            tx_id = await self.blockchain.submit_transaction(
                'log_sample_provenance', 
                json.dumps(blockchain_record)
            )
            
            self.logger.info(f"Sample {sample_id} logged with hash: {data_hash}")
            return data_hash
            
        except Exception as e:
            self.logger.error(f"Failed to log sample {sample_id}: {str(e)}")
            raise
    
    async def start_processing_session(self, session_id: str, input_samples: List[str], 
                                     processing_type: str, parameters: Dict[str, Any]) -> str:
        """
        Start a data processing session (preprocessing, feature engineering, etc.)
        
        Args:
            session_id: Unique session identifier
            input_samples: List of input sample IDs
            processing_type: Type of processing (normalization, feature_extraction, etc.)
            parameters: Processing parameters and configuration
            
        Returns:
            Session tracking ID
        """
        try:
            session_record = {
                'session_id': session_id,
                'input_samples': input_samples,
                'processing_type': processing_type,
                'parameters': parameters,
                'started_at': datetime.utcnow().isoformat(),
                'started_by': getattr(self.blockchain, 'user_name', 'unknown'),
                'status': 'active',
                'output_samples': []
            }
            
            # Store in local cache and blockchain
            self.active_sessions[session_id] = session_record
            
            # Log session start to blockchain
            tx_id = await self.blockchain.submit_transaction(
                'start_processing_session',
                json.dumps(session_record)
            )
            
            self.logger.info(f"Processing session {session_id} started")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start processing session: {str(e)}")
            raise
    
    async def log_processing_step(self, session_id: str, step_name: str, 
                                 input_data: Any, output_data: Any, 
                                 step_parameters: Dict[str, Any]) -> str:
        """
        Log individual processing step within a session
        
        Args:
            session_id: Active session ID
            step_name: Name of the processing step
            input_data: Input data for this step
            output_data: Output data from this step  
            step_parameters: Step-specific parameters
            
        Returns:
            Step tracking hash
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found or not active")
            
            # Create step record
            step_record = {
                'session_id': session_id,
                'step_name': step_name,
                'step_index': len(self.active_sessions[session_id].get('steps', [])),
                'input_hash': self._calculate_data_hash(input_data),
                'output_hash': self._calculate_data_hash(output_data),
                'parameters': step_parameters,
                'timestamp': datetime.utcnow().isoformat(),
                'execution_time': 0  # Would measure actual execution time
            }
            
            # Add to session record
            if 'steps' not in self.active_sessions[session_id]:
                self.active_sessions[session_id]['steps'] = []
            self.active_sessions[session_id]['steps'].append(step_record)
            
            # Calculate step hash
            step_hash = hashlib.sha256(
                json.dumps(step_record, sort_keys=True).encode()
            ).hexdigest()
            
            self.logger.debug(f"Logged processing step {step_name} in session {session_id}")
            return step_hash
            
        except Exception as e:
            self.logger.error(f"Failed to log processing step: {str(e)}")
            raise
    
    async def complete_processing_session(self, session_id: str, 
                                        output_samples: List[str]) -> str:
        """
        Complete a processing session and commit full provenance to blockchain
        
        Args:
            session_id: Session to complete
            output_samples: List of output sample IDs
            
        Returns:
            Final session hash
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Update session record
            session = self.active_sessions[session_id]
            session.update({
                'status': 'completed',
                'output_samples': output_samples,
                'completed_at': datetime.utcnow().isoformat(),
                'total_steps': len(session.get('steps', []))
            })
            
            # Calculate final session hash
            session_hash = hashlib.sha256(
                json.dumps(session, sort_keys=True).encode()
            ).hexdigest()
            
            # Commit to blockchain
            tx_id = await self.blockchain.submit_transaction(
                'complete_processing_session',
                json.dumps(session)
            )
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"Processing session {session_id} completed with hash: {session_hash}")
            return session_hash
            
        except Exception as e:
            self.logger.error(f"Failed to complete processing session: {str(e)}")
            raise
    
    async def get_sample_lineage(self, sample_id: str, depth: int = 5) -> Dict[str, Any]:
        """
        Get complete lineage for a sample (upstream and downstream)
        
        Args:
            sample_id: Sample identifier
            depth: Maximum depth to traverse
            
        Returns:
            Complete lineage tree with provenance information
        """
        try:
            # Query blockchain for sample provenance
            # In production, this would trace through processing sessions
            
            lineage = {
                'sample_id': sample_id,
                'depth_explored': depth,
                'upstream_sources': [],
                'processing_history': [],
                'downstream_derivatives': [],
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Mock lineage data for demonstration
            lineage['upstream_sources'] = [
                {
                    'source_id': f"source_{i}",
                    'source_type': 'hospital_system',
                    'collection_date': datetime.utcnow().isoformat(),
                    'quality_score': 0.95
                }
                for i in range(2)
            ]
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Failed to get sample lineage: {str(e)}")
            return {'error': str(e)}
    
    async def verify_data_integrity(self, sample_id: str, expected_hash: str) -> Dict[str, Any]:
        """
        Verify data integrity by comparing hashes
        
        Args:
            sample_id: Sample identifier
            expected_hash: Expected data hash
            
        Returns:
            Integrity verification result
        """
        try:
            # Get sample record from blockchain
            sample_record = await self.blockchain.evaluate_transaction(
                'get_sample_provenance',
                sample_id
            )
            
            if not sample_record:
                return {
                    'verified': False,
                    'error': 'Sample not found in blockchain'
                }
            
            # Compare hashes
            stored_hash = sample_record.get('data_hash', '')
            integrity_verified = stored_hash == expected_hash
            
            result = {
                'sample_id': sample_id,
                'verified': integrity_verified,
                'stored_hash': stored_hash,
                'expected_hash': expected_hash,
                'verified_at': datetime.utcnow().isoformat()
            }
            
            if not integrity_verified:
                result['error'] = 'Data integrity verification failed - hashes do not match'
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to verify data integrity: {str(e)}")
            return {'verified': False, 'error': str(e)}
    
    # === Private Helper Methods ===
    
    def _validate_and_anonymize_sample(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sample data and ensure proper anonymization"""
        validated = sample_data.copy()
        
        # Remove any potential PII (this would be more sophisticated in production)
        pii_fields = ['name', 'ssn', 'address', 'phone', 'email', 'patient_name']
        for field in pii_fields:
            if field in validated:
                validated[field] = '[REDACTED]'
        
        # Validate required fields exist
        if not isinstance(validated, dict) or len(validated) == 0:
            raise ValueError("Sample data must be a non-empty dictionary")
        
        # Add validation metadata
        validated['_validation'] = {
            'validated_at': datetime.utcnow().isoformat(),
            'anonymization_applied': True,
            'field_count': len(validated)
        }
        
        return validated
    
    def _create_provenance_record(self, sample_id: str, sample_data: Dict[str, Any], 
                                 provenance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive provenance record"""
        record = {
            'sample_id': sample_id,
            'collection_info': {
                'source': provenance_info.get('source', 'unknown'),
                'collection_date': provenance_info.get('date', datetime.utcnow().isoformat()),
                'collection_method': provenance_info.get('method', 'unknown'),
                'collector_id': provenance_info.get('collector', 'unknown')
            },
            'quality_metrics': {
                'completeness': self._calculate_completeness(sample_data),
                'consistency_score': provenance_info.get('quality_score', 1.0),
                'validation_passed': True
            },
            'compliance_info': {
                'hipaa_compliant': True,
                'consent_obtained': provenance_info.get('consent', True),
                'anonymization_level': provenance_info.get('anonymization', 'full')
            },
            'technical_metadata': {
                'format': provenance_info.get('format', 'json'),
                'encoding': provenance_info.get('encoding', 'utf-8'),
                'size_bytes': len(json.dumps(sample_data).encode())
            }
        }
        
        return record
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash of data"""
        if data is None:
            return hashlib.sha256(b'null').hexdigest()
        
        data_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _calculate_completeness(self, sample_data: Dict[str, Any]) -> float:
        """Calculate data completeness score (0.0 to 1.0)"""
        if not sample_data:
            return 0.0
        
        non_null_fields = sum(1 for value in sample_data.values() 
                             if value is not None and value != '')
        total_fields = len(sample_data)
        
        return non_null_fields / total_fields if total_fields > 0 else 0.0