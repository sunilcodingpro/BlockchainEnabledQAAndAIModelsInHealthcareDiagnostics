"""
DataProvenanceLogger: Logs sample provenance and metadata for traceability.
"""
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


class DataProvenanceLogger:
    """
    Logger for data provenance and traceability on blockchain.
    
    This class tracks data sources, transformations, and lineage
    to ensure regulatory compliance and audit trails.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the data provenance logger.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain = blockchain_connector
        self._logger = logging.getLogger(__name__)
    
    def log_sample(self, sample_id: str, sample_data: Dict[str, Any], 
                   provenance_info: Dict[str, Any]) -> Optional[str]:
        """
        Log data sample with provenance information to blockchain.
        
        Args:
            sample_id: Unique identifier for the data sample
            sample_data: The actual data sample
            provenance_info: Provenance metadata (source, acquisition date, etc.)
            
        Returns:
            Sample hash if successful, None if failed
        """
        try:
            # Validate inputs
            if not sample_id or not sample_data:
                raise ValueError("Sample ID and data are required")
            
            # Calculate sample hash
            sample_hash = self._calculate_data_hash(sample_data)
            self._logger.info(f"Calculated hash for sample {sample_id}: {sample_hash}")
            
            # Prepare complete provenance record
            provenance_record = {
                'sample_id': sample_id,
                'sample_hash': sample_hash,
                'data_schema': self._extract_data_schema(sample_data),
                'timestamp': datetime.now().isoformat(),
                'provenance': {
                    'source': provenance_info.get('source', 'Unknown'),
                    'acquisition_date': provenance_info.get('date', datetime.now().isoformat()),
                    'acquisition_method': provenance_info.get('method', 'Unknown'),
                    'quality_score': provenance_info.get('quality_score'),
                    'preprocessing_steps': provenance_info.get('preprocessing_steps', []),
                    'data_lineage': provenance_info.get('lineage', []),
                    **{k: v for k, v in provenance_info.items() 
                       if k not in ['source', 'date', 'method', 'quality_score', 
                                   'preprocessing_steps', 'lineage']}
                }
            }
            
            # Store provenance on blockchain
            response = self.blockchain.invoke_chaincode(
                "logProvenance",
                [sample_id, sample_hash, json.dumps(provenance_record)]
            )
            
            if response.success:
                self._logger.info(f"Successfully logged provenance for sample {sample_id}")
                return sample_hash
            else:
                self._logger.error(f"Failed to log provenance on blockchain: {response.data}")
                return None
                
        except Exception as e:
            self._logger.error(f"Error logging sample provenance for {sample_id}: {e}")
            return None
    
    def get_sample_provenance(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve provenance information for a data sample.
        
        Args:
            sample_id: ID of the sample to retrieve
            
        Returns:
            Provenance information dictionary or None if not found
        """
        try:
            response = self.blockchain.query_chaincode("getProvenance", [sample_id])
            
            if response.success and response.data:
                return response.data
            else:
                self._logger.debug(f"Provenance for sample {sample_id} not found")
                return None
                
        except Exception as e:
            self._logger.error(f"Error retrieving provenance for sample {sample_id}: {e}")
            return None
    
    def trace_data_lineage(self, sample_id: str) -> List[Dict[str, Any]]:
        """
        Trace the complete lineage of a data sample.
        
        Args:
            sample_id: ID of the sample to trace
            
        Returns:
            List of lineage records from source to current sample
        """
        try:
            lineage_chain = []
            current_id = sample_id
            visited_ids = set()
            
            while current_id and current_id not in visited_ids:
                visited_ids.add(current_id)
                
                # Get provenance for current sample
                provenance = self.get_sample_provenance(current_id)
                if not provenance:
                    break
                
                lineage_chain.append({
                    'sample_id': current_id,
                    'timestamp': provenance.get('timestamp'),
                    'source': provenance.get('provenance', {}).get('source'),
                    'transformations': provenance.get('provenance', {}).get('preprocessing_steps', [])
                })
                
                # Check for parent sample in lineage
                lineage = provenance.get('provenance', {}).get('lineage', [])
                current_id = lineage[0] if lineage else None
            
            return lineage_chain
            
        except Exception as e:
            self._logger.error(f"Error tracing data lineage for {sample_id}: {e}")
            return []
    
    def log_data_transformation(self, input_sample_id: str, output_sample_id: str,
                              transformation_info: Dict[str, Any]) -> bool:
        """
        Log a data transformation operation.
        
        Args:
            input_sample_id: ID of the input sample
            output_sample_id: ID of the output sample
            transformation_info: Details about the transformation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get input sample provenance
            input_provenance = self.get_sample_provenance(input_sample_id)
            if not input_provenance:
                self._logger.error(f"Input sample {input_sample_id} not found")
                return False
            
            # Create transformation record
            transformation_record = {
                'transformation_id': self._generate_transformation_id(input_sample_id, output_sample_id),
                'input_sample_id': input_sample_id,
                'output_sample_id': output_sample_id,
                'transformation_type': transformation_info.get('type', 'unknown'),
                'algorithm': transformation_info.get('algorithm'),
                'parameters': transformation_info.get('parameters', {}),
                'timestamp': datetime.now().isoformat(),
                'operator': transformation_info.get('operator', 'system')
            }
            
            # Update output sample's lineage
            updated_lineage = input_provenance.get('provenance', {}).get('lineage', [])
            updated_lineage.append(input_sample_id)
            
            # Update preprocessing steps
            preprocessing_steps = input_provenance.get('provenance', {}).get('preprocessing_steps', [])
            preprocessing_steps.append(transformation_record)
            
            self._logger.info(f"Logged transformation from {input_sample_id} to {output_sample_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error logging data transformation: {e}")
            return False
    
    def validate_sample_integrity(self, sample_id: str, sample_data: Dict[str, Any]) -> bool:
        """
        Validate that sample data matches its recorded hash.
        
        Args:
            sample_id: ID of the sample to validate
            sample_data: Current sample data
            
        Returns:
            True if integrity check passes, False otherwise
        """
        try:
            # Get recorded provenance
            provenance = self.get_sample_provenance(sample_id)
            if not provenance:
                self._logger.error(f"Sample {sample_id} not found in provenance records")
                return False
            
            # Calculate current hash
            current_hash = self._calculate_data_hash(sample_data)
            recorded_hash = provenance.get('sample_hash')
            
            if current_hash == recorded_hash:
                self._logger.info(f"Sample {sample_id} integrity verified")
                return True
            else:
                self._logger.warning(f"Sample {sample_id} integrity check failed. "
                                   f"Expected: {recorded_hash}, Got: {current_hash}")
                return False
                
        except Exception as e:
            self._logger.error(f"Error validating sample integrity: {e}")
            return False
    
    def list_all_samples(self, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all samples with optional source filtering.
        
        Args:
            source_filter: Optional source name to filter by
            
        Returns:
            List of sample provenance records
        """
        try:
            response = self.blockchain.query_chaincode("getAllProvenance", [])
            
            if response.success and response.data:
                samples = response.data
                
                # Apply source filter if provided
                if source_filter:
                    samples = [s for s in samples 
                             if s.get('provenance', {}).get('source') == source_filter]
                
                return samples
            else:
                return []
                
        except Exception as e:
            self._logger.error(f"Error listing samples: {e}")
            return []
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of data dictionary.
        
        Args:
            data: Data dictionary to hash
            
        Returns:
            Hexadecimal hash string
        """
        # Convert to JSON with sorted keys for consistent hashing
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _extract_data_schema(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract schema information from data sample.
        
        Args:
            data: Data sample
            
        Returns:
            Schema dictionary with field names and types
        """
        schema = {}
        for key, value in data.items():
            schema[key] = type(value).__name__
        return schema
    
    def _generate_transformation_id(self, input_id: str, output_id: str) -> str:
        """
        Generate unique ID for a transformation.
        
        Args:
            input_id: Input sample ID
            output_id: Output sample ID
            
        Returns:
            Transformation ID
        """
        combined = f"{input_id}->{output_id}-{datetime.now().isoformat()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]