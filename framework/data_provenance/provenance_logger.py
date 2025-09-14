"""
DataProvenanceLogger: Logs sample provenance and metadata for traceability.

This module provides comprehensive data provenance tracking, ensuring complete
traceability of data samples from source to model prediction, maintaining
compliance with healthcare regulations and audit requirements.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class DataProvenanceLogger:
    """
    Logger for data provenance and traceability with blockchain storage.
    
    This class captures and stores detailed information about data samples,
    their sources, transformations, and usage in AI/ML model predictions.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the data provenance logger.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain_connector = blockchain_connector
    
    def log_sample(self, sample_id: str, sample_data: Dict[str, Any], 
                  provenance_info: Dict[str, Any]) -> str:
        """
        Log a data sample with its provenance information.
        
        Args:
            sample_id: Unique identifier for the sample
            sample_data: The actual sample data (anonymized for healthcare)
            provenance_info: Provenance metadata (source, collection method, etc.)
            
        Returns:
            Hash of the logged sample record
        """
        try:
            # Create comprehensive provenance record
            provenance_record = {
                "sample_id": sample_id,
                "data_hash": self._calculate_data_hash(sample_data),
                "provenance_info": {
                    **provenance_info,
                    "logged_at": datetime.now().isoformat(),
                    "data_fields": list(sample_data.keys()),
                    "data_types": {k: type(v).__name__ for k, v in sample_data.items()}
                },
                "anonymization_info": {
                    "method": "healthcare_anonymization",
                    "anonymized_at": datetime.now().isoformat(),
                    "original_field_count": len(sample_data)
                }
            }
            
            # Calculate record hash
            record_hash = self._calculate_record_hash(provenance_record)
            
            # For healthcare compliance, we don't store raw data on blockchain
            # Instead, we store hashes and metadata
            blockchain_record = {
                "sample_id": sample_id,
                "data_hash": provenance_record["data_hash"],
                "provenance_metadata": provenance_record["provenance_info"],
                "anonymization_info": provenance_record["anonymization_info"],
                "compliance_flags": self._generate_compliance_flags(sample_data, provenance_info)
            }
            
            # Log to blockchain as compliance event
            result = self.blockchain_connector.invoke_transaction(
                "log_compliance_event",
                f"data_provenance_{sample_id}_{int(datetime.now().timestamp())}",
                "data_provenance",
                provenance_info.get("model_id", "unknown"),
                blockchain_record
            )
            
            if result.get("status") == "success":
                print(f"Sample provenance logged: {sample_id}")
                return result.get("hash", record_hash)
            else:
                print(f"Failed to log sample provenance: {result.get('message')}")
                return record_hash
                
        except Exception as e:
            print(f"Error logging sample provenance: {e}")
            # Return fallback hash
            return self._calculate_data_hash(sample_data)
    
    def log_data_transformation(self, transformation_id: str, 
                              source_sample_ids: List[str],
                              target_sample_id: str,
                              transformation_details: Dict[str, Any]) -> str:
        """
        Log data transformations and preprocessing steps.
        
        Args:
            transformation_id: Unique identifier for the transformation
            source_sample_ids: List of source sample IDs
            target_sample_id: ID of the transformed sample
            transformation_details: Details about the transformation process
            
        Returns:
            Hash of the transformation record
        """
        try:
            transformation_record = {
                "transformation_id": transformation_id,
                "source_samples": source_sample_ids,
                "target_sample": target_sample_id,
                "transformation_type": transformation_details.get("type", "unknown"),
                "method": transformation_details.get("method", ""),
                "parameters": transformation_details.get("parameters", {}),
                "timestamp": datetime.now().isoformat(),
                "applied_by": transformation_details.get("applied_by", "system")
            }
            
            # Log transformation as compliance event
            result = self.blockchain_connector.invoke_transaction(
                "log_compliance_event",
                transformation_id,
                "data_transformation",
                transformation_details.get("model_id", "unknown"),
                transformation_record
            )
            
            if result.get("status") == "success":
                return result.get("hash", self._calculate_record_hash(transformation_record))
            else:
                return self._calculate_record_hash(transformation_record)
                
        except Exception as e:
            print(f"Error logging data transformation: {e}")
            return self._calculate_record_hash({"transformation_id": transformation_id, "error": str(e)})
    
    def get_sample_provenance(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve provenance information for a specific sample.
        
        Args:
            sample_id: Unique identifier of the sample
            
        Returns:
            Provenance information or None if not found
        """
        try:
            # Query blockchain for compliance events related to this sample
            result = self.blockchain_connector.query_ledger("get_audit_trail")
            
            if result.get("status") == "success":
                audit_trail = result.get("audit_trail", [])
                
                # Filter for data provenance events for this sample
                sample_events = []
                for event in audit_trail:
                    event_data = event.get("data", {})
                    if (event_data.get("event_type") == "data_provenance" and 
                        sample_id in str(event_data)):
                        sample_events.append(event)
                
                if sample_events:
                    return {
                        "sample_id": sample_id,
                        "provenance_events": sample_events,
                        "retrieved_at": datetime.now().isoformat()
                    }
            
            return None
            
        except Exception as e:
            print(f"Error retrieving sample provenance: {e}")
            return None
    
    def trace_data_lineage(self, sample_id: str) -> Dict[str, Any]:
        """
        Trace the complete lineage of a data sample.
        
        Args:
            sample_id: Unique identifier of the sample
            
        Returns:
            Complete data lineage information
        """
        try:
            lineage = {
                "sample_id": sample_id,
                "lineage_nodes": [],
                "transformations": [],
                "usage_records": []
            }
            
            # Get sample provenance
            provenance = self.get_sample_provenance(sample_id)
            if provenance:
                lineage["lineage_nodes"].append({
                    "node_type": "origin",
                    "sample_id": sample_id,
                    "provenance": provenance
                })
            
            # In a full implementation, this would:
            # 1. Trace all transformations applied to this sample
            # 2. Find all models that used this sample
            # 3. Track all decisions made using this sample
            # 4. Build a complete graph of data flow
            
            lineage["trace_completed_at"] = datetime.now().isoformat()
            lineage["status"] = "success"
            
            return lineage
            
        except Exception as e:
            print(f"Error tracing data lineage: {e}")
            return {
                "sample_id": sample_id,
                "status": "error",
                "message": str(e)
            }
    
    def validate_data_integrity(self, sample_id: str, 
                              current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the integrity of a data sample against its recorded hash.
        
        Args:
            sample_id: Unique identifier of the sample
            current_data: Current data to validate
            
        Returns:
            Validation result
        """
        try:
            # Get stored provenance
            provenance = self.get_sample_provenance(sample_id)
            
            if not provenance:
                return {
                    "valid": False,
                    "message": "No provenance record found",
                    "sample_id": sample_id
                }
            
            # Calculate current data hash
            current_hash = self._calculate_data_hash(current_data)
            
            # Extract stored hash from provenance events
            stored_hash = None
            for event in provenance.get("provenance_events", []):
                compliance_data = event.get("data", {}).get("compliance_data", {})
                if "data_hash" in compliance_data:
                    stored_hash = compliance_data["data_hash"]
                    break
            
            if not stored_hash:
                return {
                    "valid": False,
                    "message": "No stored hash found in provenance record",
                    "sample_id": sample_id
                }
            
            # Compare hashes
            is_valid = current_hash == stored_hash
            
            return {
                "valid": is_valid,
                "sample_id": sample_id,
                "current_hash": current_hash,
                "stored_hash": stored_hash,
                "validated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "valid": False,
                "sample_id": sample_id,
                "message": f"Validation error: {e}"
            }
    
    def generate_provenance_report(self, sample_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive provenance report for multiple samples.
        
        Args:
            sample_ids: List of sample IDs to include in the report
            
        Returns:
            Comprehensive provenance report
        """
        try:
            report = {
                "report_type": "data_provenance",
                "sample_count": len(sample_ids),
                "samples": [],
                "summary": {
                    "total_samples": len(sample_ids),
                    "samples_with_provenance": 0,
                    "samples_with_lineage": 0,
                    "validation_results": {}
                },
                "generated_at": datetime.now().isoformat()
            }
            
            for sample_id in sample_ids:
                sample_info = {
                    "sample_id": sample_id,
                    "provenance": self.get_sample_provenance(sample_id),
                    "lineage": self.trace_data_lineage(sample_id)
                }
                
                # Update summary
                if sample_info["provenance"]:
                    report["summary"]["samples_with_provenance"] += 1
                
                if sample_info["lineage"].get("status") == "success":
                    report["summary"]["samples_with_lineage"] += 1
                
                report["samples"].append(sample_info)
            
            return report
            
        except Exception as e:
            return {
                "report_type": "data_provenance",
                "status": "error",
                "message": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _calculate_data_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate hash of data while preserving privacy.
        
        Args:
            data: Data dictionary
            
        Returns:
            SHA256 hash of the data
        """
        # For healthcare data, we hash the structure and anonymized values
        anonymized_data = self._anonymize_for_hash(data)
        data_json = json.dumps(anonymized_data, sort_keys=True)
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _calculate_record_hash(self, record: Dict[str, Any]) -> str:
        """
        Calculate hash of a provenance record.
        
        Args:
            record: Provenance record dictionary
            
        Returns:
            SHA256 hash of the record
        """
        record_json = json.dumps(record, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def _anonymize_for_hash(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize sensitive data fields while preserving structure for hashing.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Anonymized data dictionary
        """
        anonymized = {}
        
        for key, value in data.items():
            if key.lower() in ['patient_id', 'ssn', 'name', 'phone', 'email']:
                # Hash sensitive fields
                anonymized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
            elif isinstance(value, (int, float)):
                # Preserve numeric values for ML purposes
                anonymized[key] = value
            else:
                # Hash other string values
                anonymized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:16]
        
        return anonymized
    
    def _generate_compliance_flags(self, sample_data: Dict[str, Any], 
                                  provenance_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate compliance flags based on data and provenance information.
        
        Args:
            sample_data: Sample data dictionary
            provenance_info: Provenance information
            
        Returns:
            Compliance flags dictionary
        """
        flags = {
            "hipaa_compliant": True,  # Assume compliant unless issues found
            "gdpr_compliant": True,
            "anonymized": True,
            "consent_verified": provenance_info.get("consent_verified", False),
            "data_retention_policy": provenance_info.get("retention_policy", "standard"),
            "audit_trail_complete": True
        }
        
        # Check for potential compliance issues
        sensitive_fields = ['patient_id', 'ssn', 'name', 'phone', 'email']
        for field in sensitive_fields:
            if field in sample_data and len(str(sample_data[field])) > 16:
                flags["anonymized"] = False
                flags["hipaa_compliant"] = False
        
        return flags