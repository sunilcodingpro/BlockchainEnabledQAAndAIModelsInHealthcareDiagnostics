"""
DecisionAuditLogger: Logs AI/ML decisions and explanations to the audit trail.

This module provides comprehensive logging of AI/ML model decisions, including
input data, predictions, confidence scores, and explainability information.
All records are stored on the blockchain for immutable audit trails.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List


class DecisionAuditLogger:
    """
    Logger for AI/ML decision audit trails with blockchain storage.
    
    This class captures and stores all relevant information about AI/ML model
    decisions, ensuring complete traceability and regulatory compliance.
    """
    
    def __init__(self, blockchain_connector):
        """
        Initialize the decision audit logger.
        
        Args:
            blockchain_connector: Instance of HyperledgerConnector for blockchain operations
        """
        self.blockchain_connector = blockchain_connector
    
    def log_decision(self, case_id: str, input_data: Dict[str, Any], 
                    model_name: str, decision: str, 
                    explanation: Dict[str, Any], 
                    confidence_score: Optional[float] = None,
                    additional_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an AI/ML model decision to the audit trail.
        
        Args:
            case_id: Unique identifier for this diagnostic case
            input_data: Input features used for the decision
            model_name: Name/ID of the model used
            decision: The actual decision made by the model
            explanation: Explainability information (SHAP values, LIME results, etc.)
            confidence_score: Optional confidence score of the decision
            additional_metadata: Any additional metadata to store
            
        Returns:
            Hash of the logged decision record
        """
        try:
            # Generate unique diagnostic ID
            diagnostic_id = self._generate_diagnostic_id(case_id, model_name)
            
            # Prepare comprehensive decision record
            decision_record = {
                "case_id": case_id,
                "diagnostic_id": diagnostic_id,
                "model_name": model_name,
                "input_data": input_data,
                "decision": decision,
                "confidence_score": confidence_score,
                "explanation": explanation,
                "timestamp": datetime.now().isoformat(),
                "metadata": additional_metadata or {}
            }
            
            # Calculate record hash
            record_hash = self._calculate_record_hash(decision_record)
            decision_record["record_hash"] = record_hash
            
            # Extract patient ID safely (anonymized)
            patient_id = self._extract_patient_id(input_data, case_id)
            
            # Log decision to blockchain
            result = self.blockchain_connector.invoke_transaction(
                "log_diagnostic",
                diagnostic_id,
                patient_id,
                model_name,
                input_data,
                {"decision": decision},
                confidence_score or 0.0,
                explanation
            )
            
            if result.get("status") == "success":
                print(f"Decision logged successfully: {diagnostic_id}")
                return result.get("hash", record_hash)
            else:
                print(f"Failed to log decision to blockchain: {result.get('message')}")
                return record_hash
                
        except Exception as e:
            print(f"Error logging decision: {e}")
            # Return a fallback hash
            return self._calculate_record_hash({"case_id": case_id, "error": str(e)})
    
    def log_prediction_batch(self, batch_predictions: List[Dict[str, Any]]) -> List[str]:
        """
        Log multiple predictions in batch for efficiency.
        
        Args:
            batch_predictions: List of prediction dictionaries, each containing
                             case_id, input_data, model_name, decision, explanation
            
        Returns:
            List of hashes for the logged predictions
        """
        hashes = []
        
        for prediction in batch_predictions:
            try:
                hash_value = self.log_decision(
                    prediction.get("case_id"),
                    prediction.get("input_data"),
                    prediction.get("model_name"),
                    prediction.get("decision"),
                    prediction.get("explanation"),
                    prediction.get("confidence_score"),
                    prediction.get("additional_metadata")
                )
                hashes.append(hash_value)
            except Exception as e:
                print(f"Error logging batch prediction: {e}")
                hashes.append(None)
        
        return hashes
    
    def get_decision_audit(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve audit information for a specific case.
        
        Args:
            case_id: Unique identifier of the case
            
        Returns:
            Audit information or None if not found
        """
        try:
            # Generate diagnostic ID based on case ID
            diagnostic_id = self._generate_diagnostic_id(case_id, "")
            
            # Query blockchain for diagnostic information
            result = self.blockchain_connector.query_ledger("query_diagnostic", diagnostic_id)
            
            if result.get("status") == "success":
                return result.get("diagnostic")
            else:
                print(f"Audit record for case {case_id} not found")
                return None
                
        except Exception as e:
            print(f"Error retrieving decision audit: {e}")
            return None
    
    def get_model_decisions(self, model_name: str, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all decisions made by a specific model within a time range.
        
        Args:
            model_name: Name of the model
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format) 
            limit: Maximum number of records to return
            
        Returns:
            List of decision records
        """
        try:
            # Get audit trail filtered by model
            result = self.blockchain_connector.query_ledger("get_audit_trail", model_name, limit)
            
            if result.get("status") == "success":
                audit_entries = result.get("audit_trail", [])
                
                # Filter by date range if provided
                if start_date or end_date:
                    filtered_entries = []
                    for entry in audit_entries:
                        entry_date = entry.get("timestamp", "")
                        if self._is_date_in_range(entry_date, start_date, end_date):
                            filtered_entries.append(entry)
                    return filtered_entries
                
                return audit_entries
            else:
                print(f"No decisions found for model {model_name}")
                return []
                
        except Exception as e:
            print(f"Error retrieving model decisions: {e}")
            return []
    
    def generate_audit_report(self, case_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report for a specific case.
        
        Args:
            case_id: Unique identifier of the case
            
        Returns:
            Comprehensive audit report
        """
        try:
            # Get decision audit
            decision_audit = self.get_decision_audit(case_id)
            
            if not decision_audit:
                return {
                    "status": "error",
                    "message": f"No audit record found for case {case_id}"
                }
            
            # Prepare comprehensive report
            report = {
                "case_id": case_id,
                "audit_summary": {
                    "decision_timestamp": decision_audit.get("timestamp"),
                    "model_used": decision_audit.get("model_id"),
                    "decision": decision_audit.get("prediction", {}).get("decision"),
                    "confidence_score": decision_audit.get("confidence_score")
                },
                "input_data_summary": self._summarize_input_data(decision_audit.get("input_data", {})),
                "explainability_summary": self._summarize_explanation(decision_audit.get("explanation", {})),
                "blockchain_verification": {
                    "hash": decision_audit.get("hash"),
                    "diagnostic_id": decision_audit.get("diagnostic_id")
                },
                "report_generated_at": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "report": report
            }
            
        except Exception as e:
            print(f"Error generating audit report: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def verify_decision_integrity(self, case_id: str, 
                                expected_hash: str) -> bool:
        """
        Verify the integrity of a logged decision.
        
        Args:
            case_id: Unique identifier of the case
            expected_hash: Expected hash of the decision record
            
        Returns:
            True if integrity is verified, False otherwise
        """
        try:
            decision_audit = self.get_decision_audit(case_id)
            
            if not decision_audit:
                return False
            
            stored_hash = decision_audit.get("hash")
            return stored_hash == expected_hash
            
        except Exception as e:
            print(f"Error verifying decision integrity: {e}")
            return False
    
    def _generate_diagnostic_id(self, case_id: str, model_name: str) -> str:
        """
        Generate a unique diagnostic identifier.
        
        Args:
            case_id: Case identifier
            model_name: Model name
            
        Returns:
            Unique diagnostic identifier
        """
        timestamp = str(int(datetime.now().timestamp() * 1000))
        data = f"{case_id}_{model_name}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_record_hash(self, record: Dict[str, Any]) -> str:
        """
        Calculate hash of a decision record.
        
        Args:
            record: Decision record dictionary
            
        Returns:
            SHA256 hash of the record
        """
        # Create a copy without the hash field itself
        record_copy = {k: v for k, v in record.items() if k != "record_hash"}
        record_json = json.dumps(record_copy, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def _extract_patient_id(self, input_data: Dict[str, Any], case_id: str) -> str:
        """
        Extract or generate anonymized patient ID.
        
        Args:
            input_data: Input data dictionary
            case_id: Case identifier
            
        Returns:
            Anonymized patient identifier
        """
        # Look for patient ID in input data
        patient_id = input_data.get("patient_id") or input_data.get("id")
        
        if patient_id:
            # Anonymize the patient ID
            return hashlib.sha256(str(patient_id).encode()).hexdigest()[:12]
        else:
            # Generate from case ID
            return hashlib.sha256(case_id.encode()).hexdigest()[:12]
    
    def _is_date_in_range(self, date_str: str, start_date: Optional[str], 
                         end_date: Optional[str]) -> bool:
        """
        Check if a date string falls within the specified range.
        
        Args:
            date_str: Date string to check
            start_date: Start date of range
            end_date: End date of range
            
        Returns:
            True if date is in range, False otherwise
        """
        try:
            if not date_str:
                return False
            
            if start_date and date_str < start_date:
                return False
            
            if end_date and date_str > end_date:
                return False
            
            return True
        except Exception:
            return False
    
    def _summarize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of input data for the report.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Summarized input data
        """
        return {
            "feature_count": len(input_data),
            "features": list(input_data.keys())[:10],  # Limit to first 10 features
            "sample_values": {k: v for k, v in list(input_data.items())[:5]}  # Sample values
        }
    
    def _summarize_explanation(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of explainability information.
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Summarized explanation
        """
        if "feature_importance" in explanation:
            feature_importance = explanation["feature_importance"]
            return {
                "type": "feature_importance",
                "top_features": sorted(feature_importance.items(), 
                                     key=lambda x: abs(x[1]), reverse=True)[:5],
                "explanation_method": explanation.get("method", "unknown")
            }
        else:
            return {
                "type": "other",
                "available_fields": list(explanation.keys())
            }