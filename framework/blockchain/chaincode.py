"""
Chaincode for Blockchain-Enabled QA and AI Models in Healthcare Diagnostics

This module provides chaincode functions for Hyperledger Fabric to handle:
- Model registry and metadata storage
- Diagnostic data and results logging
- Model drift detection tracking
- Regulatory compliance audit trails
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional


class HealthcareDiagnosticsChaincode:
    """
    Chaincode implementation for healthcare diagnostics blockchain operations.
    
    This chaincode handles model registration, diagnostic logging, drift detection,
    and compliance tracking for AI/ML models in healthcare applications.
    """
    
    def __init__(self):
        """Initialize the chaincode with empty state."""
        self.models = {}
        self.diagnostics = {}
        self.audit_logs = {}
        self.compliance_records = {}
    
    def init_ledger(self) -> Dict[str, Any]:
        """
        Initialize the ledger with default values.
        
        Returns:
            Dict indicating successful initialization
        """
        return {
            "status": "success",
            "message": "Healthcare Diagnostics Chaincode initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    def register_model(self, model_id: str, model_name: str, version: str, 
                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new AI/ML model on the blockchain.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            version: Model version string
            metadata: Model metadata including accuracy, training data info, etc.
            
        Returns:
            Dict with registration details and blockchain hash
        """
        timestamp = datetime.now().isoformat()
        
        # Create model record
        model_record = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "metadata": metadata,
            "registered_at": timestamp,
            "status": "active"
        }
        
        # Generate hash for model record
        model_hash = self._generate_hash(json.dumps(model_record, sort_keys=True))
        model_record["hash"] = model_hash
        
        # Store in ledger
        self.models[model_id] = model_record
        
        # Log the registration
        self._add_audit_log("model_registration", {
            "model_id": model_id,
            "action": "registered",
            "timestamp": timestamp,
            "hash": model_hash
        })
        
        return {
            "status": "success",
            "model_id": model_id,
            "hash": model_hash,
            "timestamp": timestamp
        }
    
    def log_diagnostic(self, diagnostic_id: str, patient_id: str, model_id: str,
                      input_data: Dict[str, Any], prediction: Dict[str, Any],
                      confidence_score: float, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log a diagnostic prediction made by an AI model.
        
        Args:
            diagnostic_id: Unique identifier for this diagnostic
            patient_id: Anonymous patient identifier
            model_id: ID of the model used for prediction
            input_data: Input features used for prediction
            prediction: Model prediction results
            confidence_score: Confidence score of the prediction
            explanation: Explainability information (SHAP/LIME results)
            
        Returns:
            Dict with logging confirmation and hash
        """
        timestamp = datetime.now().isoformat()
        
        # Create diagnostic record
        diagnostic_record = {
            "diagnostic_id": diagnostic_id,
            "patient_id": patient_id,
            "model_id": model_id,
            "input_data": input_data,
            "prediction": prediction,
            "confidence_score": confidence_score,
            "explanation": explanation,
            "timestamp": timestamp
        }
        
        # Generate hash
        diagnostic_hash = self._generate_hash(json.dumps(diagnostic_record, sort_keys=True))
        diagnostic_record["hash"] = diagnostic_hash
        
        # Store in ledger
        self.diagnostics[diagnostic_id] = diagnostic_record
        
        # Log the diagnostic
        self._add_audit_log("diagnostic_prediction", {
            "diagnostic_id": diagnostic_id,
            "model_id": model_id,
            "timestamp": timestamp,
            "hash": diagnostic_hash
        })
        
        return {
            "status": "success",
            "diagnostic_id": diagnostic_id,
            "hash": diagnostic_hash,
            "timestamp": timestamp
        }
    
    def detect_model_drift(self, model_id: str, drift_metrics: Dict[str, Any],
                          threshold: float) -> Dict[str, Any]:
        """
        Log model drift detection results.
        
        Args:
            model_id: ID of the model being monitored
            drift_metrics: Calculated drift metrics
            threshold: Drift threshold for alerts
            
        Returns:
            Dict with drift detection results
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate if drift detected
        drift_detected = any(
            metric > threshold for metric in drift_metrics.values() 
            if isinstance(metric, (int, float))
        )
        
        drift_record = {
            "model_id": model_id,
            "drift_metrics": drift_metrics,
            "threshold": threshold,
            "drift_detected": drift_detected,
            "timestamp": timestamp
        }
        
        drift_hash = self._generate_hash(json.dumps(drift_record, sort_keys=True))
        drift_record["hash"] = drift_hash
        
        # Update model status if drift detected
        if drift_detected and model_id in self.models:
            self.models[model_id]["status"] = "drift_detected"
            self.models[model_id]["drift_detected_at"] = timestamp
        
        # Log drift detection
        self._add_audit_log("drift_detection", {
            "model_id": model_id,
            "drift_detected": drift_detected,
            "timestamp": timestamp,
            "hash": drift_hash
        })
        
        return {
            "status": "success",
            "model_id": model_id,
            "drift_detected": drift_detected,
            "hash": drift_hash,
            "timestamp": timestamp
        }
    
    def log_compliance_event(self, event_id: str, event_type: str, 
                           model_id: str, compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log regulatory compliance events.
        
        Args:
            event_id: Unique identifier for the compliance event
            event_type: Type of compliance event (audit, validation, etc.)
            model_id: ID of the model related to this event
            compliance_data: Compliance-related data and results
            
        Returns:
            Dict with compliance logging confirmation
        """
        timestamp = datetime.now().isoformat()
        
        compliance_record = {
            "event_id": event_id,
            "event_type": event_type,
            "model_id": model_id,
            "compliance_data": compliance_data,
            "timestamp": timestamp
        }
        
        compliance_hash = self._generate_hash(json.dumps(compliance_record, sort_keys=True))
        compliance_record["hash"] = compliance_hash
        
        # Store compliance record
        self.compliance_records[event_id] = compliance_record
        
        # Log compliance event
        self._add_audit_log("compliance_event", {
            "event_id": event_id,
            "event_type": event_type,
            "model_id": model_id,
            "timestamp": timestamp,
            "hash": compliance_hash
        })
        
        return {
            "status": "success",
            "event_id": event_id,
            "hash": compliance_hash,
            "timestamp": timestamp
        }
    
    def query_model(self, model_id: str) -> Dict[str, Any]:
        """
        Query model information from the ledger.
        
        Args:
            model_id: ID of the model to query
            
        Returns:
            Model information or error if not found
        """
        if model_id in self.models:
            return {
                "status": "success",
                "model": self.models[model_id]
            }
        else:
            return {
                "status": "error",
                "message": f"Model {model_id} not found"
            }
    
    def query_diagnostic(self, diagnostic_id: str) -> Dict[str, Any]:
        """
        Query diagnostic information from the ledger.
        
        Args:
            diagnostic_id: ID of the diagnostic to query
            
        Returns:
            Diagnostic information or error if not found
        """
        if diagnostic_id in self.diagnostics:
            return {
                "status": "success",
                "diagnostic": self.diagnostics[diagnostic_id]
            }
        else:
            return {
                "status": "error",
                "message": f"Diagnostic {diagnostic_id} not found"
            }
    
    def get_audit_trail(self, model_id: Optional[str] = None, 
                       limit: int = 100) -> Dict[str, Any]:
        """
        Get audit trail entries, optionally filtered by model.
        
        Args:
            model_id: Optional model ID to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of audit trail entries
        """
        audit_entries = list(self.audit_logs.values())
        
        # Filter by model_id if provided
        if model_id:
            audit_entries = [
                entry for entry in audit_entries 
                if entry.get("data", {}).get("model_id") == model_id
            ]
        
        # Sort by timestamp (newest first) and limit
        audit_entries.sort(key=lambda x: x["timestamp"], reverse=True)
        audit_entries = audit_entries[:limit]
        
        return {
            "status": "success",
            "audit_trail": audit_entries,
            "total_entries": len(audit_entries)
        }
    
    def get_compliance_report(self, model_id: str, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate compliance report for a model.
        
        Args:
            model_id: ID of the model to report on
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            Compliance report data
        """
        # Get model info
        model_info = self.query_model(model_id)
        if model_info["status"] == "error":
            return model_info
        
        # Get related diagnostics
        model_diagnostics = [
            diag for diag in self.diagnostics.values() 
            if diag["model_id"] == model_id
        ]
        
        # Get related compliance events
        model_compliance = [
            comp for comp in self.compliance_records.values() 
            if comp["model_id"] == model_id
        ]
        
        # Get audit trail for this model
        audit_trail = self.get_audit_trail(model_id)
        
        report = {
            "model_id": model_id,
            "model_info": model_info["model"],
            "diagnostics_count": len(model_diagnostics),
            "compliance_events_count": len(model_compliance),
            "audit_entries_count": audit_trail["total_entries"],
            "report_generated_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "compliance_report": report
        }
    
    def _generate_hash(self, data: str) -> str:
        """Generate SHA256 hash of the input data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _add_audit_log(self, action: str, data: Dict[str, Any]) -> None:
        """Add an entry to the audit log."""
        log_id = f"{action}_{int(time.time() * 1000)}"
        
        audit_entry = {
            "log_id": log_id,
            "action": action,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.audit_logs[log_id] = audit_entry


# Example chaincode functions for Hyperledger Fabric deployment
def init(stub):
    """Initialize the chaincode."""
    chaincode = HealthcareDiagnosticsChaincode()
    result = chaincode.init_ledger()
    return result


def invoke(stub, function_name, args):
    """
    Invoke chaincode functions based on function name.
    
    This would be the main entry point when deployed to Hyperledger Fabric.
    """
    chaincode = HealthcareDiagnosticsChaincode()
    
    function_map = {
        "register_model": chaincode.register_model,
        "log_diagnostic": chaincode.log_diagnostic,
        "detect_model_drift": chaincode.detect_model_drift,
        "log_compliance_event": chaincode.log_compliance_event,
        "query_model": chaincode.query_model,
        "query_diagnostic": chaincode.query_diagnostic,
        "get_audit_trail": chaincode.get_audit_trail,
        "get_compliance_report": chaincode.get_compliance_report
    }
    
    if function_name in function_map:
        try:
            # Parse arguments and call function
            return function_map[function_name](*args)
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    else:
        return {
            "status": "error",
            "message": f"Unknown function: {function_name}"
        }