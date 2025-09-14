"""
Backend REST API for Blockchain-Enabled QA and AI Models in Healthcare Diagnostics

This module provides a comprehensive REST API using Flask for interacting with
the healthcare diagnostics blockchain system. It includes endpoints for model
registration, diagnostic submission, audit trails, compliance reports, and simulation.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import framework modules
from framework.blockchain.hyperledger_connector import HyperledgerConnector
from framework.model_registry.registry import ModelRegistry
from framework.data_provenance.provenance_logger import DataProvenanceLogger
from framework.decision_audit.audit_logger import DecisionAuditLogger
from framework.regulatory.report_generator import RegulatoryReportGenerator
from framework.explainability.shap_explainer import SHAPExplainer
from framework.explainability.lime_explainer import LIMEExplainer
from framework.simulation.simulator import HealthcareSimulator, SimulationConfig


class HealthcareDiagnosticsAPI:
    """
    REST API for Healthcare Diagnostics Blockchain System.
    
    This class provides a complete REST API interface for interacting with
    the blockchain-enabled healthcare diagnostics system.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the API with framework components.
        
        Args:
            config_path: Path to configuration file
        """
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web frontend
        
        # Initialize blockchain connector
        self.blockchain = HyperledgerConnector(
            config_path="network.yaml",
            channel_name="qahealthchannel", 
            chaincode_name="aiqa_cc",
            org_name="HealthcareOrg",
            user_name="api_user"
        )
        
        # Initialize framework components
        self.model_registry = ModelRegistry(self.blockchain)
        self.data_logger = DataProvenanceLogger(self.blockchain)
        self.audit_logger = DecisionAuditLogger(self.blockchain)
        self.report_generator = RegulatoryReportGenerator(self.blockchain)
        self.simulator = HealthcareSimulator(self.blockchain)
        
        # Setup API routes
        self._setup_routes()
        
        print("Healthcare Diagnostics API initialized successfully")
    
    def _setup_routes(self) -> None:
        """Setup all API routes."""
        
        # Health check endpoint
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "blockchain_connected": self.blockchain.connected
            })
        
        # Model registration endpoint
        @self.app.route('/api/models/register', methods=['POST'])
        def register_model():
            """
            Register a new AI/ML model.
            
            Expected JSON payload:
            {
                "model_name": "CardioNet_v1",
                "model_path": "models/cardionet_v1.bin",
                "metadata": {
                    "accuracy": 0.95,
                    "version": "1.0",
                    "description": "Cardiovascular risk prediction model"
                }
            }
            """
            try:
                data = request.get_json()
                
                if not data or not all(k in data for k in ['model_name', 'model_path', 'metadata']):
                    return jsonify({
                        "status": "error",
                        "message": "Missing required fields: model_name, model_path, metadata"
                    }), 400
                
                model_hash = self.model_registry.register_model(
                    data['model_name'],
                    data['model_path'],
                    data['metadata']
                )
                
                return jsonify({
                    "status": "success",
                    "model_name": data['model_name'],
                    "model_hash": model_hash,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        # Diagnostic submission endpoint
        @self.app.route('/api/diagnostics/submit', methods=['POST'])
        def submit_diagnostic():
            """
            Submit a diagnostic case for AI analysis.
            
            Expected JSON payload:
            {
                "case_id": "case_001",
                "patient_data": {
                    "age": 65,
                    "gender": "M",
                    "symptoms": ["chest_pain", "shortness_of_breath"]
                },
                "clinical_data": {
                    "blood_pressure": [140, 90],
                    "heart_rate": 85,
                    "cholesterol": 220
                },
                "model_name": "CardioNet_v1",
                "explainability_method": "shap"
            }
            """
            try:
                data = request.get_json()
                
                if not data or not all(k in data for k in ['case_id', 'patient_data', 'clinical_data', 'model_name']):
                    return jsonify({
                        "status": "error",
                        "message": "Missing required fields"
                    }), 400
                
                # Combine input data
                input_data = {
                    **data['patient_data'],
                    **data['clinical_data']
                }
                
                # Simulate model prediction (in real implementation, this would call the actual model)
                prediction = self._simulate_model_prediction(data['model_name'], input_data)
                
                # Generate explanation
                explanation = self._generate_explanation(
                    data['model_name'],
                    input_data,
                    prediction,
                    data.get('explainability_method', 'shap')
                )
                
                # Log decision to audit trail
                decision_hash = self.audit_logger.log_decision(
                    data['case_id'],
                    input_data,
                    data['model_name'],
                    json.dumps(prediction),
                    explanation,
                    prediction.get('confidence', 0.85)
                )
                
                # Log data provenance
                provenance_hash = self.data_logger.log_sample(
                    f"sample_{data['case_id']}",
                    input_data,
                    {
                        "source": "api_submission",
                        "case_id": data['case_id'],
                        "submitted_at": datetime.now().isoformat()
                    }
                )
                
                return jsonify({
                    "status": "success",
                    "case_id": data['case_id'],
                    "prediction": prediction,
                    "explanation": explanation,
                    "decision_hash": decision_hash,
                    "provenance_hash": provenance_hash,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        # Audit trail endpoint
        @self.app.route('/api/audit/trail', methods=['GET'])
        def get_audit_trail():
            """
            Get audit trail entries.
            
            Query parameters:
            - model_id: Filter by model ID (optional)
            - limit: Maximum number of entries (default: 100)
            - case_id: Filter by case ID (optional)
            """
            try:
                model_id = request.args.get('model_id')
                limit = int(request.args.get('limit', 100))
                case_id = request.args.get('case_id')
                
                if case_id:
                    # Get audit for specific case
                    audit_info = self.audit_logger.get_decision_audit(case_id)
                    if audit_info:
                        return jsonify({
                            "status": "success",
                            "case_audit": audit_info
                        })
                    else:
                        return jsonify({
                            "status": "not_found",
                            "message": f"No audit found for case {case_id}"
                        }), 404
                
                # Get general audit trail
                result = self.blockchain.query_ledger("get_audit_trail", model_id, limit)
                
                return jsonify({
                    "status": "success",
                    "audit_trail": result.get("audit_trail", []),
                    "total_entries": result.get("total_entries", 0),
                    "retrieved_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        # Compliance report endpoint
        @self.app.route('/api/compliance/report', methods=['POST'])
        def generate_compliance_report():
            """
            Generate compliance report.
            
            Expected JSON payload:
            {
                "report_type": "model_report" | "decision_audit" | "compliance_summary",
                "model_name": "CardioNet_v1",  // for model_report
                "case_id": "case_001",         // for decision_audit
                "start_date": "2024-01-01",    // for compliance_summary
                "end_date": "2024-12-31"       // for compliance_summary
            }
            """
            try:
                data = request.get_json()
                
                if not data or 'report_type' not in data:
                    return jsonify({
                        "status": "error",
                        "message": "Missing report_type field"
                    }), 400
                
                report_type = data['report_type']
                
                if report_type == "model_report":
                    if 'model_name' not in data:
                        return jsonify({
                            "status": "error",
                            "message": "Missing model_name for model report"
                        }), 400
                    
                    report_path = self.report_generator.generate_model_report(data['model_name'])
                    
                elif report_type == "decision_audit":
                    if 'case_id' not in data:
                        return jsonify({
                            "status": "error",
                            "message": "Missing case_id for decision audit report"
                        }), 400
                    
                    report_path = self.report_generator.generate_decision_audit_report(data['case_id'])
                    
                elif report_type == "compliance_summary":
                    report_path = self.report_generator.generate_compliance_summary_report(
                        data.get('start_date'),
                        data.get('end_date')
                    )
                    
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Unknown report type: {report_type}"
                    }), 400
                
                # Read and return report content
                if report_path and os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report_content = json.load(f)
                    
                    return jsonify({
                        "status": "success",
                        "report": report_content,
                        "report_path": report_path,
                        "generated_at": datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Failed to generate report"
                    }), 500
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        # Simulation endpoint
        @self.app.route('/api/simulation/run', methods=['POST'])
        def run_simulation():
            """
            Run healthcare AI simulation.
            
            Expected JSON payload:
            {
                "num_patients": 50,
                "diagnostic_types": ["cardiology", "radiology"],
                "time_range_days": 30,
                "drift_probability": 0.1
            }
            """
            try:
                data = request.get_json() or {}
                
                # Create simulation config
                config = SimulationConfig(
                    num_patients=data.get('num_patients', 20),
                    time_range_days=data.get('time_range_days', 14),
                    drift_probability=data.get('drift_probability', 0.05)
                )
                
                # Run simulation
                simulator = HealthcareSimulator(self.blockchain, config)
                results = simulator.run_comprehensive_simulation()
                
                return jsonify({
                    "status": "success",
                    "simulation_results": results
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        # Model information endpoint
        @self.app.route('/api/models/<model_id>', methods=['GET'])
        def get_model_info(model_id: str):
            """Get information about a specific model."""
            try:
                model_info = self.model_registry.get_model(model_id)
                
                if model_info:
                    return jsonify({
                        "status": "success",
                        "model": model_info
                    })
                else:
                    return jsonify({
                        "status": "not_found",
                        "message": f"Model {model_id} not found"
                    }), 404
                    
            except Exception as e:
                return jsonify({
                    "status": "error", 
                    "message": str(e)
                }), 500
        
        # Model verification endpoint
        @self.app.route('/api/models/<model_id>/verify', methods=['POST'])
        def verify_model_integrity(model_id: str):
            """Verify model integrity against blockchain record."""
            try:
                data = request.get_json()
                
                if not data or 'model_path' not in data:
                    return jsonify({
                        "status": "error",
                        "message": "Missing model_path field"
                    }), 400
                
                is_valid = self.model_registry.verify_model_integrity(
                    model_id, data['model_path']
                )
                
                return jsonify({
                    "status": "success",
                    "model_id": model_id,
                    "integrity_verified": is_valid,
                    "verified_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
    
    def _simulate_model_prediction(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate model prediction (placeholder for actual model inference).
        
        Args:
            model_name: Name of the model to use
            input_data: Input features
            
        Returns:
            Prediction results
        """
        # In a real implementation, this would load and run the actual model
        # For demo purposes, we simulate predictions based on input data
        
        if "cardio" in model_name.lower():
            # Cardiovascular risk prediction
            age = input_data.get('age', 50)
            bp_systolic = input_data.get('blood_pressure_systolic', 120)
            cholesterol = input_data.get('cholesterol', 180)
            
            # Simple risk calculation
            risk_score = (age - 30) * 0.01 + (bp_systolic - 120) * 0.005 + (cholesterol - 180) * 0.002
            risk_score = max(0.1, min(0.9, risk_score))
            
            if risk_score < 0.3:
                diagnosis = "low_risk"
            elif risk_score < 0.6:
                diagnosis = "medium_risk"  
            else:
                diagnosis = "high_risk"
            
            return {
                "diagnosis": diagnosis,
                "risk_score": round(risk_score, 3),
                "confidence": 0.85 + (risk_score * 0.1),
                "recommendations": self._get_cardio_recommendations(diagnosis)
            }
        
        else:
            # Generic prediction
            return {
                "diagnosis": "normal",
                "confidence": 0.82,
                "recommendations": ["Regular follow-up recommended"]
            }
    
    def _generate_explanation(self, model_name: str, input_data: Dict[str, Any], 
                            prediction: Dict[str, Any], method: str = "shap") -> Dict[str, Any]:
        """
        Generate explanation for prediction.
        
        Args:
            model_name: Name of the model
            input_data: Input features
            prediction: Model prediction
            method: Explanation method ("shap" or "lime")
            
        Returns:
            Explanation results
        """
        try:
            if method.lower() == "lime":
                explainer = LIMEExplainer(None)  # Mock model
                explanation = explainer.explain(input_data)
            else:
                explainer = SHAPExplainer(None)  # Mock model
                explanation = explainer.explain(input_data)
            
            return explanation
            
        except Exception as e:
            return {
                "explanation_type": method,
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _get_cardio_recommendations(self, diagnosis: str) -> List[str]:
        """Get cardiovascular recommendations based on diagnosis."""
        recommendations_map = {
            "low_risk": [
                "Maintain healthy lifestyle",
                "Regular exercise recommended", 
                "Annual checkup advised"
            ],
            "medium_risk": [
                "Lifestyle modifications recommended",
                "Consider dietary changes",
                "Monitor blood pressure regularly",
                "Follow-up in 6 months"
            ],
            "high_risk": [
                "Immediate lifestyle changes required",
                "Medication may be necessary",
                "Frequent monitoring required",
                "Cardiology consultation recommended"
            ]
        }
        
        return recommendations_map.get(diagnosis, ["Consult with physician"])
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = True) -> None:
        """
        Run the Flask API server.
        
        Args:
            host: Host address to bind to
            port: Port number to bind to
            debug: Enable debug mode
        """
        print(f"Starting Healthcare Diagnostics API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app(config_path: str = "config.yaml") -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application
    """
    api = HealthcareDiagnosticsAPI(config_path)
    return api.app


# CLI entry point
if __name__ == '__main__':
    api = HealthcareDiagnosticsAPI()
    api.run()