"""
Simulator: Simulates data, model predictions, and process flows for testing.
"""
import json
import logging
import numpy as np
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using mock ML models.")


class Simulator:
    """
    Simulation framework for testing the complete blockchain-enabled AI pipeline.
    
    This class generates synthetic healthcare data, trains models, makes predictions,
    and logs all activities to the blockchain for comprehensive testing and demonstration.
    """
    
    def __init__(self, blockchain_connector, model_registry, provenance_logger, 
                 audit_logger, report_generator, config=None):
        """
        Initialize the simulation framework.
        
        Args:
            blockchain_connector: Blockchain connector instance
            model_registry: Model registry instance
            provenance_logger: Data provenance logger instance
            audit_logger: Decision audit logger instance
            report_generator: Regulatory report generator instance
            config: Configuration dictionary
        """
        self.blockchain = blockchain_connector
        self.model_registry = model_registry
        self.provenance_logger = provenance_logger
        self.audit_logger = audit_logger
        self.report_generator = report_generator
        
        self.config = config or self._default_config()
        self._logger = logging.getLogger(__name__)
        
        # Simulation state
        self.simulation_id = self._generate_simulation_id()
        self.models = {}
        self.datasets = {}
        self.simulation_log = []
    
    def run(self, scenario: str = "healthcare_diagnostic") -> Dict[str, Any]:
        """
        Run a complete simulation scenario.
        
        Args:
            scenario: Type of simulation scenario to run
            
        Returns:
            Dictionary containing simulation results and summary
        """
        try:
            self._logger.info(f"Starting simulation: {scenario} (ID: {self.simulation_id})")
            
            if scenario == "healthcare_diagnostic":
                return self._run_healthcare_diagnostic_scenario()
            elif scenario == "model_lifecycle":
                return self._run_model_lifecycle_scenario()
            elif scenario == "compliance_audit":
                return self._run_compliance_audit_scenario()
            else:
                raise ValueError(f"Unknown scenario: {scenario}")
                
        except Exception as e:
            self._logger.error(f"Simulation failed: {e}")
            return {"error": str(e), "simulation_id": self.simulation_id}
    
    def _run_healthcare_diagnostic_scenario(self) -> Dict[str, Any]:
        """Run healthcare diagnostic simulation scenario."""
        results = {
            "scenario": "healthcare_diagnostic",
            "simulation_id": self.simulation_id,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        try:
            # Step 1: Generate synthetic patient data
            self._log_step("Generating synthetic patient data")
            patient_data = self._generate_patient_data()
            results["steps"].append({"step": 1, "description": "Patient data generated", 
                                   "records": len(patient_data)})
            
            # Step 2: Log data provenance
            self._log_step("Logging data provenance")
            provenance_hashes = self._log_patient_provenance(patient_data)
            results["steps"].append({"step": 2, "description": "Data provenance logged", 
                                   "hashes": len(provenance_hashes)})
            
            # Step 3: Train diagnostic model
            self._log_step("Training diagnostic model")
            model_info = self._train_diagnostic_model(patient_data)
            results["steps"].append({"step": 3, "description": "Model trained and registered",
                                   "model_name": model_info["name"]})
            
            # Step 4: Make predictions and log decisions
            self._log_step("Making diagnostic predictions")
            predictions = self._make_diagnostic_predictions(model_info, patient_data)
            results["steps"].append({"step": 4, "description": "Predictions made and logged",
                                   "predictions": len(predictions)})
            
            # Step 5: Generate explainability
            self._log_step("Generating explanations")
            explanations = self._generate_explanations(model_info, patient_data, predictions)
            results["steps"].append({"step": 5, "description": "Explanations generated",
                                   "explanations": len(explanations)})
            
            # Step 6: Generate regulatory reports
            self._log_step("Generating regulatory reports")
            reports = self._generate_simulation_reports(model_info["name"])
            results["steps"].append({"step": 6, "description": "Reports generated",
                                   "report_files": reports})
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            results["summary"] = self._generate_simulation_summary()
            
            self._logger.info("Healthcare diagnostic simulation completed successfully")
            return results
            
        except Exception as e:
            self._logger.error(f"Healthcare diagnostic simulation failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
    
    def _run_model_lifecycle_scenario(self) -> Dict[str, Any]:
        """Run model lifecycle simulation scenario."""
        results = {
            "scenario": "model_lifecycle",
            "simulation_id": self.simulation_id,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        try:
            # Simulate multiple model versions
            for version in range(1, 4):  # 3 model versions
                self._log_step(f"Training model version {version}")
                
                # Generate data for this version
                data = self._generate_patient_data(seed=42+version)
                
                # Train model
                model_info = self._train_diagnostic_model(data, version=f"v{version}")
                
                # Make some predictions
                predictions = self._make_diagnostic_predictions(model_info, data[:10])  # Limited predictions
                
                results["steps"].append({
                    "step": version, 
                    "description": f"Model v{version} trained and tested",
                    "model_name": model_info["name"],
                    "predictions": len(predictions)
                })
            
            # Generate lifecycle report
            reports = self._generate_simulation_reports("HealthcareDiagnosticModel")
            results["steps"].append({"step": 4, "description": "Lifecycle reports generated",
                                   "report_files": reports})
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            results["summary"] = {"models_created": 3, "total_predictions": 30}
            
            return results
            
        except Exception as e:
            self._logger.error(f"Model lifecycle simulation failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
    
    def _run_compliance_audit_scenario(self) -> Dict[str, Any]:
        """Run compliance audit simulation scenario."""
        results = {
            "scenario": "compliance_audit",
            "simulation_id": self.simulation_id,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        try:
            # Generate test data and models
            data = self._generate_patient_data()
            model_info = self._train_diagnostic_model(data)
            predictions = self._make_diagnostic_predictions(model_info, data)
            
            # Generate comprehensive audit reports
            model_report = self.report_generator.generate_model_report(model_info["name"])
            
            # Generate decision audit reports for random cases
            case_reports = []
            for i in range(min(5, len(predictions))):
                case_id = f"case_{i+1}"
                report = self.report_generator.generate_decision_audit_report(case_id)
                if report:
                    case_reports.append(report)
            
            # Generate comprehensive report
            comprehensive_report = self.report_generator.generate_comprehensive_audit_report()
            
            results["steps"].append({
                "step": 1,
                "description": "Compliance audit completed",
                "model_report": model_report,
                "case_reports": len(case_reports),
                "comprehensive_report": comprehensive_report
            })
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            results["summary"] = {
                "audit_reports": len(case_reports) + 2,  # case reports + model + comprehensive
                "compliance_status": "assessed"
            }
            
            return results
            
        except Exception as e:
            self._logger.error(f"Compliance audit simulation failed: {e}")
            results["error"] = str(e)
            results["status"] = "failed"
            return results
    
    def _generate_patient_data(self, num_samples: int = None, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate synthetic patient data for healthcare simulation."""
        if num_samples is None:
            num_samples = self.config.get('num_samples', 100)
        
        np.random.seed(seed)
        
        patient_data = []
        
        for i in range(num_samples):
            # Generate realistic healthcare features
            patient = {
                'patient_id': f"P{i+1:04d}",
                'age': int(np.random.normal(50, 15)),  # Age distribution
                'gender': str(np.random.choice(['M', 'F'], p=[0.48, 0.52])),
                'blood_pressure_systolic': int(np.random.normal(130, 20)),
                'blood_pressure_diastolic': int(np.random.normal(85, 12)),
                'cholesterol': int(np.random.normal(200, 40)),
                'blood_sugar': int(np.random.normal(100, 25)),
                'heart_rate': int(np.random.normal(75, 12)),
                'bmi': float(round(np.random.normal(26, 5), 1)),
                'smoking_history': int(np.random.choice([0, 1], p=[0.7, 0.3])),
                'family_history': int(np.random.choice([0, 1], p=[0.6, 0.4])),
            }
            
            # Ensure realistic bounds
            patient['age'] = max(18, min(100, patient['age']))
            patient['blood_pressure_systolic'] = max(90, min(200, patient['blood_pressure_systolic']))
            patient['blood_pressure_diastolic'] = max(60, min(120, patient['blood_pressure_diastolic']))
            patient['cholesterol'] = max(120, min(400, patient['cholesterol']))
            patient['blood_sugar'] = max(70, min(300, patient['blood_sugar']))
            patient['heart_rate'] = max(50, min(120, patient['heart_rate']))
            patient['bmi'] = max(15, min(50, patient['bmi']))
            
            # Generate target variable (risk assessment)
            # Higher risk if multiple risk factors present
            risk_score = (
                (patient['age'] > 60) * 0.2 +
                (patient['blood_pressure_systolic'] > 140) * 0.3 +
                (patient['cholesterol'] > 240) * 0.2 +
                (patient['bmi'] > 30) * 0.1 +
                patient['smoking_history'] * 0.1 +
                patient['family_history'] * 0.1
            )
            
            patient['risk_level'] = 1 if risk_score > 0.4 else 0  # Binary classification
            
            patient_data.append(patient)
        
        return patient_data
    
    def _log_patient_provenance(self, patient_data: List[Dict[str, Any]]) -> List[str]:
        """Log provenance information for patient data."""
        provenance_hashes = []
        
        for i, patient in enumerate(patient_data):
            provenance_info = {
                'source': 'Simulated_Healthcare_System',
                'date': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                'method': 'electronic_health_record',
                'quality_score': np.random.uniform(0.85, 0.99),
                'preprocessing_steps': ['data_validation', 'range_checking', 'anonymization'],
                'simulation_id': self.simulation_id,
                'batch_id': f"batch_{i//10}"  # Group in batches of 10
            }
            
            sample_hash = self.provenance_logger.log_sample(
                patient['patient_id'], patient, provenance_info
            )
            
            if sample_hash:
                provenance_hashes.append(sample_hash)
        
        return provenance_hashes
    
    def _train_diagnostic_model(self, patient_data: List[Dict[str, Any]], 
                              version: str = "v1") -> Dict[str, Any]:
        """Train a diagnostic model on patient data."""
        model_name = f"HealthcareDiagnosticModel_{version}"
        
        if SKLEARN_AVAILABLE:
            # Prepare data for sklearn
            feature_names = ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                           'cholesterol', 'blood_sugar', 'heart_rate', 'bmi', 
                           'smoking_history', 'family_history']
            
            X = np.array([[patient[feature] for feature in feature_names] 
                         for patient in patient_data])
            y = np.array([patient['risk_level'] for patient in patient_data])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            model_path = f"models/{model_name}.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            
        else:
            # Mock model for when sklearn is not available
            accuracy = 0.87  # Mock accuracy
            model_path = f"models/{model_name}_mock.bin"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                f.write(os.urandom(1024))  # Mock model content
        
        # Register model
        metadata = {
            'accuracy': float(accuracy),
            'training_date': datetime.now().isoformat(),
            'training_samples': len(patient_data),
            'model_type': 'RandomForestClassifier',
            'features': ['age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                        'cholesterol', 'blood_sugar', 'heart_rate', 'bmi', 
                        'smoking_history', 'family_history'],
            'target': 'cardiovascular_risk',
            'simulation_id': self.simulation_id,
            'version': version
        }
        
        model_hash = self.model_registry.register_model(model_name, model_path, metadata)
        
        model_info = {
            'name': model_name,
            'path': model_path,
            'hash': model_hash,
            'metadata': metadata,
            'sklearn_available': SKLEARN_AVAILABLE
        }
        
        self.models[model_name] = model_info
        return model_info
    
    def _make_diagnostic_predictions(self, model_info: Dict[str, Any], 
                                   patient_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions using the trained model."""
        predictions = []
        
        feature_names = model_info['metadata']['features']
        
        for i, patient in enumerate(patient_data):
            case_id = f"case_{i+1}"
            
            # Prepare input data
            input_data = {feature: patient[feature] for feature in feature_names}
            
            # Make prediction
            if model_info['sklearn_available'] and SKLEARN_AVAILABLE:
                # Load actual model
                model = joblib.load(model_info['path'])
                X = np.array([[patient[feature] for feature in feature_names]])
                prediction = model.predict(X)[0]
                confidence = float(np.max(model.predict_proba(X)))
            else:
                # Mock prediction
                prediction = np.random.choice([0, 1], p=[0.7, 0.3])
                confidence = np.random.uniform(0.6, 0.95)
            
            # Convert prediction to human-readable format
            decision = "High cardiovascular risk" if prediction == 1 else "Low cardiovascular risk"
            
            # Generate mock explanation
            explanation = {
                'method': 'feature_importance',
                'top_features': {
                    'cholesterol': float(np.random.uniform(-0.3, 0.3)),
                    'age': float(np.random.uniform(-0.2, 0.2)),
                    'blood_pressure_systolic': float(np.random.uniform(-0.25, 0.25))
                }
            }
            
            # Log decision
            self.audit_logger.log_decision(
                case_id=case_id,
                input_data=input_data,
                model_name=model_info['name'],
                decision=decision,
                explanation=explanation,
                confidence=confidence,
                metadata={
                    'simulation_id': self.simulation_id,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'patient_id': patient['patient_id']
                }
            )
            
            predictions.append({
                'case_id': case_id,
                'patient_id': patient['patient_id'],
                'prediction': prediction,
                'decision': decision,
                'confidence': confidence,
                'explanation': explanation
            })
        
        return predictions
    
    def _generate_explanations(self, model_info: Dict[str, Any], 
                             patient_data: List[Dict[str, Any]], 
                             predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed explanations using explainability frameworks."""
        explanations = []
        
        # Import explainability modules
        try:
            from framework.explainability.shap_explainer import SHAPExplainer
            from framework.explainability.lime_explainer import LIMEExplainer
            
            if model_info['sklearn_available'] and SKLEARN_AVAILABLE:
                # Load the actual model
                model = joblib.load(model_info['path'])
                feature_names = model_info['metadata']['features']
                
                # Prepare background data for explainers
                X_background = np.array([[patient[feature] for feature in feature_names] 
                                       for patient in patient_data[:20]])  # Use first 20 as background
                
                # Initialize explainers
                shap_explainer = SHAPExplainer(
                    model, 
                    model_type="tree", 
                    feature_names=feature_names,
                    background_data=X_background
                )
                
                lime_explainer = LIMEExplainer(
                    model,
                    training_data=X_background,
                    feature_names=feature_names
                )
                
                # Generate explanations for a subset of predictions
                for i, (patient, prediction) in enumerate(zip(patient_data[:5], predictions[:5])):
                    input_data = [patient[feature] for feature in feature_names]
                    
                    # SHAP explanation
                    shap_explanation = shap_explainer.explain(input_data, return_format="detailed")
                    
                    # LIME explanation
                    lime_explanation = lime_explainer.explain(input_data, return_format="detailed")
                    
                    explanations.append({
                        'case_id': prediction['case_id'],
                        'patient_id': patient['patient_id'],
                        'shap_explanation': shap_explanation,
                        'lime_explanation': lime_explanation
                    })
            else:
                # Mock explanations
                for prediction in predictions[:5]:
                    explanations.append({
                        'case_id': prediction['case_id'],
                        'shap_explanation': {'method': 'SHAP (Mock)', 'warning': 'Mock implementation'},
                        'lime_explanation': {'method': 'LIME (Mock)', 'warning': 'Mock implementation'}
                    })
                    
        except Exception as e:
            self._logger.warning(f"Could not generate detailed explanations: {e}")
            # Provide basic mock explanations
            for prediction in predictions[:5]:
                explanations.append({
                    'case_id': prediction['case_id'],
                    'basic_explanation': prediction['explanation']
                })
        
        return explanations
    
    def _generate_simulation_reports(self, model_name: str) -> List[str]:
        """Generate regulatory reports for the simulation."""
        reports = []
        
        try:
            # Generate model report
            model_report = self.report_generator.generate_model_report(model_name)
            if model_report:
                reports.append(model_report)
            
            # Generate comprehensive audit report
            comprehensive_report = self.report_generator.generate_comprehensive_audit_report()
            if comprehensive_report:
                reports.append(comprehensive_report)
            
        except Exception as e:
            self._logger.warning(f"Could not generate all reports: {e}")
        
        return reports
    
    def _generate_simulation_summary(self) -> Dict[str, Any]:
        """Generate summary of simulation results."""
        return {
            'simulation_id': self.simulation_id,
            'models_trained': len(self.models),
            'blockchain_connected': self.blockchain.is_connected(),
            'total_steps': len(self.simulation_log),
            'completion_time': datetime.now().isoformat()
        }
    
    def _log_step(self, step_description: str):
        """Log a simulation step."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': len(self.simulation_log) + 1,
            'description': step_description
        }
        self.simulation_log.append(log_entry)
        self._logger.info(f"Simulation step {log_entry['step']}: {step_description}")
    
    def _generate_simulation_id(self) -> str:
        """Generate unique simulation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"SIM_{timestamp}"
    
    def _default_config(self) -> Dict[str, Any]:
        """Default simulation configuration."""
        return {
            'num_samples': 50,
            'random_seed': 42,
            'model_type': 'RandomForestClassifier',
            'test_size': 0.2,
            'reports_format': 'json'
        }