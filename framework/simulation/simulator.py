"""
Simulator: Simulates data, model predictions, and process flows for testing.

This module provides comprehensive simulation capabilities for healthcare AI/ML systems,
including patient data generation, diagnostic workflows, model drift scenarios,
and regulatory compliance testing.
"""

import json
import random
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DiagnosticType(Enum):
    """Types of diagnostic cases."""
    CARDIOLOGY = "cardiology"
    RADIOLOGY = "radiology" 
    PATHOLOGY = "pathology"
    GENERAL = "general"


class PatientRisk(Enum):
    """Patient risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    num_patients: int = 100
    diagnostic_types: List[DiagnosticType] = None
    risk_distribution: Dict[PatientRisk, float] = None
    time_range_days: int = 30
    noise_level: float = 0.1
    drift_probability: float = 0.05
    
    def __post_init__(self):
        if self.diagnostic_types is None:
            self.diagnostic_types = list(DiagnosticType)
        if self.risk_distribution is None:
            self.risk_distribution = {
                PatientRisk.LOW: 0.4,
                PatientRisk.MEDIUM: 0.3,
                PatientRisk.HIGH: 0.2,
                PatientRisk.CRITICAL: 0.1
            }


class HealthcareSimulator:
    """
    Comprehensive healthcare AI/ML system simulator.
    
    This class simulates various aspects of a healthcare AI system including
    patient data, diagnostic workflows, model predictions, and regulatory scenarios.
    """
    
    def __init__(self, blockchain_connector=None, config: SimulationConfig = None):
        """
        Initialize the healthcare simulator.
        
        Args:
            blockchain_connector: Optional blockchain connector for logging
            config: Simulation configuration parameters
        """
        self.blockchain_connector = blockchain_connector
        self.config = config or SimulationConfig()
        self.patients = []
        self.diagnostic_sessions = []
        self.models = {}
        self.simulation_state = {}
        
        # Initialize simulation components
        self._setup_models()
        
    def run_comprehensive_simulation(self) -> Dict[str, Any]:
        """
        Run a comprehensive simulation of the healthcare AI system.
        
        Returns:
            Comprehensive simulation results
        """
        print("Starting comprehensive healthcare AI simulation...")
        
        results = {
            "simulation_metadata": {
                "started_at": datetime.now().isoformat(),
                "config": self._config_to_dict(),
                "simulation_id": self._generate_simulation_id()
            },
            "phases": {}
        }
        
        # Phase 1: Generate synthetic patients
        print("Phase 1: Generating synthetic patients...")
        patients_result = self.generate_synthetic_patients(self.config.num_patients)
        results["phases"]["patient_generation"] = patients_result
        
        # Phase 2: Simulate diagnostic workflows
        print("Phase 2: Simulating diagnostic workflows...")
        diagnostics_result = self.simulate_diagnostic_workflows(patients_result["patients"])
        results["phases"]["diagnostic_workflows"] = diagnostics_result
        
        # Phase 3: Simulate model predictions and explanations
        print("Phase 3: Simulating model predictions...")
        predictions_result = self.simulate_model_predictions(diagnostics_result["sessions"])
        results["phases"]["model_predictions"] = predictions_result
        
        # Phase 4: Simulate regulatory compliance scenarios
        print("Phase 4: Simulating regulatory compliance...")
        compliance_result = self.simulate_regulatory_scenarios()
        results["phases"]["regulatory_compliance"] = compliance_result
        
        # Phase 5: Simulate model drift detection
        print("Phase 5: Simulating model drift...")
        drift_result = self.simulate_model_drift_scenarios()
        results["phases"]["model_drift"] = drift_result
        
        # Generate summary
        results["summary"] = self._generate_simulation_summary(results)
        results["completed_at"] = datetime.now().isoformat()
        
        print("Comprehensive simulation completed successfully!")
        return results
    
    def generate_synthetic_patients(self, num_patients: int) -> Dict[str, Any]:
        """
        Generate synthetic patient data for simulation.
        
        Args:
            num_patients: Number of patients to generate
            
        Returns:
            Dictionary containing generated patients and metadata
        """
        patients = []
        
        for i in range(num_patients):
            # Determine risk level based on distribution
            risk_level = self._sample_risk_level()
            
            # Generate patient demographics
            age = self._generate_age_by_risk(risk_level)
            gender = random.choice(["M", "F", "Other"])
            
            # Generate clinical features based on risk
            clinical_features = self._generate_clinical_features(risk_level, age, gender)
            
            # Generate patient record
            patient = {
                "patient_id": f"SIM_P_{i:04d}",
                "demographics": {
                    "age": age,
                    "gender": gender,
                    "risk_level": risk_level.value
                },
                "clinical_features": clinical_features,
                "medical_history": self._generate_medical_history(risk_level),
                "generated_at": datetime.now().isoformat()
            }
            
            patients.append(patient)
        
        self.patients = patients
        
        return {
            "status": "success",
            "patients": patients,
            "count": len(patients),
            "risk_distribution": self._analyze_risk_distribution(patients)
        }
    
    def simulate_diagnostic_workflows(self, patients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate diagnostic workflows for patients.
        
        Args:
            patients: List of patient records
            
        Returns:
            Dictionary containing diagnostic sessions
        """
        diagnostic_sessions = []
        
        for patient in patients:
            # Each patient may have multiple diagnostic sessions
            num_sessions = random.randint(1, 3)
            
            for session_idx in range(num_sessions):
                # Determine diagnostic type
                diagnostic_type = random.choice(self.config.diagnostic_types)
                
                # Generate session data
                session = {
                    "session_id": f"SIM_S_{patient['patient_id']}_{session_idx}",
                    "patient_id": patient["patient_id"],
                    "diagnostic_type": diagnostic_type.value,
                    "timestamp": self._random_timestamp(),
                    "presenting_symptoms": self._generate_symptoms(diagnostic_type, patient),
                    "clinical_measurements": self._generate_measurements(diagnostic_type, patient),
                    "workflow_steps": self._simulate_workflow_steps(diagnostic_type)
                }
                
                diagnostic_sessions.append(session)
        
        self.diagnostic_sessions = diagnostic_sessions
        
        return {
            "status": "success",
            "sessions": diagnostic_sessions,
            "count": len(diagnostic_sessions),
            "diagnostic_type_distribution": self._analyze_diagnostic_types(diagnostic_sessions)
        }
    
    def simulate_model_predictions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate AI model predictions for diagnostic sessions.
        
        Args:
            sessions: List of diagnostic sessions
            
        Returns:
            Dictionary containing prediction results
        """
        predictions = []
        
        for session in sessions:
            # Select appropriate model for diagnostic type
            model_name = self._select_model_for_diagnostic(session["diagnostic_type"])
            model = self.models.get(model_name)
            
            if not model:
                continue
            
            # Prepare input features
            input_features = {
                **session["clinical_measurements"],
                "age": self._get_patient_age(session["patient_id"]),
                "symptoms_severity": random.uniform(0.1, 1.0)
            }
            
            # Generate prediction
            prediction_result = model.predict(input_features)
            
            # Generate explanation
            explanation = model.explain_prediction(input_features, prediction_result)
            
            # Create prediction record
            prediction = {
                "prediction_id": f"SIM_PRED_{session['session_id']}",
                "session_id": session["session_id"],
                "patient_id": session["patient_id"],
                "model_name": model_name,
                "input_features": input_features,
                "prediction": prediction_result,
                "explanation": explanation,
                "confidence": random.uniform(0.7, 0.98),
                "timestamp": datetime.now().isoformat()
            }
            
            predictions.append(prediction)
            
            # Log to blockchain if connector available
            if self.blockchain_connector:
                try:
                    self.blockchain_connector.invoke_transaction(
                        "log_diagnostic",
                        prediction["prediction_id"],
                        session["patient_id"],
                        model_name,
                        input_features,
                        prediction_result,
                        prediction["confidence"],
                        explanation
                    )
                except Exception as e:
                    print(f"Warning: Could not log prediction to blockchain: {e}")
        
        return {
            "status": "success",
            "predictions": predictions,
            "count": len(predictions),
            "model_usage": self._analyze_model_usage(predictions)
        }
    
    def simulate_regulatory_scenarios(self) -> Dict[str, Any]:
        """
        Simulate various regulatory compliance scenarios.
        
        Returns:
            Dictionary containing regulatory simulation results
        """
        scenarios = []
        
        # Scenario 1: HIPAA Compliance Audit
        hipaa_scenario = {
            "scenario_type": "hipaa_audit",
            "description": "Simulated HIPAA compliance audit",
            "compliance_checks": {
                "data_anonymization": self._check_data_anonymization(),
                "access_controls": self._check_access_controls(),
                "audit_logging": self._check_audit_logging(),
                "patient_consent": self._check_patient_consent()
            },
            "overall_score": random.uniform(0.85, 0.98)
        }
        scenarios.append(hipaa_scenario)
        
        # Scenario 2: FDA AI/ML Validation
        fda_scenario = {
            "scenario_type": "fda_validation",
            "description": "Simulated FDA AI/ML device validation",
            "validation_checks": {
                "model_validation": self._check_model_validation(),
                "clinical_evidence": self._check_clinical_evidence(),
                "risk_management": self._check_risk_management(),
                "performance_monitoring": self._check_performance_monitoring()
            },
            "overall_score": random.uniform(0.80, 0.95)
        }
        scenarios.append(fda_scenario)
        
        # Scenario 3: GDPR Compliance
        gdpr_scenario = {
            "scenario_type": "gdpr_compliance",
            "description": "Simulated GDPR compliance assessment",
            "compliance_checks": {
                "data_minimization": self._check_data_minimization(),
                "right_to_explanation": self._check_explainability(),
                "data_portability": self._check_data_portability(),
                "privacy_by_design": self._check_privacy_design()
            },
            "overall_score": random.uniform(0.82, 0.96)
        }
        scenarios.append(gdpr_scenario)
        
        return {
            "status": "success",
            "scenarios": scenarios,
            "summary": {
                "total_scenarios": len(scenarios),
                "average_compliance_score": sum(s["overall_score"] for s in scenarios) / len(scenarios)
            }
        }
    
    def simulate_model_drift_scenarios(self) -> Dict[str, Any]:
        """
        Simulate model drift detection scenarios.
        
        Returns:
            Dictionary containing drift simulation results
        """
        drift_scenarios = []
        
        for model_name, model in self.models.items():
            # Simulate different types of drift
            drift_types = ["data_drift", "concept_drift", "performance_drift"]
            
            for drift_type in drift_types:
                if random.random() < self.config.drift_probability:
                    # Generate drift metrics
                    drift_metrics = self._generate_drift_metrics(drift_type)
                    
                    # Check if drift detected
                    drift_detected = any(metric > 0.1 for metric in drift_metrics.values())
                    
                    scenario = {
                        "model_name": model_name,
                        "drift_type": drift_type,
                        "drift_metrics": drift_metrics,
                        "drift_detected": drift_detected,
                        "timestamp": datetime.now().isoformat(),
                        "recommended_action": self._get_drift_action(drift_type, drift_detected)
                    }
                    
                    drift_scenarios.append(scenario)
                    
                    # Log to blockchain if connector available
                    if self.blockchain_connector and drift_detected:
                        try:
                            self.blockchain_connector.invoke_transaction(
                                "detect_model_drift",
                                model_name,
                                drift_metrics,
                                0.1  # threshold
                            )
                        except Exception as e:
                            print(f"Warning: Could not log drift to blockchain: {e}")
        
        return {
            "status": "success",
            "drift_scenarios": drift_scenarios,
            "summary": {
                "total_scenarios": len(drift_scenarios),
                "drift_detected_count": sum(1 for s in drift_scenarios if s["drift_detected"]),
                "models_affected": len(set(s["model_name"] for s in drift_scenarios))
            }
        }
    
    def _setup_models(self) -> None:
        """Initialize simulation models."""
        self.models = {
            "CardioNet_v1": MockDiagnosticModel("CardioNet_v1", DiagnosticType.CARDIOLOGY),
            "RadiologyAI_v2": MockDiagnosticModel("RadiologyAI_v2", DiagnosticType.RADIOLOGY),
            "PathologyNet_v1": MockDiagnosticModel("PathologyNet_v1", DiagnosticType.PATHOLOGY),
            "GeneralDx_v3": MockDiagnosticModel("GeneralDx_v3", DiagnosticType.GENERAL)
        }
    
    def _sample_risk_level(self) -> PatientRisk:
        """Sample risk level based on distribution."""
        rand = random.random()
        cumulative = 0
        
        for risk, prob in self.config.risk_distribution.items():
            cumulative += prob
            if rand <= cumulative:
                return risk
        
        return PatientRisk.LOW
    
    def _generate_age_by_risk(self, risk_level: PatientRisk) -> int:
        """Generate age based on risk level."""
        if risk_level == PatientRisk.LOW:
            return random.randint(18, 45)
        elif risk_level == PatientRisk.MEDIUM:
            return random.randint(30, 65)
        elif risk_level == PatientRisk.HIGH:
            return random.randint(50, 80)
        else:  # CRITICAL
            return random.randint(60, 90)
    
    def _generate_clinical_features(self, risk_level: PatientRisk, age: int, gender: str) -> Dict[str, Any]:
        """Generate clinical features based on patient characteristics."""
        base_values = {
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "heart_rate": 70,
            "cholesterol": 180,
            "blood_sugar": 90,
            "bmi": 22
        }
        
        # Adjust based on risk level
        risk_multipliers = {
            PatientRisk.LOW: 1.0,
            PatientRisk.MEDIUM: 1.2,
            PatientRisk.HIGH: 1.4,
            PatientRisk.CRITICAL: 1.6
        }
        
        multiplier = risk_multipliers[risk_level]
        noise_level = self.config.noise_level
        
        features = {}
        for feature, base_value in base_values.items():
            # Apply risk multiplier and noise
            value = base_value * multiplier * random.uniform(1 - noise_level, 1 + noise_level)
            
            # Age adjustment
            if age > 50:
                value *= 1.1
            
            features[feature] = round(value, 2)
        
        return features
    
    def _generate_medical_history(self, risk_level: PatientRisk) -> Dict[str, Any]:
        """Generate medical history based on risk level."""
        conditions = ["hypertension", "diabetes", "heart_disease", "stroke", "cancer"]
        
        history = {}
        for condition in conditions:
            # Higher risk patients more likely to have conditions
            probability = 0.1 * (list(PatientRisk).index(risk_level) + 1)
            history[condition] = random.random() < probability
        
        return history
    
    def _random_timestamp(self) -> str:
        """Generate random timestamp within the configured range."""
        start_date = datetime.now() - timedelta(days=self.config.time_range_days)
        end_date = datetime.now()
        
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        return random_date.isoformat()
    
    def _generate_symptoms(self, diagnostic_type: DiagnosticType, patient: Dict[str, Any]) -> List[str]:
        """Generate symptoms based on diagnostic type and patient."""
        symptom_map = {
            DiagnosticType.CARDIOLOGY: ["chest_pain", "shortness_of_breath", "palpitations", "fatigue"],
            DiagnosticType.RADIOLOGY: ["headache", "abdominal_pain", "back_pain", "joint_pain"],
            DiagnosticType.PATHOLOGY: ["unusual_growth", "skin_changes", "persistent_cough", "weight_loss"],
            DiagnosticType.GENERAL: ["fever", "nausea", "dizziness", "general_malaise"]
        }
        
        available_symptoms = symptom_map[diagnostic_type]
        num_symptoms = random.randint(1, 3)
        
        return random.sample(available_symptoms, min(num_symptoms, len(available_symptoms)))
    
    def _generate_measurements(self, diagnostic_type: DiagnosticType, patient: Dict[str, Any]) -> Dict[str, float]:
        """Generate clinical measurements for diagnostic type."""
        measurements = {}
        
        if diagnostic_type == DiagnosticType.CARDIOLOGY:
            measurements.update({
                "ecg_score": random.uniform(0.1, 1.0),
                "ejection_fraction": random.uniform(0.4, 0.7),
                "troponin_level": random.uniform(0, 0.5)
            })
        elif diagnostic_type == DiagnosticType.RADIOLOGY:
            measurements.update({
                "image_quality_score": random.uniform(0.7, 1.0),
                "lesion_size": random.uniform(0, 5.0),
                "contrast_enhancement": random.uniform(0.2, 0.8)
            })
        
        # Add common measurements
        measurements.update({
            "measurement_confidence": random.uniform(0.8, 1.0),
            "data_quality_score": random.uniform(0.85, 1.0)
        })
        
        return measurements
    
    def _simulate_workflow_steps(self, diagnostic_type: DiagnosticType) -> List[Dict[str, Any]]:
        """Simulate workflow steps for diagnostic process."""
        base_steps = [
            {"step": "patient_intake", "duration_minutes": random.randint(5, 15)},
            {"step": "initial_assessment", "duration_minutes": random.randint(10, 30)},
            {"step": "data_collection", "duration_minutes": random.randint(15, 45)}
        ]
        
        if diagnostic_type == DiagnosticType.RADIOLOGY:
            base_steps.append({"step": "imaging", "duration_minutes": random.randint(20, 60)})
        
        base_steps.extend([
            {"step": "ai_analysis", "duration_minutes": random.randint(2, 10)},
            {"step": "physician_review", "duration_minutes": random.randint(10, 30)},
            {"step": "report_generation", "duration_minutes": random.randint(5, 15)}
        ])
        
        return base_steps
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "num_patients": self.config.num_patients,
            "diagnostic_types": [dt.value for dt in self.config.diagnostic_types],
            "risk_distribution": {k.value: v for k, v in self.config.risk_distribution.items()},
            "time_range_days": self.config.time_range_days,
            "noise_level": self.config.noise_level,
            "drift_probability": self.config.drift_probability
        }
    
    def _generate_simulation_id(self) -> str:
        """Generate unique simulation ID."""
        return f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
    
    # Additional helper methods for analysis and simulation would go here...
    # (Truncated for brevity, but would include all the referenced methods)
    
    def _analyze_risk_distribution(self, patients: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze risk distribution in generated patients."""
        distribution = {}
        for patient in patients:
            risk = patient["demographics"]["risk_level"]
            distribution[risk] = distribution.get(risk, 0) + 1
        return distribution
    
    def _analyze_diagnostic_types(self, sessions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze diagnostic type distribution."""
        distribution = {}
        for session in sessions:
            dtype = session["diagnostic_type"]
            distribution[dtype] = distribution.get(dtype, 0) + 1
        return distribution
    
    def _select_model_for_diagnostic(self, diagnostic_type: str) -> str:
        """Select appropriate model for diagnostic type."""
        model_map = {
            DiagnosticType.CARDIOLOGY.value: "CardioNet_v1",
            DiagnosticType.RADIOLOGY.value: "RadiologyAI_v2",
            DiagnosticType.PATHOLOGY.value: "PathologyNet_v1",
            DiagnosticType.GENERAL.value: "GeneralDx_v3"
        }
        return model_map.get(diagnostic_type, "GeneralDx_v3")
    
    def _get_patient_age(self, patient_id: str) -> int:
        """Get patient age by ID."""
        for patient in self.patients:
            if patient["patient_id"] == patient_id:
                return patient["demographics"]["age"]
        return 50  # Default age
    
    def _analyze_model_usage(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze model usage distribution."""
        usage = {}
        for prediction in predictions:
            model = prediction["model_name"]
            usage[model] = usage.get(model, 0) + 1
        return usage
    
    def _check_data_anonymization(self) -> Dict[str, Any]:
        """Check data anonymization compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.9, 1.0),
            "details": "Patient data properly anonymized"
        }
    
    def _check_access_controls(self) -> Dict[str, Any]:
        """Check access control compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.85, 0.98),
            "details": "Proper access controls in place"
        }
    
    def _check_audit_logging(self) -> Dict[str, Any]:
        """Check audit logging compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.88, 0.96),
            "details": "Comprehensive audit logging implemented"
        }
    
    def _check_patient_consent(self) -> Dict[str, Any]:
        """Check patient consent compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.82, 0.94),
            "details": "Patient consent properly documented"
        }
    
    def _check_model_validation(self) -> Dict[str, Any]:
        """Check model validation compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.80, 0.92),
            "details": "Model validation documentation complete"
        }
    
    def _check_clinical_evidence(self) -> Dict[str, Any]:
        """Check clinical evidence compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.78, 0.90),
            "details": "Clinical evidence supports model use"
        }
    
    def _check_risk_management(self) -> Dict[str, Any]:
        """Check risk management compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.85, 0.95),
            "details": "Risk management procedures in place"
        }
    
    def _check_performance_monitoring(self) -> Dict[str, Any]:
        """Check performance monitoring compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.83, 0.93),
            "details": "Performance monitoring system active"
        }
    
    def _check_data_minimization(self) -> Dict[str, Any]:
        """Check data minimization compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.86, 0.96),
            "details": "Data minimization principles followed"
        }
    
    def _check_explainability(self) -> Dict[str, Any]:
        """Check explainability compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.84, 0.94),
            "details": "Model explanations provided to patients"
        }
    
    def _check_data_portability(self) -> Dict[str, Any]:
        """Check data portability compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.80, 0.90),
            "details": "Patient data portability supported"
        }
    
    def _check_privacy_design(self) -> Dict[str, Any]:
        """Check privacy by design compliance."""
        return {
            "status": "compliant",
            "score": random.uniform(0.82, 0.92),
            "details": "Privacy by design principles implemented"
        }
    
    def _generate_drift_metrics(self, drift_type: str) -> Dict[str, float]:
        """Generate drift metrics based on drift type."""
        if drift_type == "data_drift":
            return {
                "population_stability_index": random.uniform(0.0, 0.25),
                "feature_drift_score": random.uniform(0.0, 0.2),
                "distribution_distance": random.uniform(0.0, 0.15)
            }
        elif drift_type == "concept_drift":
            return {
                "prediction_drift_score": random.uniform(0.0, 0.3),
                "target_correlation_change": random.uniform(0.0, 0.2),
                "model_performance_drop": random.uniform(0.0, 0.15)
            }
        else:  # performance_drift
            return {
                "accuracy_drop": random.uniform(0.0, 0.1),
                "precision_drop": random.uniform(0.0, 0.08),
                "recall_drop": random.uniform(0.0, 0.12)
            }
    
    def _get_drift_action(self, drift_type: str, drift_detected: bool) -> str:
        """Get recommended action for drift scenario."""
        if not drift_detected:
            return "Continue monitoring"
        
        actions = {
            "data_drift": "Retrain model with recent data",
            "concept_drift": "Update model architecture or features",
            "performance_drift": "Investigate and retrain model"
        }
        return actions.get(drift_type, "Investigate further")
    
    def _generate_simulation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of simulation results."""
        phases = results.get("phases", {})
        
        summary = {
            "total_patients": phases.get("patient_generation", {}).get("count", 0),
            "total_sessions": phases.get("diagnostic_workflows", {}).get("count", 0),
            "total_predictions": phases.get("model_predictions", {}).get("count", 0),
            "regulatory_scenarios": len(phases.get("regulatory_compliance", {}).get("scenarios", [])),
            "drift_scenarios": len(phases.get("model_drift", {}).get("drift_scenarios", [])),
            "overall_success": True
        }
        
        return summary


class Simulator:
    """
    Legacy Simulator class for backward compatibility.
    
    This class provides a simple interface to the HealthcareSimulator
    for backward compatibility with existing code.
    """
    
    def __init__(self):
        """Initialize the simulator."""
        self.healthcare_simulator = HealthcareSimulator()
    
    def run(self) -> Dict[str, Any]:
        """
        Run a basic simulation.
        
        Returns:
            Simulation results
        """
        print("Running healthcare AI simulation...")
        
        # Run a simplified simulation
        config = SimulationConfig(num_patients=10, time_range_days=7)
        simulator = HealthcareSimulator(config=config)
        
        results = simulator.run_comprehensive_simulation()
        
        print("Simulation completed successfully!")
        return results


class MockDiagnosticModel:
    """Mock diagnostic model for simulation purposes."""
    
    def __init__(self, model_name: str, diagnostic_type: DiagnosticType):
        self.model_name = model_name
        self.diagnostic_type = diagnostic_type
    
    def predict(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock prediction."""
        # Simulate prediction based on diagnostic type
        if self.diagnostic_type == DiagnosticType.CARDIOLOGY:
            conditions = ["normal", "arrhythmia", "ischemia", "heart_failure"]
        elif self.diagnostic_type == DiagnosticType.RADIOLOGY:
            conditions = ["normal", "abnormal_finding", "requires_followup", "urgent"]
        else:
            conditions = ["normal", "abnormal", "inconclusive"]
        
        prediction = random.choice(conditions)
        
        return {
            "diagnosis": prediction,
            "risk_score": random.uniform(0.1, 0.9),
            "recommended_action": self._get_recommendation(prediction)
        }
    
    def explain_prediction(self, input_features: Dict[str, Any], 
                         prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock explanation."""
        # Create feature importance based on input features
        feature_importance = {}
        total_importance = 0
        
        for feature, value in input_features.items():
            importance = random.uniform(-0.3, 0.3) * (1 + abs(value) * 0.01)
            feature_importance[feature] = importance
            total_importance += abs(importance)
        
        # Normalize
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        return {
            "method": "SHAP",
            "feature_importance": feature_importance,
            "explanation_confidence": random.uniform(0.7, 0.95)
        }
    
    def _get_recommendation(self, diagnosis: str) -> str:
        """Get recommendation based on diagnosis."""
        recommendations = {
            "normal": "No immediate action required",
            "abnormal": "Follow-up recommended",
            "urgent": "Immediate medical attention required",
            "arrhythmia": "Cardiology consultation recommended",
            "heart_failure": "Immediate cardiology referral"
        }
        return recommendations.get(diagnosis, "Consult with physician")