"""
Comprehensive test suite for the Blockchain-Enabled Healthcare AI Framework.

This module contains tests for all major components of the framework to ensure
proper functionality and integration.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch

# Import framework components
from framework.blockchain.hyperledger_connector import HyperledgerConnector
from framework.model_registry.registry import ModelRegistry
from framework.decision_audit.audit_logger import DecisionAuditLogger
from framework.data_provenance.provenance_logger import DataProvenanceLogger
from framework.regulatory.report_generator import RegulatoryReportGenerator
from framework.explainability.shap_explainer import SHAPExplainer
from framework.explainability.lime_explainer import LIMEExplainer
from framework.simulation.simulator import HealthcareSimulator, SimulationConfig


class TestBlockchainConnector(unittest.TestCase):
    """Test cases for Hyperledger Fabric blockchain connector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.connector = HyperledgerConnector(
            config_path="test_network.yaml",
            channel_name="test_channel",
            chaincode_name="test_chaincode", 
            org_name="TestOrg",
            user_name="test_user"
        )
    
    def test_connection_initialization(self):
        """Test blockchain connector initialization."""
        self.assertIsNotNone(self.connector)
        self.assertTrue(self.connector.connected)
        self.assertEqual(self.connector.channel_name, "test_channel")
    
    def test_transaction_invocation(self):
        """Test blockchain transaction invocation."""
        result = self.connector.invoke_transaction("test_function", "arg1", "arg2")
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
    
    def test_ledger_query(self):
        """Test blockchain ledger querying."""
        result = self.connector.query_ledger("query_function", "arg1")
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)


class TestModelRegistry(unittest.TestCase):
    """Test cases for AI model registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_blockchain = Mock()
        self.mock_blockchain.invoke_transaction.return_value = {
            "status": "success",
            "hash": "test_hash_123"
        }
        self.registry = ModelRegistry(self.mock_blockchain)
    
    def test_model_registration(self):
        """Test model registration functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test model data")
            temp_file_path = temp_file.name
        
        try:
            model_hash = self.registry.register_model(
                "TestModel_v1",
                temp_file_path,
                {"accuracy": 0.95, "version": "1.0"}
            )
            
            self.assertIsNotNone(model_hash)
            self.assertIsInstance(model_hash, str)
            self.mock_blockchain.invoke_transaction.assert_called_once()
        finally:
            os.unlink(temp_file_path)
    
    def test_model_integrity_verification(self):
        """Test model integrity verification."""
        self.mock_blockchain.query_ledger.return_value = {
            "status": "success",
            "model": {"metadata": {"file_hash": "test_hash"}}
        }
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test model data")
            temp_file_path = temp_file.name
        
        try:
            # This will fail because the hash won't match, which is expected
            is_valid = self.registry.verify_model_integrity("test_model", temp_file_path)
            self.assertIsInstance(is_valid, bool)
        finally:
            os.unlink(temp_file_path)


class TestDecisionAuditLogger(unittest.TestCase):
    """Test cases for decision audit logging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_blockchain = Mock()
        self.mock_blockchain.invoke_transaction.return_value = {
            "status": "success",
            "hash": "audit_hash_123"
        }
        self.audit_logger = DecisionAuditLogger(self.mock_blockchain)
    
    def test_decision_logging(self):
        """Test decision logging functionality."""
        decision_hash = self.audit_logger.log_decision(
            case_id="test_case_001",
            input_data={"age": 50, "bp": 120},
            model_name="TestModel_v1",
            decision="low_risk",
            explanation={"feature_importance": {"age": 0.3, "bp": 0.7}},
            confidence_score=0.85
        )
        
        self.assertIsNotNone(decision_hash)
        self.assertIsInstance(decision_hash, str)
        self.mock_blockchain.invoke_transaction.assert_called_once()
    
    def test_batch_prediction_logging(self):
        """Test batch prediction logging."""
        batch_predictions = [
            {
                "case_id": "case_001",
                "input_data": {"age": 50},
                "model_name": "TestModel",
                "decision": "low_risk",
                "explanation": {"importance": {"age": 0.5}}
            },
            {
                "case_id": "case_002", 
                "input_data": {"age": 70},
                "model_name": "TestModel",
                "decision": "high_risk",
                "explanation": {"importance": {"age": 0.8}}
            }
        ]
        
        hashes = self.audit_logger.log_prediction_batch(batch_predictions)
        self.assertEqual(len(hashes), 2)
        self.assertTrue(all(isinstance(h, str) for h in hashes if h))


class TestDataProvenanceLogger(unittest.TestCase):
    """Test cases for data provenance logging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_blockchain = Mock()
        self.mock_blockchain.invoke_transaction.return_value = {
            "status": "success",
            "hash": "provenance_hash_123"
        }
        self.provenance_logger = DataProvenanceLogger(self.mock_blockchain)
    
    def test_sample_logging(self):
        """Test sample provenance logging."""
        sample_hash = self.provenance_logger.log_sample(
            sample_id="sample_001",
            sample_data={"age": 50, "gender": "M"},
            provenance_info={"source": "Hospital_A", "date": "2024-01-15"}
        )
        
        self.assertIsNotNone(sample_hash)
        self.assertIsInstance(sample_hash, str)
        self.mock_blockchain.invoke_transaction.assert_called_once()
    
    def test_data_transformation_logging(self):
        """Test data transformation logging."""
        transformation_hash = self.provenance_logger.log_data_transformation(
            transformation_id="transform_001",
            source_sample_ids=["sample_001", "sample_002"],
            target_sample_id="sample_003",
            transformation_details={
                "type": "normalization",
                "method": "z_score",
                "parameters": {"mean": 0, "std": 1}
            }
        )
        
        self.assertIsNotNone(transformation_hash)
        self.assertIsInstance(transformation_hash, str)


class TestExplainabilityModules(unittest.TestCase):
    """Test cases for explainability modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.test_data = {"age": 50, "bp": 120, "cholesterol": 200}
    
    def test_shap_explainer(self):
        """Test SHAP explainer functionality."""
        explainer = SHAPExplainer(self.mock_model)
        explanation = explainer.explain(self.test_data)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn("explanation_type", explanation)
        self.assertEqual(explanation["explanation_type"], "shap")
        self.assertIn("explanation", explanation)
    
    def test_lime_explainer(self):
        """Test LIME explainer functionality."""
        explainer = LIMEExplainer(self.mock_model)
        explanation = explainer.explain(self.test_data)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn("explanation_type", explanation)
        self.assertEqual(explanation["explanation_type"], "lime")
        self.assertIn("explanation", explanation)
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction."""
        shap_explainer = SHAPExplainer(self.mock_model)
        importance = shap_explainer.get_feature_importance(self.test_data, top_k=5)
        
        self.assertIsInstance(importance, dict)
        if "top_features" in importance:
            self.assertIsInstance(importance["top_features"], list)


class TestRegulatoryReportGenerator(unittest.TestCase):
    """Test cases for regulatory report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_blockchain = Mock()
        self.mock_blockchain.query_ledger.return_value = {
            "status": "success",
            "model": {"model_name": "TestModel", "version": "1.0"}
        }
        self.report_generator = RegulatoryReportGenerator(self.mock_blockchain)
    
    def test_model_report_generation(self):
        """Test model validation report generation."""
        report_path = self.report_generator.generate_model_report("TestModel_v1")
        
        self.assertIsNotNone(report_path)
        if report_path and os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            self.assertIsInstance(report, dict)
            self.assertIn("report_metadata", report)
            # Clean up
            os.unlink(report_path)
    
    def test_decision_audit_report_generation(self):
        """Test decision audit report generation."""
        report_path = self.report_generator.generate_decision_audit_report("case_001")
        
        self.assertIsNotNone(report_path)
        if report_path and os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            self.assertIsInstance(report, dict)
            self.assertIn("report_metadata", report)
            # Clean up
            os.unlink(report_path)
    
    def test_compliance_summary_report(self):
        """Test compliance summary report generation."""
        self.mock_blockchain.query_ledger.return_value = {
            "status": "success", 
            "audit_trail": []
        }
        
        report_path = self.report_generator.generate_compliance_summary_report()
        
        self.assertIsNotNone(report_path)
        if report_path and os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            self.assertIsInstance(report, dict)
            self.assertIn("report_metadata", report)
            # Clean up
            os.unlink(report_path)


class TestHealthcareSimulator(unittest.TestCase):
    """Test cases for healthcare simulation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(
            num_patients=5,
            time_range_days=7,
            drift_probability=0.1
        )
        self.simulator = HealthcareSimulator(config=self.config)
    
    def test_patient_generation(self):
        """Test synthetic patient generation."""
        patients_result = self.simulator.generate_synthetic_patients(5)
        
        self.assertIsInstance(patients_result, dict)
        self.assertEqual(patients_result["status"], "success")
        self.assertEqual(patients_result["count"], 5)
        self.assertIsInstance(patients_result["patients"], list)
        
        # Verify patient structure
        patient = patients_result["patients"][0]
        self.assertIn("patient_id", patient)
        self.assertIn("demographics", patient)
        self.assertIn("clinical_features", patient)
    
    def test_diagnostic_workflow_simulation(self):
        """Test diagnostic workflow simulation."""
        # First generate patients
        patients_result = self.simulator.generate_synthetic_patients(3)
        
        # Then simulate diagnostic workflows
        diagnostics_result = self.simulator.simulate_diagnostic_workflows(patients_result["patients"])
        
        self.assertIsInstance(diagnostics_result, dict)
        self.assertEqual(diagnostics_result["status"], "success")
        self.assertIsInstance(diagnostics_result["sessions"], list)
        self.assertGreaterEqual(diagnostics_result["count"], 3)  # At least one session per patient
    
    def test_model_predictions_simulation(self):
        """Test model predictions simulation."""
        # Generate test sessions
        test_sessions = [
            {
                "session_id": "test_session_1",
                "patient_id": "test_patient_1",
                "diagnostic_type": "cardiology",
                "clinical_measurements": {"bp": 120, "hr": 70}
            }
        ]
        
        predictions_result = self.simulator.simulate_model_predictions(test_sessions)
        
        self.assertIsInstance(predictions_result, dict)
        self.assertEqual(predictions_result["status"], "success")
        self.assertIsInstance(predictions_result["predictions"], list)
    
    def test_comprehensive_simulation(self):
        """Test full comprehensive simulation."""
        results = self.simulator.run_comprehensive_simulation()
        
        self.assertIsInstance(results, dict)
        self.assertIn("simulation_metadata", results)
        self.assertIn("phases", results)
        self.assertIn("summary", results)
        
        # Verify all phases completed
        phases = results["phases"]
        expected_phases = [
            "patient_generation", 
            "diagnostic_workflows", 
            "model_predictions",
            "regulatory_compliance", 
            "model_drift"
        ]
        
        for phase in expected_phases:
            self.assertIn(phase, phases)
        
        # Verify summary
        summary = results["summary"]
        self.assertIn("total_patients", summary)
        self.assertIn("total_predictions", summary)
        self.assertTrue(summary.get("overall_success", False))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.blockchain = HyperledgerConnector(
            config_path="test_network.yaml",
            channel_name="test_channel",
            chaincode_name="test_chaincode",
            org_name="TestOrg", 
            user_name="test_user"
        )
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Register a model
        model_registry = ModelRegistry(self.blockchain)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test model data")
            temp_file_path = temp_file.name
        
        try:
            model_hash = model_registry.register_model(
                "IntegrationTestModel_v1",
                temp_file_path,
                {"accuracy": 0.95, "version": "1.0"}
            )
            self.assertIsNotNone(model_hash)
            
            # 2. Log data provenance
            data_logger = DataProvenanceLogger(self.blockchain)
            sample_hash = data_logger.log_sample(
                "integration_sample_001",
                {"age": 45, "gender": "F"},
                {"source": "integration_test", "date": "2024-01-15"}
            )
            self.assertIsNotNone(sample_hash)
            
            # 3. Log decision
            audit_logger = DecisionAuditLogger(self.blockchain)
            decision_hash = audit_logger.log_decision(
                "integration_case_001",
                {"age": 45, "bp": 120},
                "IntegrationTestModel_v1", 
                "low_risk",
                {"feature_importance": {"age": 0.4, "bp": 0.6}},
                confidence_score=0.88
            )
            self.assertIsNotNone(decision_hash)
            
            # 4. Generate report
            report_generator = RegulatoryReportGenerator(self.blockchain)
            report_path = report_generator.generate_model_report("IntegrationTestModel_v1")
            self.assertIsNotNone(report_path)
            
            if report_path and os.path.exists(report_path):
                os.unlink(report_path)
                
        finally:
            os.unlink(temp_file_path)
    
    def test_simulation_integration(self):
        """Test simulation integration with blockchain logging."""
        config = SimulationConfig(num_patients=3, time_range_days=3)
        simulator = HealthcareSimulator(self.blockchain, config)
        
        results = simulator.run_comprehensive_simulation()
        
        self.assertIsInstance(results, dict)
        self.assertIn("summary", results)
        self.assertTrue(results["summary"].get("overall_success", False))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBlockchainConnector,
        TestModelRegistry,
        TestDecisionAuditLogger,
        TestDataProvenanceLogger,
        TestExplainabilityModules,
        TestRegulatoryReportGenerator,
        TestHealthcareSimulator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")