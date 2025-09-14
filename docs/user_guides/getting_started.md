# Getting Started Guide

## Blockchain-Enabled Healthcare AI QA Framework

### Overview

This guide will help you get started with the blockchain-enabled quality assurance framework for AI models in healthcare diagnostics. The system provides comprehensive audit trails, regulatory compliance, and explainable AI capabilities through distributed ledger technology.

---

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 50GB available disk space
- **Network**: Stable internet connection for blockchain synchronization

### Required Knowledge
- Basic understanding of REST APIs
- Familiarity with healthcare AI/ML workflows
- Understanding of regulatory compliance requirements (FDA, HIPAA, etc.)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sunilcodingpro/BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics.git
cd BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration Setup
Create a `.env` file in the root directory:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your_secret_key_here

# Blockchain Configuration  
BLOCKCHAIN_NETWORK=development
BLOCKCHAIN_CHANNEL=qahealthchannel
CHAINCODE_NAME=aiqa_cc
ORGANIZATION_NAME=HealthcareOrg
USER_NAME=api_user

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthcare_qa
DB_USER=qa_user
DB_PASSWORD=secure_password

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_here
JWT_EXPIRATION_HOURS=24
ENCRYPTION_KEY=your_encryption_key_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/api.log
```

### 4. Initialize Database
```bash
# Create database schema
python scripts/init_database.py

# Run migrations
python scripts/migrate.py
```

---

## Quick Start

### 1. Start the API Server
```bash
# Development mode with auto-reload
python framework/backend/api_server.py

# Production mode with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker framework.backend.api_server:app
```

The API server will start at `http://localhost:8000`

### 2. Access API Documentation
Open your browser and navigate to:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 3. Health Check
Verify the system is running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "success": true,
  "message": "Health check completed",
  "data": {
    "status": "healthy",
    "blockchain_status": "connected",
    "components": {
      "blockchain_connector": true,
      "model_registry": true,
      "audit_logger": true,
      "report_generator": true,
      "simulator": true
    }
  }
}
```

---

## Basic Workflow

### Step 1: Register an AI Model

Register your AI/ML model in the blockchain registry:

```python
import requests

# API endpoint
url = "http://localhost:8000/api/v1/register_model"

# Model registration data
model_data = {
    "model_name": "DiabetesPredictor_v1.0",
    "model_path": "/models/diabetes_predictor.pkl",
    "metadata": {
        "algorithm": "Random Forest",
        "accuracy": 0.92,
        "version": "1.0",
        "training_date": "2024-01-15",
        "validation_metrics": {
            "precision": 0.91,
            "recall": 0.89,
            "f1_score": 0.90
        },
        "description": "Diabetes risk prediction model"
    }
}

# Submit registration
response = requests.post(url, json=model_data)
result = response.json()

print(f"Model registered: {result['data']['model_hash']}")
```

### Step 2: Submit a Diagnostic

Submit an AI diagnostic operation with audit trail:

```python
# Diagnostic submission
diagnostic_url = "http://localhost:8000/api/v1/submit_diagnostic"

diagnostic_data = {
    "case_id": "patient_001_2024",
    "model_id": "DiabetesPredictor_v1.0",
    "input_features": {
        "age": 45,
        "bmi": 28.5,
        "glucose": 140,
        "blood_pressure": 80,
        "family_history": True
    },
    "prediction": {
        "risk_level": "moderate",
        "probability": 0.67,
        "recommendation": "Lifestyle modification recommended"
    },
    "confidence_score": 0.85,
    "explanation": {
        "type": "shap",
        "feature_importance": {
            "glucose": 0.35,
            "bmi": 0.25,
            "age": 0.20,
            "family_history": 0.15,
            "blood_pressure": 0.05
        }
    }
}

response = requests.post(diagnostic_url, json=diagnostic_data)
result = response.json()

print(f"Diagnostic ID: {result['data']['decision_id']}")
```

### Step 3: Retrieve Audit Trail

Get the complete audit trail for compliance:

```python
# Audit trail request
audit_url = "http://localhost:8000/api/v1/get_audit_trail"

audit_request = {
    "model_id": "DiabetesPredictor_v1.0",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z",
    "include_compliance_events": True
}

response = requests.post(audit_url, json=audit_request)
audit_trail = response.json()

print(f"Total diagnostics: {audit_trail['data']['summary']['total_diagnostics']}")
```

### Step 4: Generate Compliance Report

Create a regulatory compliance report:

```python
# Compliance report request
report_url = "http://localhost:8000/api/v1/compliance_report"

report_request = {
    "model_id": "DiabetesPredictor_v1.0",
    "regulation": "fda_21cfr820",
    "period_days": 30,
    "format_type": "json"
}

response = requests.post(report_url, json=report_request)
report_result = response.json()

print(f"Compliance score: {report_result['data']['report_summary']['compliance_score']}")
```

---

## Advanced Features

### Simulation Testing

Run realistic simulation scenarios to test your models:

```python
# Simulation request
sim_url = "http://localhost:8000/api/v1/simulate_case"

simulation_config = {
    "simulation_type": "patient_case",
    "scenario_parameters": {
        "medical_condition": "diabetes",
        "case_complexity": "medium",
        "data_quality": "high"
    },
    "model_id": "DiabetesPredictor_v1.0",
    "case_count": 100
}

response = requests.post(sim_url, json=simulation_config)
sim_results = response.json()

print(f"Simulated {sim_results['data']['cases_simulated']} cases")
print(f"Quality score: {sim_results['data']['results_summary']['data_quality_score']}")
```

### Model Performance Monitoring

Check model performance and drift detection:

```python
# Get model performance metrics
from framework.model_registry.registry import ModelRegistry
from framework.blockchain.hyperledger_connector import HyperledgerConnector

# Initialize components
blockchain = HyperledgerConnector(
    config_path="network.yaml",
    channel_name="qahealthchannel",
    chaincode_name="aiqa_cc",
    org_name="HealthcareOrg", 
    user_name="analyst",
    mock_mode=True
)

registry = ModelRegistry(blockchain)

# Get performance metrics
performance = await registry.get_model_performance_metrics(
    "DiabetesPredictor_v1.0", 
    days=30
)

print(f"Average confidence: {performance['average_confidence']}")
print(f"Drift status: {performance['drift_status']}")
```

---

## Integration Examples

### Hospital Information System Integration

```python
# Example HIS integration
class HISIntegration:
    def __init__(self, api_base_url):
        self.api_url = api_base_url
    
    async def process_patient_data(self, patient_data):
        # Anonymize patient data for AI processing
        anonymized_data = self.anonymize_patient_data(patient_data)
        
        # Submit to AI QA framework
        diagnostic_request = {
            "case_id": f"his_{patient_data['patient_id']}_anon",
            "model_id": "DiabetesPredictor_v1.0", 
            "input_features": anonymized_data,
            "prediction": await self.run_ai_model(anonymized_data),
            "confidence_score": 0.87
        }
        
        # Submit diagnostic with audit trail
        response = requests.post(
            f"{self.api_url}/submit_diagnostic",
            json=diagnostic_request
        )
        
        return response.json()
    
    def anonymize_patient_data(self, data):
        # Remove PII and create anonymized feature set
        return {
            "age": data["age"],
            "bmi": self.calculate_bmi(data["height"], data["weight"]),
            "glucose": data["lab_results"]["glucose"],
            "blood_pressure": data["vital_signs"]["bp_systolic"],
            "family_history": data["family_history"]["diabetes"]
        }
```

### EHR System Integration

```python
# FHIR-based EHR integration
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation

class EHRIntegration:
    def __init__(self, fhir_server_url, api_base_url):
        self.fhir_url = fhir_server_url
        self.api_url = api_base_url
    
    async def process_fhir_patient(self, patient_id):
        # Fetch patient data from FHIR server
        patient = await self.get_fhir_patient(patient_id)
        observations = await self.get_patient_observations(patient_id)
        
        # Convert FHIR data to AI model input
        model_input = self.fhir_to_model_input(patient, observations)
        
        # Submit diagnostic through QA framework
        diagnostic_data = {
            "case_id": f"ehr_{patient_id}",
            "model_id": "CardioRiskAssessment_v2.0",
            "input_features": model_input,
            "prediction": await self.run_cardio_model(model_input),
            "confidence_score": 0.91
        }
        
        # Record in blockchain audit trail
        response = requests.post(
            f"{self.api_url}/submit_diagnostic",
            json=diagnostic_data
        )
        
        return response.json()
```

---

## Troubleshooting

### Common Issues

#### 1. Blockchain Connection Failed
```
Error: Blockchain connector not initialized
```

**Solution:**
- Check blockchain network configuration
- Verify network connectivity
- Ensure proper certificates are installed
- Use mock mode for development: `mock_mode=True`

#### 2. Model Registration Failed
```
Error: Model registration failed: Missing required metadata field: accuracy
```

**Solution:**
- Ensure all required metadata fields are provided:
  - `accuracy`: Model accuracy (0.0 to 1.0)
  - `algorithm`: Algorithm description
  - `version`: Model version string

#### 3. API Authentication Error
```
Error: 401 Unauthorized
```

**Solution:**
- Obtain valid JWT token from authentication service
- Include token in Authorization header: `Bearer <token>`
- Check token expiration and refresh if needed

#### 4. Performance Issues
```
Error: Request timeout after 30 seconds
```

**Solution:**
- Check system resources (CPU, memory, disk)
- Optimize database queries and indexes
- Scale horizontally with load balancer
- Implement caching for frequent queries

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Set debug logging level
logging.basicConfig(level=logging.DEBUG)

# Enable API debug mode
import os
os.environ['DEBUG'] = 'true'

# Run API server in debug mode
python framework/backend/api_server.py --debug
```

### Log Files

Check log files for detailed error information:
- **API Logs**: `logs/api.log`
- **Blockchain Logs**: `logs/blockchain.log`  
- **Audit Logs**: `logs/audit.log`
- **System Logs**: `logs/system.log`

---

## Support and Documentation

### Additional Resources
- **System Architecture**: `docs/architecture/system_overview.md`
- **API Reference**: `docs/api/endpoints.md`
- **UML Diagrams**: `docs/uml/`
- **User Roles**: `docs/uml/user_roles.puml`

### Getting Help
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Join our healthcare AI QA community forums

### Contributing
- **Development**: Follow contribution guidelines in `CONTRIBUTING.md`
- **Testing**: Run test suite with `pytest`
- **Code Quality**: Use `black` for formatting, `flake8` for linting

---

## Next Steps

1. **Production Deployment**: Configure production environment with real blockchain network
2. **Security Hardening**: Implement production-grade security measures
3. **Integration**: Connect with your existing healthcare systems
4. **Training**: Train your team on the QA framework workflows
5. **Compliance**: Work with regulatory experts to ensure full compliance