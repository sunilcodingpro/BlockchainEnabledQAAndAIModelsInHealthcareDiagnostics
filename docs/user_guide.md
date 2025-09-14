# User Guide

This guide provides comprehensive instructions for using the Blockchain-Enabled QA and AI Models in Healthcare Diagnostics framework.

## Table of Contents
1. [Getting Started](#getting-started)
2. [API Reference](#api-reference)
3. [Model Registration](#model-registration)
4. [Diagnostic Submission](#diagnostic-submission)
5. [Audit and Compliance](#audit-and-compliance)
6. [Simulation](#simulation)
7. [Examples](#examples)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git for version control
- Basic understanding of REST APIs

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sunilcodingpro/BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics.git
   cd BlockchainEnabledQAAndAIModelsInHealthcareDiagnostics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment:
   ```bash
   export PYTHONPATH=.
   ```

### Quick Start
Run the demonstration:
```bash
python test/demo.py
```

Start the API server:
```bash
python framework/backend/api.py
```

## API Reference

### Base URL
```
http://localhost:5000/api
```

### Authentication
Currently uses basic authentication. In production, implement proper JWT or OAuth2.

### Endpoints Overview

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/health` | GET | Health check |
| `/api/models/register` | POST | Register AI model |
| `/api/diagnostics/submit` | POST | Submit diagnostic case |
| `/api/audit/trail` | GET | Get audit trail |
| `/api/compliance/report` | POST | Generate compliance report |
| `/api/simulation/run` | POST | Run simulation |
| `/api/models/{id}` | GET | Get model information |
| `/api/models/{id}/verify` | POST | Verify model integrity |

## Model Registration

### Registering a New Model
```bash
curl -X POST http://localhost:5000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "CardioNet_v2",
    "model_path": "models/cardionet_v2.bin",
    "metadata": {
      "accuracy": 0.96,
      "version": "2.0",
      "description": "Enhanced cardiovascular risk prediction model",
      "training_data_size": 50000,
      "validation_accuracy": 0.94
    }
  }'
```

### Response
```json
{
  "status": "success",
  "model_name": "CardioNet_v2",
  "model_hash": "abc123...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Model Verification
```bash
curl -X POST http://localhost:5000/api/models/CardioNet_v2/verify \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/cardionet_v2.bin"
  }'
```

## Diagnostic Submission

### Submitting a Diagnostic Case
```bash
curl -X POST http://localhost:5000/api/diagnostics/submit \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case_12345",
    "patient_data": {
      "age": 65,
      "gender": "M",
      "symptoms": ["chest_pain", "shortness_of_breath"]
    },
    "clinical_data": {
      "blood_pressure_systolic": 140,
      "blood_pressure_diastolic": 90,
      "heart_rate": 85,
      "cholesterol": 220,
      "bmi": 28.5
    },
    "model_name": "CardioNet_v2",
    "explainability_method": "shap"
  }'
```

### Response
```json
{
  "status": "success",
  "case_id": "case_12345",
  "prediction": {
    "diagnosis": "high_risk",
    "risk_score": 0.75,
    "confidence": 0.89,
    "recommendations": [
      "Immediate lifestyle changes required",
      "Medication may be necessary",
      "Cardiology consultation recommended"
    ]
  },
  "explanation": {
    "explanation_type": "shap",
    "explanation": {
      "feature_importance": {
        "age": 0.25,
        "blood_pressure_systolic": 0.35,
        "cholesterol": 0.20,
        "bmi": 0.15,
        "symptoms": 0.05
      }
    }
  },
  "decision_hash": "def456...",
  "provenance_hash": "ghi789...",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

## Audit and Compliance

### Getting Audit Trail
```bash
# Get all audit entries
curl http://localhost:5000/api/audit/trail

# Filter by model
curl http://localhost:5000/api/audit/trail?model_id=CardioNet_v2

# Filter by case
curl http://localhost:5000/api/audit/trail?case_id=case_12345

# Limit results
curl http://localhost:5000/api/audit/trail?limit=50
```

### Generating Compliance Reports

#### Model Validation Report
```bash
curl -X POST http://localhost:5000/api/compliance/report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "model_report",
    "model_name": "CardioNet_v2"
  }'
```

#### Decision Audit Report
```bash
curl -X POST http://localhost:5000/api/compliance/report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "decision_audit",
    "case_id": "case_12345"
  }'
```

#### Compliance Summary Report
```bash
curl -X POST http://localhost:5000/api/compliance/report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "compliance_summary",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }'
```

## Simulation

### Running Healthcare AI Simulation
```bash
curl -X POST http://localhost:5000/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "num_patients": 100,
    "diagnostic_types": ["cardiology", "radiology"],
    "time_range_days": 30,
    "drift_probability": 0.1
  }'
```

### Simulation Response
The simulation will return comprehensive results including:
- Generated patient demographics
- Diagnostic workflow metrics
- Model prediction statistics
- Regulatory compliance scenarios
- Model drift detection results

## Examples

### Complete Workflow Example

1. **Register a model:**
```python
import requests

response = requests.post('http://localhost:5000/api/models/register', json={
    "model_name": "DiabetesPredictor_v1",
    "model_path": "models/diabetes_v1.bin",
    "metadata": {
        "accuracy": 0.92,
        "version": "1.0",
        "description": "Type 2 diabetes risk prediction"
    }
})
print(response.json())
```

2. **Submit diagnostic cases:**
```python
cases = [
    {
        "case_id": "case_001",
        "patient_data": {"age": 45, "gender": "F"},
        "clinical_data": {"blood_sugar": 95, "bmi": 24.5},
        "model_name": "DiabetesPredictor_v1"
    },
    {
        "case_id": "case_002", 
        "patient_data": {"age": 60, "gender": "M"},
        "clinical_data": {"blood_sugar": 125, "bmi": 30.2},
        "model_name": "DiabetesPredictor_v1"
    }
]

for case in cases:
    response = requests.post('http://localhost:5000/api/diagnostics/submit', json=case)
    print(f"Case {case['case_id']}: {response.json()['prediction']['diagnosis']}")
```

3. **Generate compliance report:**
```python
response = requests.post('http://localhost:5000/api/compliance/report', json={
    "report_type": "model_report",
    "model_name": "DiabetesPredictor_v1"
})
print(response.json()['report']['regulatory_compliance'])
```

### Python SDK Usage

```python
from framework.blockchain.hyperledger_connector import HyperledgerConnector
from framework.model_registry.registry import ModelRegistry
from framework.decision_audit.audit_logger import DecisionAuditLogger

# Initialize components
blockchain = HyperledgerConnector(
    config_path="network.yaml",
    channel_name="qahealthchannel",
    chaincode_name="aiqa_cc",
    org_name="HealthcareOrg",
    user_name="user1"
)

model_registry = ModelRegistry(blockchain)
audit_logger = DecisionAuditLogger(blockchain)

# Register model
model_hash = model_registry.register_model(
    "MyModel_v1",
    "path/to/model.bin",
    {"accuracy": 0.95, "version": "1.0"}
)

# Log decision
decision_hash = audit_logger.log_decision(
    "case_123",
    {"age": 50, "bp": 130},
    "MyModel_v1",
    "low_risk",
    {"feature_importance": {"age": 0.3, "bp": 0.7}},
    confidence_score=0.85
)
```

## Best Practices

### Model Registration
- Include comprehensive metadata
- Use semantic versioning
- Validate model integrity regularly
- Document model purpose and limitations

### Data Privacy
- Always anonymize patient data
- Implement proper access controls
- Log all data access and usage
- Follow HIPAA guidelines

### Explainability
- Always provide explanations for high-risk predictions
- Use appropriate explanation methods (SHAP for global, LIME for local)
- Validate explanation quality
- Document explanation methodology

### Compliance
- Generate regular compliance reports
- Maintain complete audit trails
- Monitor regulatory requirement changes
- Implement automated compliance checks

## Troubleshooting

### Common Issues

1. **Model registration fails:**
   - Check file path exists
   - Verify metadata format
   - Ensure blockchain connection

2. **Diagnostic submission errors:**
   - Validate input data format
   - Check model exists and is active
   - Verify required fields present

3. **Compliance report generation fails:**
   - Check date range validity
   - Ensure sufficient audit data exists
   - Verify report type parameter

4. **API connection issues:**
   - Verify server is running
   - Check firewall settings
   - Validate API endpoint URLs

### Support
For additional support:
- Check logs in `logs/` directory
- Review error messages in API responses
- Consult the architecture documentation
- Run simulation tests to verify system health

## Security Considerations

### Production Deployment
- Implement proper authentication and authorization
- Use HTTPS for all API communications
- Encrypt sensitive data at rest and in transit
- Regular security audits and penetration testing
- Implement rate limiting and DDoS protection

### Data Protection
- Follow HIPAA and GDPR guidelines
- Implement data minimization principles
- Regular data purging according to retention policies
- Anonymization and pseudonymization techniques