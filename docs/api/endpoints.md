# API Endpoints Documentation

## Blockchain-Enabled Healthcare AI QA Framework - REST API

### Base URL
```
https://api.healthcare-ai-qa.example.com/api/v1
```

### Authentication
All API endpoints require authentication using Bearer tokens:
```
Authorization: Bearer <jwt_token>
```

---

## Core Endpoints

### 1. Model Registration

#### POST `/register_model`
Register a new AI/ML model in the blockchain registry.

**Request Body:**
```json
{
  "model_name": "CardioNet_v2.1",
  "model_path": "/models/cardionet_v2.1.bin", 
  "metadata": {
    "algorithm": "Deep Neural Network",
    "accuracy": 0.954,
    "version": "2.1",
    "training_date": "2024-01-15",
    "validation_metrics": {
      "precision": 0.95,
      "recall": 0.92, 
      "f1_score": 0.935,
      "auc_roc": 0.97
    },
    "description": "Cardiovascular risk assessment model"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model registered successfully",
  "data": {
    "model_name": "CardioNet_v2.1",
    "model_hash": "a1b2c3d4e5f6...",
    "registration_status": "success",
    "regulatory_status": "pending",
    "next_steps": [
      "Model registered successfully",
      "Compliance check scheduled", 
      "Awaiting regulatory approval"
    ]
  },
  "timestamp": "2024-01-20T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Model registered successfully
- `400 Bad Request`: Invalid model data or metadata
- `401 Unauthorized`: Invalid or missing authentication
- `409 Conflict`: Model name already exists

---

### 2. Diagnostic Submission

#### POST `/submit_diagnostic`
Submit an AI diagnostic operation with complete audit trail.

**Request Body:**
```json
{
  "case_id": "case_2024_001234",
  "model_id": "CardioNet_v2.1",
  "input_features": {
    "age": 65,
    "gender": "M",
    "systolic_bp": 140,
    "diastolic_bp": 90,
    "cholesterol": 220,
    "smoking": false,
    "family_history": true
  },
  "prediction": {
    "risk_category": "moderate",
    "risk_score": 0.67,
    "recommendation": "Lifestyle modification and follow-up in 6 months"
  },
  "confidence_score": 0.89,
  "explanation": {
    "type": "shap",
    "feature_importance": {
      "age": 0.25,
      "cholesterol": 0.30,
      "systolic_bp": 0.20,
      "family_history": 0.15,
      "gender": 0.10
    },
    "explanation_text": "High cholesterol and age are primary risk factors"
  },
  "human_review": {
    "reviewed_by": "dr_smith_12345", 
    "review_status": "approved",
    "notes": "Agrees with AI assessment"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Diagnostic submitted successfully",
  "data": {
    "decision_id": "decision_case_2024_001234_8f7e6d5c",
    "case_id": "case_2024_001234",
    "model_id": "CardioNet_v2.1", 
    "confidence_score": 0.89,
    "audit_trail_created": true,
    "compliance_status": "monitored"
  },
  "timestamp": "2024-01-20T14:45:30Z"
}
```

---

### 3. Audit Trail Retrieval

#### POST `/get_audit_trail`
Retrieve comprehensive audit trail from blockchain.

**Request Body:**
```json
{
  "model_id": "CardioNet_v2.1",
  "case_id": "",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z", 
  "include_compliance_events": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Audit trail retrieved successfully",
  "data": {
    "audit_trail": {
      "model_id": "CardioNet_v2.1",
      "model_info": {
        "name": "CardioNet_v2.1",
        "version": "2.1",
        "regulatory_status": "approved"
      },
      "diagnostics": [
        {
          "diagnostic_id": "decision_case_2024_001234_8f7e6d5c",
          "case_id": "case_2024_001234", 
          "timestamp": "2024-01-20T14:45:30Z",
          "confidence_score": 0.89,
          "prediction": "moderate_risk",
          "reviewed_by": "dr_smith_12345"
        }
      ],
      "compliance_events": [
        {
          "event_id": "drift_CardioNet_v2.1_1642684530",
          "event_type": "low_confidence",
          "severity": "medium",
          "description": "Low confidence prediction: 0.65"
        }
      ]
    },
    "summary": {
      "total_diagnostics": 1,
      "total_compliance_events": 1,
      "period": {
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z"
      }
    }
  },
  "timestamp": "2024-01-20T15:00:00Z"
}
```

---

### 4. Compliance Reporting

#### POST `/compliance_report`
Generate comprehensive regulatory compliance report.

**Request Body:**
```json
{
  "model_id": "CardioNet_v2.1",
  "organization": "HealthcareOrg", 
  "regulation": "fda_21cfr820",
  "period_days": 90,
  "format_type": "json"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Compliance report generated successfully",
  "data": {
    "report_path": "reports/compliance_report_fda_21cfr820_HealthcareOrg_20240120_150500.json",
    "report_type": "fda_21cfr820",
    "organization": "HealthcareOrg",
    "period_days": 90,
    "report_summary": {
      "compliance_score": 94.5,
      "total_events": 12,
      "resolved_events": 11,
      "unresolved_events": 1,
      "critical_events": 0
    },
    "generated_at": "2024-01-20T15:05:00Z",
    "download_available": true
  },
  "timestamp": "2024-01-20T15:05:00Z"
}
```

---

### 5. Simulation Scenarios

#### POST `/simulate_case`
Run simulation scenarios for testing and validation.

**Request Body:**
```json
{
  "simulation_type": "patient_case",
  "scenario_parameters": {
    "medical_condition": "cardiovascular",
    "case_complexity": "medium",
    "data_quality": "high"
  },
  "model_id": "CardioNet_v2.1",
  "case_count": 50
}
```

**Response:**
```json
{
  "success": true,
  "message": "Simulation completed successfully", 
  "data": {
    "simulation_id": "sim_patient_case_a1b2c3d4",
    "simulation_type": "patient_case",
    "cases_simulated": 50,
    "results_summary": {
      "medical_condition": "cardiovascular",
      "case_distribution": {
        "age_range": [35, 80],
        "gender_balance": {
          "male": 0.52,
          "female": 0.48
        }
      },
      "data_quality_score": 0.94
    },
    "metrics": {
      "generation_rate": 125.5,
      "average_case_complexity": 0.72,
      "data_completeness": 0.97
    },
    "recommendations": [
      "Generated cases follow realistic medical patterns",
      "All cases include proper anonymization", 
      "Data suitable for AI model training and testing"
    ]
  },
  "timestamp": "2024-01-20T15:10:00Z"
}
```

---

## Utility Endpoints

### 6. Health Check

#### GET `/health`
Check system health and component status.

**Response:**
```json
{
  "success": true,
  "message": "Health check completed",
  "data": {
    "status": "healthy",
    "blockchain_status": "connected",
    "timestamp": "2024-01-20T15:15:00Z",
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

### 7. List Models

#### GET `/models?status=approved&organization=HealthcareOrg`
List registered models with optional filtering.

**Query Parameters:**
- `status`: Filter by regulatory status (pending, approved, rejected, deprecated)
- `organization`: Filter by organization name

**Response:**
```json
{
  "success": true,
  "message": "Models retrieved successfully",
  "data": {
    "models": [
      {
        "model_id": "CardioNet_v2.1",
        "name": "CardioNet", 
        "version": "2.1",
        "regulatory_status": "approved",
        "accuracy": 0.954,
        "created_at": "2024-01-15T10:00:00Z"
      }
    ],
    "total_count": 1,
    "filters_applied": {
      "status": "approved",
      "organization": "HealthcareOrg" 
    }
  },
  "timestamp": "2024-01-20T15:20:00Z"
}
```

### 8. Download Report

#### GET `/download_report/{report_filename}`
Download generated compliance or audit report file.

**Parameters:**
- `report_filename`: Name of the report file to download

**Response:**
- **Success**: File download with appropriate MIME type
- **Error 404**: Report file not found

---

## Error Handling

### Standard Error Response Format
```json
{
  "success": false,
  "message": "API request failed",
  "error": "Detailed error description",
  "timestamp": "2024-01-20T15:25:00Z"
}
```

### Common HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required or invalid
- `403 Forbidden`: Insufficient permissions  
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

## Rate Limiting

### Default Limits
- **Public endpoints**: 100 requests per minute per IP
- **Authenticated endpoints**: 1000 requests per minute per user
- **Report generation**: 10 requests per hour per user
- **Simulation endpoints**: 5 requests per hour per user

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1642684800
```

---

## SDK and Examples

### Python SDK Example
```python
import requests
from datetime import datetime

# Initialize client
base_url = "https://api.healthcare-ai-qa.example.com/api/v1"
headers = {"Authorization": "Bearer your_jwt_token"}

# Register model
model_data = {
    "model_name": "CardioNet_v2.1",
    "model_path": "/models/cardionet_v2.1.bin",
    "metadata": {
        "algorithm": "Deep Neural Network", 
        "accuracy": 0.954,
        "version": "2.1"
    }
}

response = requests.post(
    f"{base_url}/register_model",
    json=model_data,
    headers=headers
)

print(f"Model registered: {response.json()}")

# Submit diagnostic
diagnostic_data = {
    "case_id": "case_2024_001234",
    "model_id": "CardioNet_v2.1",
    "input_features": {"age": 65, "cholesterol": 220},
    "prediction": {"risk_category": "moderate"},
    "confidence_score": 0.89
}

response = requests.post(
    f"{base_url}/submit_diagnostic", 
    json=diagnostic_data,
    headers=headers
)

print(f"Diagnostic submitted: {response.json()}")
```

### cURL Examples
```bash
# Register model
curl -X POST "https://api.healthcare-ai-qa.example.com/api/v1/register_model" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "CardioNet_v2.1",
    "model_path": "/models/cardionet_v2.1.bin",
    "metadata": {
      "algorithm": "Deep Neural Network",
      "accuracy": 0.954,
      "version": "2.1"
    }
  }'

# Get audit trail
curl -X POST "https://api.healthcare-ai-qa.example.com/api/v1/get_audit_trail" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "CardioNet_v2.1", 
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-31T23:59:59Z"
  }'
```