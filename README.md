# Blockchain-Enabled QA and AI Models in Healthcare Diagnostics

Zenodo. https://zenodo.org/records/17115755

## Overview

This framework integrates blockchain (Hyperledger Fabric) and AI quality assurance for healthcare diagnostics.  
It includes model registration, diagnostic audit trails, explainability, compliance checks, simulation, and a REST API.

## System Architecture

See docs/ for complete UML diagrams and design artifacts.


## Directory Structure

- `framework/blockchain/chaincode.py` — Blockchain/chaincode simulation
- `framework/model_registry/registry.py` — Model registry
- `framework/decision_audit/audit.py` — Decision audit logging
- `framework/explainability/explain.py` — Model explainability (SHAP/LIME)
- `framework/regulatory/compliance.py` — Regulatory compliance logic
- `framework/simulation/simulator.py` — Simulation tools
- `framework/backend/api.py` — REST API (Flask)
- `docs/` — Documentation (UML, descriptions)
- `tests/` — Test scripts

## Usage

1. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server  
   ```bash
   python framework/backend/api.py
   ```

3. Use endpoints for model registration, diagnostics, audit, compliance, and simulation.

## Documentation

See `docs/` for UML diagrams, architecture, and roles.

---

Run the Flask API locally:
bash: python app.py

Test model registration and audit logging:
bash: curl -X POST http://localhost:5000/register_model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "lung_model_v4", "metadata": {"modality": "X-ray", "contains_phi": false}}'

Generate SHAP explanation:
bash: curl -X POST http://localhost:5000/generate_explanation \
  -H "Content-Type: application/json" \
  -d '{"model_id": "lung_model_v4", "method": "SHAP"}'

Access the dashboard: http://localhost:5000/dashboard

MIT License

Copyright (c) 2025 Sunil Chinnayyagari

Permission is hereby granted, free of charge, to any person obtaining a copy...

If you use this framework in your research, please cite:
Chinnayyagari, Sunil (2025). Blockchain-Enabled QA and AI Models in Healthcare Diagnostics (v1.0.0). 
Zenodo. https://zenodo.org/records/17115755

*This is an initial skeleton. Expand each module for full production.*
